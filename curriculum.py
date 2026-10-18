"""Distance-based teaching and equal-action-budget Q-learning experiments.

The teacher may inspect the maze to choose starts. The learner still receives
only (row, column) and reward transitions, never distances or a solution route.
"""

import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys

from baseline import BANGKOK, positive_integer, summarize
from bonuses import VisitCountBonus, nonnegative_beta
from environment import DEFAULT_MAZE, MazeEnv
from qlearning import QLearningAgent, environment_spec, epsilon_at, evaluate_greedy, unit_float
from train import write_csv


PROJECT_DIR = Path(__file__).resolve().parent
METHODS = ("plain", "curriculum")
BONUS_METHOD = "curriculum_bonus"


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class DistanceCurriculum:
    """Near → middle → real start; promotion uses training episodes only.

    A stage also has a training-action cap, checked at episode boundaries,
    to limit warm-up. Very small overall budgets can still end mid-curriculum.
    """

    def __init__(self, env, *, seed=0, easy_radius=4, medium_radius=10,
                 window=50, threshold=0.8, stage_cap=20000):
        for name, value in (("easy_radius", easy_radius), ("medium_radius", medium_radius),
                            ("window", window), ("stage_cap", stage_cap)):
            positive_int(value, name)
        threshold = unit_float(threshold, "threshold", positive=True)
        if medium_radius <= easy_radius:
            raise ValueError("medium_radius must exceed easy_radius")
        distances = env.distances_to_goal()
        near = tuple(sorted(p for p, distance in distances.items() if 1 <= distance <= easy_radius))
        middle = tuple(sorted(p for p, distance in distances.items() if easy_radius < distance <= medium_radius))
        if not near or not middle:
            raise ValueError("Both distance bands must have non-goal reachable starts")
        self.levels = (
            {"name": "near", "starts": near},
            {"name": "middle", "starts": middle},
            {"name": "real_start", "starts": (env.start,)},
        )
        self.index = 0
        self.stage_actions = 0
        self.window = window
        self.threshold = threshold
        self.stage_cap = stage_cap
        self.recent = deque(maxlen=window)
        self.rng = random.Random(f"curriculum-start-v1:{seed}")

    @property
    def name(self):
        return self.levels[self.index]["name"]

    def select_start(self):
        return self.rng.choice(self.levels[self.index]["starts"])

    def observe_episode(self, success, steps, *, complete=True):
        if not isinstance(success, bool) or not isinstance(complete, bool):
            raise ValueError("success and complete must be booleans")
        positive_int(steps, "steps")
        self.stage_actions += steps
        if not complete:
            return None
        self.recent.append(success)
        rate = sum(self.recent) / len(self.recent)
        threshold_met = len(self.recent) == self.window and rate >= self.threshold
        cap_met = self.stage_actions >= self.stage_cap
        if self.index == len(self.levels) - 1 or not (threshold_met or cap_met):
            return None
        event = {"from_stage": self.name, "stage_training_actions": self.stage_actions,
                 "recent_complete_episodes": len(self.recent), "training_success_rate": rate,
                 "reason": "success_threshold" if threshold_met else "stage_action_cap"}
        self.index += 1
        event["to_stage"] = self.name
        self.stage_actions = 0
        self.recent.clear()
        return event


def train_budget(seed=0, *, method="plain", budget=100000, max_steps=300,
                 alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05,
                 epsilon_decay=0.99995, eval_every=1000,
                 easy_radius=4, medium_radius=10, window=50, threshold=0.8,
                 stage_cap=None, maze=DEFAULT_MAZE, checkpoint=None, bonus_beta=0.5):
    """Exactly `budget` training actions, including collisions and warm-up.

    Evaluation uses a separate environment and no training RNG. A budget cut
    can end a partial episode, which is logged explicitly and cannot promote.
    """
    if method not in (*METHODS, BONUS_METHOD):
        raise ValueError("method must be plain, curriculum, or curriculum_bonus")
    bonus_beta = nonnegative_beta(bonus_beta)
    bonus = VisitCountBonus(bonus_beta) if method == BONUS_METHOD else None
    positive_int(budget, "budget")
    positive_int(eval_every, "eval_every")
    if stage_cap is None:
        stage_cap = max(1, budget // 5)
    positive_int(stage_cap, "stage_cap")
    schedule = {"start": epsilon_start, "minimum": epsilon_min, "decay": epsilon_decay}
    epsilon_at(0, **schedule)
    env, evaluator = MazeEnv(maze, max_steps), MazeEnv(maze, max_steps)
    distances = env.distances_to_goal()
    agent = QLearningAgent(env.rows, env.cols, seed=seed, alpha=alpha, gamma=gamma)
    teacher = (DistanceCurriculum(env, seed=seed, easy_radius=easy_radius,
                                 medium_radius=medium_radius, window=window,
                                 threshold=threshold, stage_cap=stage_cap)
               if method in ("curriculum", BONUS_METHOD) else None)
    snapshot_steps = {n for n in (0, 1000, 5000, 10000, 25000, 50000, budget) if n <= budget}
    training, evaluations, stages = [], [], []
    episode = 0

    def observe(episode_complete):
        stage = teacher.name if teacher is not None else "real_start"
        stats, _ = evaluate_greedy(agent, evaluator)
        evaluations.append({"method": method, "seed": seed, "training_env_steps": agent.updates,
                            "stage": stage, "epsilon": 0.0, **stats,
                            "intrinsic_reward": 0.0, "learning_reward": stats["env_reward"]})
        if agent.updates in snapshot_steps and checkpoint is not None:
            metadata = {"method": method, "training_env_steps": agent.updates,
                        "training_budget_actions": budget,
                        "episodes_completed": episode if episode_complete else max(0, episode - 1),
                        "current_episode_complete": episode_complete, "stage": stage,
                        "epsilon_next": epsilon_at(agent.updates, **schedule),
                        "epsilon_schedule_unit": "training_action", "epsilon_schedule": schedule,
                        "snapshot_selection": "predeclared_action_counts",
                        "exploration_bonus": bonus.snapshot() if bonus else None}
            checkpoint(agent, env, method, metadata)

    observe(True)
    while agent.updates < budget:
        episode += 1
        start = teacher.select_start() if teacher is not None else env.start
        stage = teacher.name if teacher is not None else "real_start"
        state, _ = env.reset(start=start)
        intrinsic_total = 0.0
        if bonus is not None:
            bonus.begin_episode(state)
        first_epsilon = epsilon_at(agent.updates, **schedule)
        while not env.done and agent.updates < budget:
            last_epsilon = epsilon_at(agent.updates, **schedule)
            action = agent.act(state, epsilon=last_epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            intrinsic = bonus.transition(state, next_state, terminated=terminated) if bonus else 0.0
            intrinsic_total += intrinsic
            agent.update(state, action, reward + intrinsic, next_state, terminated=terminated, truncated=truncated)
            state = next_state
            if agent.updates % eval_every == 0 or agent.updates in snapshot_steps:
                observe(env.done)
        cut = not env.done
        training.append({"method": method, "seed": seed, "episode": episode,
                         "stage": stage, "start_row": start[0], "start_col": start[1],
                         "start_distance": distances[start], "steps": env.steps,
                         "env_reward": env.total_reward, "intrinsic_reward": intrinsic_total,
                         "learning_reward": env.total_reward + intrinsic_total, "success": env.terminated,
                         "terminated": env.terminated, "truncated": env.truncated,
                         "budget_cutoff": cut, "episode_complete": env.done,
                         "wall_hits": env.wall_hits, "unique_cells": len(env.visited),
                         "epsilon_first": first_epsilon, "epsilon_last": last_epsilon,
                         "training_env_steps": agent.updates})
        if teacher is not None:
            event = teacher.observe_episode(env.terminated, env.steps, complete=env.done)
            if event is not None:
                stages.append({"method": method, "seed": seed, "episode": episode,
                               "training_env_steps": agent.updates, **event})
    return agent, training, evaluations, stages


def learning_milestones(evaluations):
    """Observed milestones on a fixed evaluation grid, not exact learning time."""
    first = next((row["training_env_steps"] for row in evaluations if row["success"]), None)
    stable = next((evaluations[i]["training_env_steps"] for i in range(len(evaluations) - 2)
                   if all(row["success"] for row in evaluations[i:i + 3])), None)
    return first, stable


def latest_comparison_model(results_dir, method):
    if method not in (*METHODS, BONUS_METHOD):
        raise ValueError("Unknown comparison method")
    # Prefer a phase-5 run for all three methods, so default demos are matched.
    manifests = sorted(Path(results_dir).glob("exploration-*/summary.json"), reverse=True)
    if method != BONUS_METHOD:
        manifests += sorted(Path(results_dir).glob("curriculum-*/summary.json"), reverse=True)
    for manifest in manifests:
        summary = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("Comparison summary must be an object")
        if summary.get("status") != "complete":
            continue
        model_map = summary.get("demo_models")
        if not isinstance(model_map, dict):
            raise ValueError("Comparison summary has no demo_models object")
        relative = model_map.get(method)
        if not isinstance(relative, str):
            raise ValueError("Comparison summary has no model for this method")
        root = manifest.parent.resolve()
        model = (root / relative).resolve()
        if not model.is_relative_to(root):
            raise ValueError("Model path must stay inside the comparison directory")
        if not model.is_file():
            raise FileNotFoundError(f"Missing comparison model: {model.name}")
        return model
    command = "exploration.cmd" if method == BONUS_METHOD else "curriculum.cmd or exploration.cmd"
    raise FileNotFoundError(f"No completed comparison. Run {command} first.")


def run_comparison(*, output_root, seeds=(0, 1, 2, 3, 4), budget=100000,
                   max_steps=300, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                   epsilon_min=0.05, epsilon_decay=0.99995, eval_every=1000,
                   easy_radius=4, medium_radius=10, window=50, threshold=0.8,
                   stage_cap=None, plot=True, include_bonus=False, bonus_beta=0.5):
    bonus_beta = nonnegative_beta(bonus_beta)
    methods = (*METHODS, BONUS_METHOD) if include_bonus else METHODS
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        raise ValueError("seeds must be unique integers")
    positive_int(budget, "budget")
    positive_int(eval_every, "eval_every")
    stage_cap = max(1, budget // 5) if stage_cap is None else stage_cap
    env = MazeEnv(max_steps=max_steps)
    QLearningAgent(env.rows, env.cols, alpha=alpha, gamma=gamma)
    epsilon_at(0, start=epsilon_start, minimum=epsilon_min, decay=epsilon_decay)
    teacher_preview = DistanceCurriculum(env, easy_radius=easy_radius, medium_radius=medium_radius,
                                         window=window, threshold=threshold, stage_cap=stage_cap)
    created = datetime.now(BANGKOK)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = "exploration" if include_bonus else "curriculum"
    run_dir = output_root / f"{prefix}-{created:%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir()
    (run_dir / "models").mkdir()
    (run_dir / "logs").mkdir()
    training, evaluations, transitions, finals = [], [], [], []

    def snapshot(agent, current_env, method, metadata):
        agent.save(run_dir / "models" / f"{method}-seed-{agent.seed}-step-{agent.updates}.json",
                   current_env, metadata=metadata)

    for method in methods:
        for seed in seeds:
            agent, train_rows, eval_rows, stage_rows = train_budget(
                seed, method=method, budget=budget, max_steps=max_steps,
                alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, eval_every=eval_every,
                easy_radius=easy_radius, medium_radius=medium_radius,
                window=window, threshold=threshold, stage_cap=stage_cap, checkpoint=snapshot,
                bonus_beta=bonus_beta)
            training.extend(train_rows)
            evaluations.extend(eval_rows)
            transitions.extend(stage_rows)
            path = run_dir / "models" / f"{method}-seed-{seed}-step-{budget}.json"
            loaded = QLearningAgent.load(path, env)
            stats, _ = evaluate_greedy(loaded, env)
            if loaded.q != agent.q or stats != {key: eval_rows[-1][key] for key in stats}:
                raise RuntimeError("Saved model does not reproduce its final evaluation")
            first, stable = learning_milestones(eval_rows)
            last_stage = train_rows[-1]["stage"]
            finals.append({"method": method, "seed": seed, "training_env_steps": agent.updates,
                           "training_episodes": len(train_rows), "last_training_stage": last_stage,
                           "first_observed_success_step": first,
                           "three_point_success_start_step": stable,
                           "model": path.relative_to(run_dir).as_posix(), **stats})
            print(f"{method}, seed {seed}: {agent.updates} training actions; "
                  f"first observed success={first}, final success={stats['success']}, "
                  f"steps={stats['steps']}, last stage={last_stage}", flush=True)

    comparison = []
    for method in methods:
        selected = [row for row in finals if row["method"] == method]
        first_steps = [row["first_observed_success_step"] for row in selected
                       if row["first_observed_success_step"] is not None]
        stable_steps = [row["three_point_success_start_step"] for row in selected
                        if row["three_point_success_start_step"] is not None]
        comparison.append({"method": method, **summarize(selected),
                           "training_actions_per_seed": budget,
                           "models_with_observed_success": len(first_steps),
                           "mean_first_observed_success_step": statistics.mean(first_steps) if first_steps else None,
                           "models_with_three_point_success": len(stable_steps),
                           "mean_three_point_success_start_step": statistics.mean(stable_steps) if stable_steps else None})
    summary = {
        "schema_version": 1, "status": "complete", "created_at": created.isoformat(),
        "environment": environment_spec(env),
        "config": {"seeds": list(seeds), "methods": list(methods),
                   "bonus_beta": bonus_beta if include_bonus else 0,
                   "bonus_method": BONUS_METHOD if include_bonus else None,
                   "bonus_rule": VisitCountBonus.RULE if include_bonus else None,
                   "training_actions_per_seed": budget, "budget_unit": "training_action",
                   "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
                   "epsilon_min": epsilon_min, "epsilon_decay_per_action": epsilon_decay,
                   "evaluation_every_actions": eval_every,
                   "evaluation_additional_steps": sorted({n for n in (0, 1000, 5000, 10000, 25000, 50000, budget) if n <= budget}),
                   "easy_radius": easy_radius, "medium_radius": medium_radius,
                   "promotion_window_complete_episodes": window,
                   "promotion_training_success_threshold": threshold,
                   "warmup_stage_action_cap": stage_cap,
                   "stage_cap_checked_at_episode_boundary": True,
                   "epsilon_resets_on_promotion": False, "action_masking": False,
                   "evaluation_epsilon": 0, "evaluation_updates": False,
                   "evaluation_start": list(env.start), "checkpoint_selection": "final_budget",
                   "teacher_has_distance_information": True,
                   "agent_receives_distance_or_route": False},
        "curriculum_stages": [{"name": level["name"], "starts": [list(p) for p in level["starts"]]}
                              for level in teacher_preview.levels],
        "comparison": comparison, "per_seed": finals,
        "stage_transitions": transitions,
        "demo_models": {method: next(row["model"] for row in finals if row["method"] == method and row["seed"] == seeds[0])
                        for method in methods},
        "demo_selection": "first_requested_seed_not_best_score",
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
                          for name in ("environment.py", "qlearning.py", "curriculum.py", "train.py", "baseline.py",
                                       "bonuses.py", "exploration.py", "exploration_report.py")},
        "learning_curve": None,
    }
    write_csv(run_dir / "logs/training.csv", training)
    write_csv(run_dir / "logs/evaluation.csv", evaluations)
    write_csv(run_dir / "logs/final_evaluation.csv", finals)
    stage_fields = ("method", "seed", "episode", "training_env_steps", "from_stage", "stage_training_actions",
                    "recent_complete_episodes", "training_success_rate", "reason", "to_stage")
    # Tiny smoke runs may not promote at all; keep a useful header-only file.
    if transitions:
        write_csv(run_dir / "logs/stages.csv", transitions, stage_fields)
    else:
        with (run_dir / "logs/stages.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            import csv
            csv.DictWriter(stream, fieldnames=stage_fields).writeheader()
    write_csv(run_dir / "comparison.csv", comparison)
    if plot:
        summary["learning_curve"] = plot_learning_curve(evaluations, run_dir, methods=methods)
    if include_bonus:
        from exploration_report import write_bonus_report
        write_bonus_report(run_dir, summary, training)
    else:
        write_comparison_report(run_dir, summary)
    # A summary manifest is published last, so the GUI only sees completed runs.
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return run_dir, summary


def plot_learning_curve(evaluations, run_dir, *, methods=METHODS):
    """Optional static research plot; training and GUI stay standard-library only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"plain": "#2864b5", "curriculum": "#bc5918", BONUS_METHOD: "#14845c"}
    names = {"plain": "Q-learning", "curriculum": "Q-learning + curriculum",
             BONUS_METHOD: "Q-learning + curriculum + bonus"}
    styles = {"plain": "-", "curriculum": "--", BONUS_METHOD: ":"}
    maximum = max(row["training_env_steps"] for row in evaluations)
    # Both the full prescribed budget and a clearly labelled early view are shown.
    for axis in axes:
        for method in methods:
            selected = [row for row in evaluations if row["method"] == method]
            steps = sorted({row["training_env_steps"] for row in selected})
            rates = [100 * statistics.mean(row["success"] for row in selected if row["training_env_steps"] == step)
                     for step in steps]
            axis.plot(steps, rates, label=names[method], color=colors[method], linewidth=2,
                      linestyle=styles[method])
        axis.set_xlabel("Training actions (wall hits and warm-up included)")
        axis.set_ylabel("Greedy success across trained seeds (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=.22)
    axes[0].set_xlim(0, maximum)
    axes[0].set_title("Full equal-action budget")
    axes[1].set_xlim(0, max(1, maximum // 5))
    axes[1].set_title("Early learning (first 20% of budget)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Greedy evaluation: same maze, real start and game rewards")
    path = run_dir / "learning_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def write_comparison_report(run_dir, result):
    cfg = result["config"]
    plain, curriculum = result["comparison"]
    def fmt(value):
        return "N/A" if value is None else f"{value:,.2f}"
    model_count = len(cfg["seeds"])
    first_plain = plain["mean_first_observed_success_step"]
    first_curriculum = curriculum["mean_first_observed_success_step"]
    if (plain["models_with_observed_success"] == model_count
            and curriculum["models_with_observed_success"] == model_count):
        difference = first_curriculum - first_plain
        if difference > 0:
            finding = (f"ในการตั้งค่าชุดนี้ Curriculum ไม่ได้ช่วยให้พบความสำเร็จเร็วขึ้น "
                       f"โดยใช้ก้าวฝึกถึงจุดที่สังเกตพบ success ครั้งแรกเฉลี่ยมากกว่า {difference:,.0f} ก้าว "
                       "จึงไม่ควรสรุปว่าการเพิ่ม Curriculum ทำให้ดีขึ้นเสมอ")
        elif difference < 0:
            finding = (f"ในการตั้งค่าชุดนี้ Curriculum พบ success ครั้งแรกเร็วกว่าเฉลี่ย {-difference:,.0f} ก้าวฝึก "
                       "เป็นข้อสังเกตจาก seeds และแผนที่ชุดนี้ ไม่ใช่ข้อพิสูจน์ว่าเร็วกว่าในทุกโจทย์")
        else:
            finding = "ในการตั้งค่าชุดนี้เวลาเฉลี่ยถึงจุดที่พบ success ครั้งแรกเท่ากัน จึงยังไม่เห็นข้อได้เปรียบด้านนี้ของ Curriculum"
    else:
        finding = "บาง seeds ยังไม่พบ success ภายในงบ จึงต้องอ่านจำนวนโมเดลที่สำเร็จร่วมกับค่าเฉลี่ยเวลา และไม่เปรียบเทียบค่าเฉลี่ยแบบตัดรอบล้มเหลวออกอย่างเดียว"
    rows = "\n".join(
        f"| {row['method']} | {row['seed']} | {row['training_env_steps']:,} | "
        f"{fmt(row['first_observed_success_step'])} | {'สำเร็จ' if row['success'] else 'หมดก้าว'} | "
        f"{row['steps']} | {row['last_training_stage']} |" for row in result["per_seed"])
    stage_rows = "\n".join(
        f"| {row['seed']} | {row['from_stage']} → {row['to_stage']} | {row['training_env_steps']:,} | "
        f"{row['training_success_rate']:.0%} / {row['recent_complete_episodes']} รอบ | {row['reason']} |"
        for row in result["stage_transitions"])
    figure = f"![Learning curve]({result['learning_curve']})" if result["learning_curve"] else "กราฟไม่ได้สร้าง ใช้ logs/evaluation.csv สำหรับวาดได้"
    report = f"""# Curriculum Learning — ผลระยะที่ 4

รันเมื่อ {result['created_at']} (UTC+07:00) เปิดหน้านี้เป็นรายงานหลักได้เลย ส่วน logs/ และ models/ เป็นหลักฐานประกอบ

## เปรียบเทียบด้วยงบเท่ากัน

| ตัวชี้วัด | Q-learning ปกติ | Q-learning + Curriculum |
|---|---:|---:|
| ก้าวฝึกต่อ seed รวม warm-up และชนกำแพง | {plain['training_actions_per_seed']:,} | {curriculum['training_actions_per_seed']:,} |
| โมเดลที่สำเร็จเมื่อจบงบ | {plain['successes']}/{model_count} | {curriculum['successes']}/{model_count} |
| Success rate สุดท้าย | {plain['success_rate_percent']:.2f}% | {curriculum['success_rate_percent']:.2f}% |
| ก้าวเฉลี่ยเมื่อทดสอบสุดท้าย รวมล้มเหลว | {plain['mean_steps_all']:.2f} | {curriculum['mean_steps_all']:.2f} |
| รางวัลเฉลี่ยเมื่อทดสอบสุดท้าย | {plain['mean_env_reward']:.2f} | {curriculum['mean_env_reward']:.2f} |
| ก้าวฝึกเฉลี่ยที่พบ greedy success ครั้งแรก* | {fmt(plain['mean_first_observed_success_step'])} | {fmt(curriculum['mean_first_observed_success_step'])} |
| โมเดลที่เคยพบ greedy success | {plain['models_with_observed_success']}/{model_count} | {curriculum['models_with_observed_success']}/{model_count} |
| ก้าวเริ่มต้นเฉลี่ยของ success 3 จุดประเมินติดกัน* | {fmt(plain['mean_three_point_success_start_step'])} | {fmt(curriculum['mean_three_point_success_start_step'])} |
| โมเดลที่มี success 3 จุดประเมินติดกัน | {plain['models_with_three_point_success']}/{model_count} | {curriculum['models_with_three_point_success']}/{model_count} |

*เป็นเวลาที่สังเกตพบจากกริดประเมิน ไม่ใช่ก้าวแรกที่เรียนรู้ได้อย่างแน่นอน ค่าเฉลี่ยคิดเฉพาะ seeds ที่พบเกณฑ์ ต้องอ่านคู่กับจำนวนโมเดลที่พบเกณฑ์ ไม่ใช่เวลารันเครื่องหรือจำนวน episode และ success 3 จุดติดกันไม่ได้รับประกันว่าจะสำเร็จตลอดจากนั้น

{finding}

สมมติฐานที่ควรตรวจในงานต่อไปคือเวลาที่ใช้ฝึกจากจุดเริ่มใกล้/กลางอาจไม่คุ้มสำหรับเขาวงกตขนาดนี้ แต่การรันนี้เพียงอย่างเดียวยังยืนยันสาเหตุไม่ได้ ไม่เปลี่ยนเงื่อนไขการฝึกภายหลังเพื่อให้ผลรันนี้ดูดีขึ้น

## กราฟการเรียนรู้จากจุดเริ่มจริง

{figure}

แกน X นับเฉพาะ action ฝึก รวมช่วงเริ่มใกล้เป้าหมาย การประเมินแยกต่างหากและไม่ถูกนับเป็นข้อมูลฝึก กราฟซ้ายแสดงเต็มงบ กราฟขวาขยาย 20% แรก

## Curriculum ทำอย่างไร

1. near: สุ่มจุดเริ่มที่ระยะทางเดินสั้นสุดถึงเป้าหมาย 1–{cfg['easy_radius']} ช่อง
2. middle: สุ่มจุดเริ่มที่ระยะ {cfg['easy_radius'] + 1}–{cfg['medium_radius']} ช่อง
3. real_start: กลับไปเริ่มจาก S เดิมทุก episode

เลื่อนระดับเมื่อสำเร็จอย่างน้อย {cfg['promotion_training_success_threshold']:.0%} ใน {cfg['promotion_window_complete_episodes']} episode ฝึกล่าสุดของระดับนั้น หรือเมื่อใช้งบในระดับ warm-up ครบ {cfg['warmup_stage_action_cap']:,} action โดยตรวจ cap ที่ขอบ episode จึงอาจเกิน cap ได้ไม่เกินขนาดหนึ่ง episode ทั้งหมดนับรวมอยู่ในงบหลักเท่าเดิม ไม่มี reset epsilon เมื่อเลื่อนระดับ

ผู้สอนใช้ BFS อ่านแผนที่เพื่อจัดกลุ่มจุดเริ่มตามระยะทางจริงที่หลบกำแพง นี่เป็นข้อมูลเพิ่มเติมที่ Curriculum ใช้เลือกบทเรียน Agent ไม่ได้รับระยะทาง แผนที่ หรือเส้นทางเฉลยเป็น observation และไม่มีโบนัสสำรวจในระยะนี้

ทั้งสองวิธีใช้ alpha={cfg['alpha']}, gamma={cfg['gamma']} และ epsilon ตามจำนวน action ฝึก: เริ่ม {cfg['epsilon_start']} คูณ {cfg['epsilon_decay_per_action']} ต่อ action ต่ำสุด {cfg['epsilon_min']} วิธีนี้ต่างจากระยะที่ 3 ที่ลด epsilon ต่อ episode จึงฝึก plain ใหม่พร้อมกันและไม่เทียบความเร็วกับไฟล์เก่าโดยตรง

## ผลแยก seed

| วิธี | Seed | Training actions | พบสำเร็จครั้งแรก* | ผลสุดท้าย | ก้าวสุดท้าย | ระดับฝึกท้ายสุด |
|---|---:|---:|---:|---|---:|---|
{rows}

## ประวัติเลื่อนระดับจากการฝึก

| Seed | ระดับ | Training actions สะสม | Training success / จำนวนรอบ | เหตุผล |
|---|---|---:|---|---|
{stage_rows if stage_rows else '| — | ยังไม่เลื่อนระดับในงบนี้ | — | — | — |'}

## ตอนโชว์ในคลาส

เปิด play.cmd เลือก **Q-learning** หรือ **Curriculum** แล้วกด **ให้ Q เดิน** โมเดลเริ่มต้นทั้งสองมาจาก seed แรกที่กำหนดไว้ล่วงหน้า ภายใต้งบฝึกเท่ากัน ไม่ได้เลือก seed ที่ดีที่สุด สามารถโหลด snapshot ใน models/ เพื่อดูผลระหว่างฝึกที่จำนวน action ตรงกัน

นำเสนอหน้าจอเดโมกับรายงานนี้พอ ไฟล์ comparison.csv มีตารางย่อ ส่วนข้อมูลทุก episode/จุดประเมิน/การเลื่อนระดับอยู่ใน logs/

## ขอบเขตและความยุติธรรมในการทดลอง

- ฝึกและประเมินบนแผนที่เดียว เป้าหมายเดิม ประเมินจาก S จริงเสมอ ไม่มีการใช้จุดเริ่มใกล้เป้าหมายทำให้คะแนนประเมินสูงขึ้น
- ทดสอบสุดท้ายหนึ่ง rollout ต่อ training seed เพราะ greedy policy และ Environment เป็น deterministic การทดสอบซ้ำโมเดลเดิมไม่ใช่ข้อมูลอิสระเพิ่ม
- ทั้งสองเริ่ม Q=0, ใช้ seeds {cfg['seeds']} และ Q-learning update/รางวัล/ชุด action เดียวกัน เลือกทิศชนกำแพงได้
- กำหนดงบ schedule เงื่อนไขเลื่อนระดับ และ checkpoint ไว้ล่วงหน้า การประเมินไม่อัปเดต Q/RNG ไม่เลื่อนระดับ ไม่ใช้หยุดฝึกหรือเลือกโมเดลที่ดีที่สุด
- episode ที่ถูกตัดกลางทางเพราะงบหลักหมดแยกเป็น budget_cutoff ไม่ถือว่าถึงเป้าหมายหรือสำเร็จจากเวลาหมด และไม่ใช้ตัดสินเลื่อนระดับ
- teacher มีความรู้แผนที่สำหรับเลือกจุดเริ่ม จึงไม่ควรอ้างว่าทั้งสองวิธีได้รับข้อมูลเท่ากันทุกด้าน แม้มีงบ action ฝึกเท่ากัน
- ผลไม่ได้พิสูจน์ว่า Curriculum ดีกว่าในทุกโจทย์ และไม่ได้ทดสอบเขาวงกตใหม่

## ทำซ้ำ

```powershell
python curriculum.py --budget {cfg['training_actions_per_seed']} --seeds {' '.join(map(str, cfg['seeds']))} --max-steps {result['environment']['max_steps']} --eval-every {cfg['evaluation_every_actions']} --alpha {cfg['alpha']} --gamma {cfg['gamma']} --epsilon-start {cfg['epsilon_start']} --epsilon-min {cfg['epsilon_min']} --epsilon-decay {cfg['epsilon_decay_per_action']} --easy-radius {cfg['easy_radius']} --medium-radius {cfg['medium_radius']} --window {cfg['promotion_window_complete_episodes']} --threshold {cfg['promotion_training_success_threshold']} --stage-cap {cfg['warmup_stage_action_cap']}
```

ใช้ --no-plot ได้หากไม่มี Matplotlib; ส่วนฝึกและหน้าจอหลักใช้ standard library เท่านั้น รันใหม่สร้างโฟลเดอร์ใหม่ ไม่ทับหลักฐานเดิม
"""
    with (run_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(report)


def main(argv=None, *, include_bonus=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=positive_integer, default=100000, help="training actions PER method PER seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=positive_integer, default=300)
    parser.add_argument("--eval-every", type=positive_integer, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995, help="decay PER TRAINING ACTION")
    parser.add_argument("--easy-radius", type=positive_integer, default=4)
    parser.add_argument("--medium-radius", type=positive_integer, default=10)
    parser.add_argument("--window", type=positive_integer, default=50)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--stage-cap", type=positive_integer, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--no-plot", action="store_true")
    if include_bonus:
        parser.add_argument("--bonus-beta", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        run_dir, summary = run_comparison(
            output_root=args.output_dir, seeds=args.seeds, budget=args.budget,
            max_steps=args.max_steps, alpha=args.alpha, gamma=args.gamma,
            epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay, eval_every=args.eval_every,
            easy_radius=args.easy_radius, medium_radius=args.medium_radius,
            window=args.window, threshold=args.threshold, stage_cap=args.stage_cap,
            plot=not args.no_plot, include_bonus=include_bonus,
            bonus_beta=args.bonus_beta if include_bonus else 0.5)
        print(f"Results: {run_dir}")
        print("Open summary.md for the comparison and presentation notes.")
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
