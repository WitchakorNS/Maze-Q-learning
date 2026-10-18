"""Train tabular Q-learning, save snapshots, and compare frozen policies to random."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from baseline import BANGKOK, EPISODE_FIELDS, evaluate_random, positive_integer, summarize
from environment import MazeEnv
from qlearning import QLearningAgent, environment_spec, epsilon_at, evaluate_greedy, train_one


PROJECT_DIR = Path(__file__).resolve().parent


def write_csv(path, records, fieldnames=None):
    if not records:
        raise ValueError("Cannot write an empty experiment log")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def run_experiment(*, output_root, seeds=(0, 1, 2, 3, 4), episodes=5000,
                   max_steps=300, alpha=0.1, gamma=0.95, epsilon_start=1.0,
                   epsilon_min=0.05, epsilon_decay=0.995, eval_interval=100,
                   random_episodes=200):
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        raise ValueError("seeds must be unique integers")
    for name, value in (("episodes", episodes), ("eval_interval", eval_interval),
                        ("random_episodes", random_episodes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    env = MazeEnv(max_steps=max_steps)
    QLearningAgent(env.rows, env.cols, alpha=alpha, gamma=gamma)
    epsilon_at(0, start=epsilon_start, minimum=epsilon_min, decay=epsilon_decay)
    created = datetime.now(BANGKOK)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"qlearning-{created:%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir()
    models = run_dir / "models"
    models.mkdir()
    training, evaluations, finals, demo_path = [], [], [], None

    def save_checkpoint(agent, current_env, episode, metadata):
        agent.save(models / f"seed-{agent.seed}-ep-{episode}.json", current_env, metadata=metadata)

    for seed in seeds:
        agent, train_rows, eval_rows = train_one(
            seed, episodes=episodes, max_steps=max_steps, alpha=alpha, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, eval_interval=eval_interval, checkpoint=save_checkpoint)
        training.extend(train_rows)
        evaluations.extend(eval_rows)
        # Evaluate the saved final checkpoint, not a model chosen by its score.
        model_path = models / f"seed-{seed}-ep-{episodes}.json"
        loaded = QLearningAgent.load(model_path, env)
        stats, trajectory = evaluate_greedy(loaded, env)
        if stats != {key: eval_rows[-1][key] for key in stats} or loaded.q != agent.q:
            raise RuntimeError("Saved checkpoint does not reproduce final evaluation")
        finals.append({"method": "q_learning", "seed": seed, "checkpoint_episode": episodes,
                       "training_env_steps": agent.updates, "epsilon": 0.0,
                       "model": model_path.relative_to(run_dir).as_posix(), **stats})
        if seed == seeds[0]:
            demo_path = trajectory
        print(f"seed {seed}: trained {episodes} episodes / {agent.updates} actions; "
              f"greedy success={stats['success']}, steps={stats['steps']}", flush=True)

    random_rows, random_summary = evaluate_random(random_episodes, seeds, max_steps)
    learned_stats = summarize(finals)
    # One frozen deterministic rollout per trained seed, not repeated trials
    # of the same policy on the same fixed start.
    successes = [100.0 if row["success"] else 0.0 for row in finals]
    learned_stats["seed_success_rate_sample_std_percent"] = (
        statistics.stdev(successes) if len(successes) > 1 else None)
    summary = {
        "schema_version": 1, "status": "complete", "method": "q_learning",
        "created_at": created.isoformat(), "environment": environment_spec(env),
        "training": {"seeds": list(seeds), "episodes_per_seed": episodes,
                     "alpha": alpha, "gamma": gamma,
                     "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
                     "epsilon_decay": epsilon_decay, "eval_interval": eval_interval,
                     "total_training_env_steps": sum(row["training_env_steps"] for row in finals),
                     "budget_unit": "episodes", "action_masking": False,
                     "truncation_bootstraps": True, "checkpoint_selection": "last_prescribed_episode",
                     "training_rng_restored_on_load": False},
        "evaluation": {"epsilon": 0.0, "updates": False,
                       "greedy_ties": "first_action_in_UP_RIGHT_DOWN_LEFT_order",
                       "rollouts_per_trained_seed": 1,
                       "fixed_start_goal_map": True,
                       "random_episodes_per_seed": random_episodes},
        "q_learning": learned_stats, "random": random_summary["aggregate"],
        "per_trained_seed": finals, "random_per_seed": random_summary["per_seed"],
        "demo_model": finals[0]["model"],
        "demo_selection": "first_requested_seed_not_best_score",
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
                          for name in ("environment.py", "agents.py", "baseline.py", "qlearning.py", "train.py")},
    }
    write_csv(run_dir / "training.csv", training)
    write_csv(run_dir / "evaluation.csv", evaluations)
    write_csv(run_dir / "final_evaluation.csv", finals)
    write_csv(run_dir / "random_evaluation.csv", random_rows, EPISODE_FIELDS)
    write_csv(run_dir / "demo_path.csv", demo_path)
    comparison = []
    for method, stats, cost in (("random", summary["random"], 0),
                                ("q_learning", learned_stats, summary["training"]["total_training_env_steps"])):
        comparison.append({"method": method, "evaluation_rollouts": stats["episodes"],
                           "successes": stats["successes"],
                           "success_rate_percent": stats["success_rate_percent"],
                           "mean_steps_all": stats["mean_steps_all"],
                           "mean_steps_success": stats["mean_steps_success"],
                           "mean_env_reward": stats["mean_env_reward"],
                           "mean_wall_hits": stats["mean_wall_hits"], "training_env_steps": cost})
    write_csv(run_dir / "comparison.csv", comparison)
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    write_report(run_dir, summary)
    return run_dir, summary


def write_report(run_dir, result):
    learned, random = result["q_learning"], result["random"]
    config = result["training"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.2f}"
    per_seed = "\n".join(
        f"| {row['seed']} | {row['training_env_steps']:,} | {'สำเร็จ' if row['success'] else 'หมดก้าว'} | "
        f"{row['steps']} | {row['env_reward']} | {row['wall_hits']} |" for row in result["per_trained_seed"])
    report = f"""# Q-learning — ผลระยะที่ 3

รันเมื่อ {result['created_at']} (UTC+07:00) ฝึกจากประสบการณ์ใน Environment โดยไม่ได้ให้เส้นทางเฉลย

## วิธีทดลอง

- แผนที่ 10×10 จุดเริ่ม (1,1) เป้าหมาย (8,8) จำกัด {result['environment']['max_steps']} action ต่อรอบ
- กติกาเดิม: เดิน -1 / ชนกำแพง -5 / ถึงเป้าหมาย +100; ทั้งสองวิธีเลือกได้ครบ 4 ทิศทาง
- Q-learning ฝึก {config['episodes_per_seed']:,} episode ต่อ seed; seeds {config['seeds']}
- alpha={config['alpha']}, gamma={config['gamma']}; epsilon เริ่ม {config['epsilon_start']} คูณ {config['epsilon_decay']} หลังแต่ละ episode และไม่ต่ำกว่า {config['epsilon_min']}
- ใช้ค่า Q เริ่มต้น 0 ทุกช่อง สุ่มเมื่อสำรวจ และสุ่มแก้ tie ขณะฝึก
- เมื่อถึงเป้าหมาย target คือ reward; เมื่อหมดเวลาให้ bootstrap ต่อ เพราะเวลาจำกัดเป็นเงื่อนไขภายนอกของ MDP ที่มีเพียงตำแหน่งเป็น state
- ทดสอบด้วย Q-table ที่บันทึกแล้ว: epsilon=0, ไม่อัปเดตค่า Q และแก้ tie ด้วยลำดับ action คงที่
- ใช้โมเดลสุดท้ายตามงบที่กำหนดล่วงหน้า ไม่มี early stopping หรือเลือก checkpoint ที่คะแนนดีที่สุด

## เปรียบเทียบผลจริง

| ตัวชี้วัด | Random Agent | Q-learning หลังฝึก |
|---|---:|---:|
| จำนวน rollout ประเมิน | {random['episodes']:,} | {learned['episodes']} (หนึ่งครั้งต่อโมเดล) |
| ถึงเป้าหมาย | {random['successes']} | {learned['successes']} |
| Success rate | {random['success_rate_percent']:.2f}% | {learned['success_rate_percent']:.2f}% |
| ก้าวเฉลี่ยรวมรอบล้มเหลว | {random['mean_steps_all']:.2f} | {learned['mean_steps_all']:.2f} |
| ก้าวเฉลี่ยเฉพาะรอบสำเร็จ | {fmt(random['mean_steps_success'])} | {fmt(learned['mean_steps_success'])} |
| รางวัลเฉลี่ย | {random['mean_env_reward']:.2f} | {learned['mean_env_reward']:.2f} |
| ชนกำแพงเฉลี่ย | {random['mean_wall_hits']:.2f} | {learned['mean_wall_hits']:.2f} |
| จำนวน action สำหรับฝึก | 0 | {config['total_training_env_steps']:,} |

Random รัน {result['evaluation']['random_episodes_per_seed']} episode ต่อ seed ส่วน Q-learning ประเมินหนึ่ง rollout ต่อโมเดลจาก training seeds ที่ต่างกัน เพราะการเล่นของนโยบาย greedy บนแผนที่/จุดเริ่มเดิมเป็น deterministic การเล่นโมเดลเดิมซ้ำหลายรอบไม่เพิ่มหลักฐานอิสระ

## ผลแยก training seed

| Seed | Training actions | ผลประเมิน | ก้าว | รางวัล | ชนกำแพง |
|---|---:|---|---:|---:|---:|
{per_seed}

## หลักฐานและเดโม

- training.csv: ทุก episode ระหว่างฝึก รวม epsilon และ training actions สะสม
- evaluation.csv: ประเมิน greedy ระหว่างฝึกด้วย Environment แยก ไม่ปะปนกับ training reward
- final_evaluation.csv: ผลหลังโหลด checkpoint สุดท้ายของแต่ละ seed
- random_evaluation.csv: baseline ที่รันด้วยกติกาเดียวกันในรอบทดลองนี้
- comparison.csv: ตัวชี้วัดเปรียบเทียบพร้อมต้นทุนฝึก
- models/: Q-table JSON ก่อนฝึก และหลัง episode 10, 100, 500, 1000 และตอนจบ (เฉพาะจุดที่ไม่เกินงบ)
- demo_path.csv: เส้นทางของโมเดลเดโมที่กำหนดเป็น seed แรกไว้ล่วงหน้า
- โมเดลเดโม: `{result['demo_model']}`

เปิด play.cmd แล้วกด **ให้ Q เดิน** เพื่อใช้โมเดลจากรันล่าสุดที่เสร็จสมบูรณ์ (เมื่อยังไม่ได้เลือกโมเดลเอง) หรือกด **โหลด Q-table** เพื่อเลือก snapshot ก่อน/ระหว่างฝึก สามารถพักและปรับความเร็วได้ การเดโมไม่อัปเดตตาราง

## ขอบเขตของข้อสรุป

ผลนี้แสดงการเรียนรู้บนแผนที่และจุดเริ่มที่ใช้ฝึก ยังไม่ใช่การทดสอบแผนที่ใหม่หรือการรับประกันความสำเร็จจากทุกช่อง ไม่ใช้ข้อมูลประเมินเพื่อปรับพารามิเตอร์ในการรันนี้ และยังไม่ได้ใช้ Curriculum หรือ Exploration Bonus

งบที่ใช้ในระยะนี้กำหนดเป็นจำนวน episode จึงต้องดูจำนวน action ฝึกจริงด้วย ก่อนเปรียบเทียบกับวิธีปรับปรุงในระยะถัดไปควรกำหนดงบเป็นจำนวน action เท่ากัน

ไฟล์ Q-table ใช้โหลดเพื่อประเมินและสาธิตได้ แต่ไม่ได้เก็บสถานะ RNG สำหรับกลับมาฝึกต่อให้ตรงกับการฝึกต่อเนื่องทุกประการ

## ทำซ้ำ

```powershell
python train.py --episodes {config['episodes_per_seed']} --seeds {' '.join(map(str, config['seeds']))} --max-steps {result['environment']['max_steps']} --alpha {config['alpha']} --gamma {config['gamma']} --epsilon-start {config['epsilon_start']} --epsilon-min {config['epsilon_min']} --epsilon-decay {config['epsilon_decay']} --eval-interval {config['eval_interval']} --random-episodes {result['evaluation']['random_episodes_per_seed']}
```

ทุกครั้งสร้างโฟลเดอร์ใหม่ ไม่ทับผลก่อนหน้า ตรวจรุ่น Python และ source hash ได้ใน summary.json
"""
    with (run_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_integer, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=positive_integer, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--eval-interval", type=positive_integer, default=100)
    parser.add_argument("--random-episodes", type=positive_integer, default=200)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--evaluate", type=Path, metavar="QTABLE_JSON", help="load and evaluate only; no training")
    args = parser.parse_args(argv)
    try:
        if args.evaluate is not None:
            env = MazeEnv(max_steps=args.max_steps)
            agent = QLearningAgent.load(args.evaluate, env)
            stats, _ = evaluate_greedy(agent, env)
            print(json.dumps(stats, indent=2))
            print("epsilon=0; no Q updates; deterministic greedy evaluation")
        else:
            run_dir, summary = run_experiment(
                output_root=args.output_dir, seeds=args.seeds, episodes=args.episodes,
                max_steps=args.max_steps, alpha=args.alpha, gamma=args.gamma,
                epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay, eval_interval=args.eval_interval,
                random_episodes=args.random_episodes)
            print(f"Frozen Q-learning: {summary['q_learning']['successes']}/{len(args.seeds)} trained models succeeded")
            print(f"Random baseline: {summary['random']['success_rate_percent']:.2f}% success")
            print(f"Results: {run_dir}")
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
