"""Reproducible, headless evaluation of the random policy (standard library only)."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from agents import RandomAgent, episode_seed
from environment import DEFAULT_MAZE, MazeEnv


PROJECT_DIR = Path(__file__).resolve().parent
BANGKOK = timezone(timedelta(hours=7), name="Asia/Bangkok")
EPISODE_FIELDS = (
    "method", "seed", "episode", "episode_seed", "steps", "env_reward",
    "success", "terminated", "truncated", "end_reason", "wall_hits",
    "unique_cells", "start_row", "start_col", "goal_row", "goal_col",
    "final_row", "final_col", "seed_env_steps", "run_env_steps",
)


def run_episode(env, agent, seed, episode):
    """Run a complete episode, without rendering or mutating the policy."""
    state, _ = env.reset()
    while not env.done:
        state, _, _, _, info = env.step(agent.act(state))
    return {
        "method": "random",
        "seed": seed,
        "episode": episode,
        "episode_seed": episode_seed(seed, episode),
        "steps": env.steps,
        "env_reward": env.total_reward,
        "success": info["is_success"],
        "terminated": env.terminated,
        "truncated": env.truncated,
        "end_reason": "goal" if env.terminated else "step_limit",
        "wall_hits": env.wall_hits,
        "unique_cells": len(env.visited),
        "start_row": env.start[0],
        "start_col": env.start[1],
        "goal_row": env.goal[0],
        "goal_col": env.goal[1],
        "final_row": state[0],
        "final_col": state[1],
    }


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty run")
    successes = [row for row in records if row["success"]]
    steps = sum(row["steps"] for row in records)
    return {
        "episodes": len(records),
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "success_rate_percent": 100 * len(successes) / len(records),
        "total_env_steps": steps,
        "mean_steps_all": statistics.mean(row["steps"] for row in records),
        "mean_steps_success": statistics.mean(row["steps"] for row in successes) if successes else None,
        "mean_env_reward": statistics.mean(row["env_reward"] for row in records),
        "mean_wall_hits": statistics.mean(row["wall_hits"] for row in records),
        "wall_hit_rate_percent": 100 * sum(row["wall_hits"] for row in records) / steps,
        "mean_unique_cells": statistics.mean(row["unique_cells"] for row in records),
    }


def evaluate_random(episodes=200, seeds=(0, 1, 2, 3, 4), max_steps=300, maze=DEFAULT_MAZE):
    """Episodes per seed; one stream per episode makes single runs replayable."""
    seeds = tuple(seeds)
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty sequence of unique integers")
    env = MazeEnv(maze=maze, max_steps=max_steps)
    records, per_seed = [], []
    run_steps = 0
    for seed in seeds:
        seed_records = []
        seed_steps = 0
        for episode in range(1, episodes + 1):
            agent = RandomAgent(episode_seed(seed, episode))
            row = run_episode(env, agent, seed, episode)
            run_steps += row["steps"]
            seed_steps += row["steps"]
            row.update(seed_env_steps=seed_steps, run_env_steps=run_steps)
            records.append(row)
            seed_records.append(row)
        per_seed.append({"seed": seed, **summarize(seed_records)})

    rates = [row["success_rate_percent"] for row in per_seed]
    aggregate = summarize(records)
    aggregate["seed_success_rate_mean_percent"] = statistics.mean(rates)
    aggregate["seed_success_rate_sample_std_percent"] = statistics.stdev(rates) if len(rates) > 1 else None
    map_rows = list(env.maze)
    fingerprint = hashlib.sha256("\n".join(map_rows).encode("utf-8")).hexdigest()
    summary = {
        "schema_version": 1,
        "method": "random",
        "policy": {
            "sampling": "uniform_all_four_actions",
            "action_masking": False,
            "learning": False,
            "episode_seed_format": "maze-random-v1:{seed}:{episode}",
        },
        "config": {
            "episodes_per_seed": episodes,
            "seeds": list(seeds),
            "max_steps": max_steps,
            "start": list(env.start),
            "goal": list(env.goal),
            "maze": map_rows,
            "maze_sha256": fingerprint,
            "rewards": {"walk": -1, "wall": -5, "goal": 100},
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "scope": "Fixed start and goal on one fixed map; not unseen-maze generalization.",
    }
    return records, summary


def format_number(value):
    return "N/A (no successful episodes)" if value is None else f"{value:.2f}"


def write_results(records, summary, output_root):
    """Create a new run directory. Never replace a previous experiment."""
    created = datetime.now(BANGKOK)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"random-{created:%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir()
    result = dict(summary)
    result["created_at"] = created.isoformat()
    result["runtime"] = {"python": sys.version, "platform": platform.platform()}
    result["source_sha256"] = {
        name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
        for name in ("environment.py", "agents.py", "baseline.py")
    }
    with (run_dir / "episodes.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "per_seed.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["per_seed"][0].keys())
        writer.writeheader()
        writer.writerows(result["per_seed"])
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    stats = result["aggregate"]
    config = result["config"]
    seed_rows = "\n".join(
        f"| {row['seed']} | {row['episodes']} | {row['successes']} | "
        f"{row['success_rate_percent']:.2f}% | {row['mean_env_reward']:.2f} |"
        for row in result["per_seed"]
    )
    report = f"""# Random Agent baseline — ระยะที่ 2

ผลจริงจากการสุ่มเดิน ไม่มีการฝึกหรือการใช้เส้นทางเฉลย

- วันที่รัน: {result['created_at']} (UTC+07:00)
- แผนที่: {len(config['maze'])}×{len(config['maze'][0])} จุดเริ่ม {tuple(config['start'])} เป้าหมาย {tuple(config['goal'])}
- จำนวนรอบ: {stats['episodes']:,} ({config['episodes_per_seed']} รอบต่อ seed)
- Seeds: {', '.join(map(str, config['seeds']))}
- จำกัด {config['max_steps']} action ต่อรอบ รวมการชนกำแพง
- สุ่มทั้ง 4 ทิศทางเท่า ๆ กัน ไม่กรองทิศที่ชนกำแพง
- รางวัล: เดิน -1 / ชน -5 / ถึงเป้าหมาย +100

| ตัวชี้วัด | ผล |
|---|---:|
| รอบที่สำเร็จ | {stats['successes']} / {stats['episodes']} |
| รอบที่หมดก้าว | {stats['failures']} |
| Success rate | {stats['success_rate_percent']:.2f}% |
| จำนวนก้าวเฉลี่ยทุก episode (รวมล้มเหลว) | {stats['mean_steps_all']:.2f} |
| จำนวนก้าวเฉลี่ยเฉพาะ episode ที่สำเร็จ | {format_number(stats['mean_steps_success'])} |
| รางวัลจากเกมเฉลี่ย | {stats['mean_env_reward']:.2f} |
| จำนวนครั้งชนกำแพงเฉลี่ยต่อ episode | {stats['mean_wall_hits']:.2f} |
| สัดส่วน action ที่ชนกำแพง | {stats['wall_hit_rate_percent']:.2f}% |
| จำนวน action รวม | {stats['total_env_steps']:,} |

## ผลแยกแต่ละ seed

| Seed | Episodes | สำเร็จ | Success rate | รางวัลเฉลี่ย |
|---|---:|---:|---:|---:|
{seed_rows}

ค่าเฉลี่ย success rate ระหว่าง seeds: {stats['seed_success_rate_mean_percent']:.2f}%

ส่วนเบี่ยงเบนมาตรฐานแบบตัวอย่างระหว่าง seeds (หน่วย percentage points): {format_number(stats['seed_success_rate_sample_std_percent']) if stats['seed_success_rate_sample_std_percent'] is not None else 'N/A (one seed)'}

## การตีความและข้อจำกัด

เป็น baseline ของนโยบายสุ่มบนแผนที่ จุดเริ่ม เป้าหมาย และขีดจำกัดก้าวที่กำหนด ไม่ใช่ผลการเรียนรู้หรือการทดสอบเขาวงกตใหม่ จำนวนก้าวเฉพาะรอบที่สำเร็จต้องอ่านคู่กับ success rate เพื่อไม่ให้มองข้ามรอบที่ล้มเหลว

Seeds นี้ควบคุมการสุ่ม action ของ baseline ไม่ใช่ training seeds ไม่มีการเลือกโมเดลหรือปรับพารามิเตอร์จากการรันนี้

ไฟล์ episodes.csv เก็บทุก episode, per_seed.csv เก็บผลรวมราย seed และ summary.json เก็บแผนที่ กติกา รุ่น Python และ hash ของ source code เพื่อช่วยทำซ้ำ

## ทำซ้ำ

เปิด Terminal ในโฟลเดอร์ maze-rl:

```powershell
python baseline.py --episodes {config['episodes_per_seed']} --seeds {' '.join(map(str, config['seeds']))} --max-steps {config['max_steps']}
```

การรันซ้ำจะสร้างโฟลเดอร์ใหม่ ค่าของผลการเล่นควรเหมือนเดิมเมื่อใช้โค้ดและรุ่น Python เดียวกัน ส่วนเวลาและข้อมูล runtime อาจต่างกัน
"""
    with (run_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    return run_dir


def positive_integer(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_integer, default=200, help="episodes PER seed (default: 200)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=positive_integer, default=300)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    records, summary = evaluate_random(args.episodes, args.seeds, args.max_steps)
    run_dir = write_results(records, summary, args.output_dir)
    stats = summary["aggregate"]
    print(f"Random baseline: {stats['episodes']} episodes, {len(args.seeds)} seeds")
    print(f"Success: {stats['successes']}/{stats['episodes']} ({stats['success_rate_percent']:.2f}%)")
    print(f"Mean steps (all): {stats['mean_steps_all']:.2f}")
    print(f"Mean steps (successful only): {format_number(stats['mean_steps_success'])}")
    print(f"Mean environment reward: {stats['mean_env_reward']:.2f}")
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
