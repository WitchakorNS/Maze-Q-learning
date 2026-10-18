"""Reproducibility, episode accounting, and persisted baseline evidence."""

import csv
import json
import random
import tempfile
import unittest

from agents import RandomAgent
from baseline import evaluate_random, run_episode, write_results
from environment import Action, MazeEnv


class RandomBaselineTest(unittest.TestCase):
    def test_private_rng_is_reproducible_and_ignores_observation(self):
        random.seed(991)
        global_state = random.getstate()
        first, second = RandomAgent(42), RandomAgent(42)
        actions = [first.act((1, 1)) for _ in range(100)]
        self.assertEqual(actions, [second.act((8, 8)) for _ in range(100)])
        self.assertEqual(set(actions), set(Action))
        self.assertEqual(random.getstate(), global_state)

    def test_whole_run_is_reproducible(self):
        first = evaluate_random(episodes=12, seeds=(7, 9), max_steps=30)
        second = evaluate_random(episodes=12, seeds=(7, 9), max_steps=30)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 24)

    def test_logged_episode_seed_can_replay_one_episode(self):
        records, _ = evaluate_random(episodes=4, seeds=(19, 20), max_steps=50)
        row = records[-1]
        replay = run_episode(MazeEnv(max_steps=50), RandomAgent(row["episode_seed"]),
                             row["seed"], row["episode"])
        self.assertEqual(replay, {key: value for key, value in row.items()
                                  if key not in ("seed_env_steps", "run_env_steps")})

    def test_each_seed_is_independent_of_run_order(self):
        forward, _ = evaluate_random(4, (1, 2), 20)
        reverse, _ = evaluate_random(4, (2, 1), 20)
        reverse_by_key = {(row["seed"], row["episode"]): row for row in reverse}
        for row in forward:
            other = reverse_by_key[row["seed"], row["episode"]]
            self.assertEqual({k: v for k, v in row.items() if k != "run_env_steps"},
                             {k: v for k, v in other.items() if k != "run_env_steps"})

    def test_reward_and_step_accounting_include_collisions_and_failures(self):
        # A tiny map exercises both successful and capped episodes.
        records, summary = evaluate_random(50, (0, 1), max_steps=3, maze=("SG",))
        self.assertGreater(summary["aggregate"]["successes"], 0)
        self.assertGreater(summary["aggregate"]["failures"], 0)
        total_steps = 0
        for row in records:
            self.assertTrue(row["terminated"] ^ row["truncated"])
            self.assertEqual(row["success"], row["terminated"])
            self.assertEqual(row["env_reward"], -row["steps"] - 4 * row["wall_hits"]
                             + (101 if row["success"] else 0))
            if row["success"]:
                self.assertEqual((row["final_row"], row["final_col"]), (0, 1))
            else:
                self.assertEqual(row["steps"], 3)
            total_steps += row["steps"]
            self.assertEqual(row["run_env_steps"], total_steps)
        self.assertEqual(summary["aggregate"]["total_env_steps"], total_steps)
        self.assertEqual(sum(row["total_env_steps"] for row in summary["per_seed"]), total_steps)

    def test_no_success_is_none_not_zero_and_single_seed_std_is_none(self):
        records, summary = evaluate_random(5, (0,), max_steps=1)
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["successes"], 0)
        self.assertEqual(aggregate["mean_steps_all"], 1)
        self.assertIsNone(aggregate["mean_steps_success"])
        self.assertIsNone(aggregate["seed_success_rate_sample_std_percent"])
        self.assertEqual(aggregate["failures"], len(records))

    def test_summary_uses_observed_successes_and_all_episodes(self):
        records, summary = evaluate_random(25, (3, 4), max_steps=8, maze=("SG",))
        aggregate = summary["aggregate"]
        successes = [row for row in records if row["success"]]
        self.assertEqual(aggregate["success_rate_percent"], 100 * len(successes) / len(records))
        self.assertEqual(aggregate["mean_steps_success"], sum(r["steps"] for r in successes) / len(successes))
        self.assertEqual(aggregate["mean_steps_all"], sum(r["steps"] for r in records) / len(records))

    def test_saved_csv_and_json_agree_and_keep_previous_run(self):
        records, summary = evaluate_random(3, (0, 1), max_steps=1)
        with tempfile.TemporaryDirectory() as temp:
            first = write_results(records, summary, temp)
            original = (first / "episodes.csv").read_bytes()
            second = write_results(records, summary, temp)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "episodes.csv").read_bytes(), original)
            with (first / "episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
                saved = list(csv.DictReader(stream))
            result = json.loads((first / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved), len(records))
            self.assertEqual(sum(int(row["steps"]) for row in saved), result["aggregate"]["total_env_steps"])
            self.assertEqual(sum(row["success"] == "True" for row in saved), result["aggregate"]["successes"])
            self.assertEqual(result["config"], summary["config"])
            self.assertEqual(set(result["source_sha256"]), {"environment.py", "agents.py", "baseline.py"})
            self.assertIn("N/A", (first / "summary.md").read_text(encoding="utf-8"))
            self.assertTrue((first / "per_seed.csv").is_file())

    def test_invalid_run_configuration(self):
        for episodes, seeds, max_steps in [(0, (1,), 30), (True, (1,), 30),
                                            (1, (), 30), (1, (1, 1), 30),
                                            (1, (True,), 30), (1, (1,), 0)]:
            with self.subTest(episodes=episodes, seeds=seeds, max_steps=max_steps):
                with self.assertRaises(ValueError):
                    evaluate_random(episodes, seeds, max_steps)


if __name__ == "__main__":
    unittest.main()
