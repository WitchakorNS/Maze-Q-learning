"""Q targets, frozen evaluation, reproducibility, and persisted training evidence."""

import copy
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from environment import Action, MazeEnv
from qlearning import QLearningAgent, epsilon_at, evaluate_greedy, latest_demo_model, train_one
from train import run_experiment


class QLearningTest(unittest.TestCase):
    def test_nonterminal_update_uses_discounted_next_max(self):
        agent = QLearningAgent(1, 2, alpha=0.5, gamma=0.9)
        agent.q[0][0][Action.RIGHT] = 2
        agent.q[0][1] = [4, 10, 0, -2]
        result = agent.update((0, 0), Action.RIGHT, -1, (0, 1), terminated=False)
        self.assertAlmostEqual(result, 5)  # 2 + .5 * (-1 + .9*10 - 2)
        self.assertEqual(agent.updates, 1)

    def test_goal_does_not_bootstrap(self):
        agent = QLearningAgent(1, 2, alpha=0.5, gamma=0.9)
        agent.q[0][1] = [1000] * 4
        result = agent.update((0, 0), Action.RIGHT, 100, (0, 1), terminated=True)
        self.assertEqual(result, 50)

    def test_external_time_limit_does_bootstrap(self):
        agent = QLearningAgent(1, 2, alpha=0.5, gamma=0.9)
        agent.q[0][1] = [10] * 4
        result = agent.update((0, 0), Action.RIGHT, -1, (0, 1), terminated=False, truncated=True)
        self.assertEqual(result, 4)

    def test_frozen_greedy_does_not_mutate_table_counters_or_rng(self):
        env = MazeEnv(("SG",), max_steps=5)
        agent = QLearningAgent(1, 2, seed=9)
        before = (copy.deepcopy(agent.q), agent.updates, agent.rng.getstate())
        first, first_path = evaluate_greedy(agent, env)
        second, second_path = evaluate_greedy(agent, env)
        self.assertEqual((first, first_path), (second, second_path))
        self.assertEqual(before, (agent.q, agent.updates, agent.rng.getstate()))
        self.assertEqual(first["wall_hits"], 5)  # Tie → UP; not silently avoiding walls.
        self.assertFalse(first["success"])

    def test_exploration_uses_all_actions_and_private_seed(self):
        a, b = QLearningAgent(1, 2, seed=10), QLearningAgent(1, 2, seed=10)
        first = [a.act((0, 0), epsilon=1) for _ in range(100)]
        self.assertEqual(first, [b.act((0, 0), epsilon=1) for _ in range(100)])
        self.assertEqual(set(first), set(Action))

    def test_epsilon_schedule_and_validation(self):
        self.assertEqual(epsilon_at(0), 1)
        self.assertEqual(epsilon_at(1), .995)
        self.assertEqual(epsilon_at(100000), .05)
        for kwargs in ({"minimum": 1, "start": .1}, {"decay": 0}, {"start": float("nan")}):
            with self.assertRaises(ValueError):
                epsilon_at(0, **kwargs)

    def test_learns_a_small_maze_from_rewards(self):
        agent, training, evaluations = train_one(7, episodes=200, max_steps=20, maze=("S.G",))
        self.assertEqual(evaluations[0]["checkpoint_episode"], 0)
        self.assertFalse(evaluations[0]["success"])
        stats, path = evaluate_greedy(agent, MazeEnv(("S.G",), max_steps=20))
        self.assertTrue(stats["success"])
        self.assertEqual((stats["steps"], stats["env_reward"], stats["wall_hits"]), (2, 99, 0))
        self.assertEqual(agent.updates, sum(row["steps"] for row in training))
        self.assertEqual(len(path), 3)

    def test_evaluation_frequency_cannot_change_training(self):
        first, rows_a, _ = train_one(3, episodes=55, max_steps=20, maze=("S.G",), eval_interval=1)
        second, rows_b, _ = train_one(3, episodes=55, max_steps=20, maze=("S.G",), eval_interval=17)
        self.assertEqual(first.q, second.q)
        self.assertEqual(first.rng.getstate(), second.rng.getstate())
        self.assertEqual(rows_a, rows_b)

    def test_checkpoint_round_trip_and_wrong_rules_rejected(self):
        env = MazeEnv(("S.G",), max_steps=20)
        agent, _, _ = train_one(7, episodes=100, max_steps=20, maze=env.maze)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "q.json"
            agent.save(path, env, metadata={"episodes_completed": 100})
            loaded = QLearningAgent.load(path, env)
            self.assertEqual(loaded.q, agent.q)
            self.assertEqual(loaded.updates, agent.updates)
            self.assertEqual(evaluate_greedy(loaded, env), evaluate_greedy(agent, env))
            with self.assertRaises(FileExistsError):
                agent.save(path, env)
            with self.assertRaises(ValueError):
                QLearningAgent.load(path, MazeEnv(("S.G",), max_steps=30))
            with self.assertRaises(ValueError):
                QLearningAgent.load(path, MazeEnv(("SG.",), max_steps=20))

    def test_corrupt_checkpoint_rejected(self):
        env = MazeEnv(("SG",))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "q.json"
            QLearningAgent(1, 2).save(path, env)
            original = json.loads(path.read_text(encoding="utf-8"))
            for name in ("nan", "shape", "action_order", "counter", "metadata", "params"):
                invalid = copy.deepcopy(original)
                if name == "nan": invalid["q_table"][0][0][0] = float("nan")
                elif name == "shape": invalid["q_table"][0][0] = [0]
                elif name == "action_order": invalid["environment"]["actions"].reverse()
                elif name == "counter": invalid["updates"] = -1
                elif name == "metadata": invalid["metadata"] = []
                else: invalid["hyperparameters"]["alpha"] = 0
                damaged = Path(temp) / f"{name}.json"
                damaged.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    QLearningAgent.load(damaged, env)

    def test_invalid_agent_arguments(self):
        for params in ({"alpha": 0}, {"gamma": 2}, {"alpha": float("inf")}, {"seed": True}):
            with self.assertRaises(ValueError):
                QLearningAgent(2, 2, **params)
        agent = QLearningAgent(2, 2)
        with self.assertRaises(ValueError): agent.act((-1, 0))
        with self.assertRaises(ValueError): agent.act((0, 0), epsilon=-1)
        with self.assertRaises(ValueError): agent.update((0, 0), True, -1, (0, 1), terminated=False)
        with self.assertRaises(ValueError): agent.update((0, 0), 1, float("nan"), (0, 1), terminated=False)

    def test_experiment_logs_costs_snapshots_and_loaded_final(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            run_dir, summary = run_experiment(output_root=temp, seeds=(4, 8), episodes=12,
                                              max_steps=25, eval_interval=5, random_episodes=3)
            with (run_dir / "training.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(int(row["steps"]) for row in rows), summary["training"]["total_training_env_steps"])
            self.assertEqual(summary["q_learning"]["episodes"], 2)
            self.assertEqual(summary["random"]["episodes"], 6)
            self.assertEqual(summary["demo_model"], "models/seed-4-ep-12.json")
            self.assertEqual(latest_demo_model(temp), (run_dir / summary["demo_model"]).resolve())
            self.assertTrue((run_dir / "models/seed-4-ep-0.json").is_file())
            self.assertTrue((run_dir / "models/seed-8-ep-10.json").is_file())
            self.assertTrue((run_dir / "summary.md").is_file())
            stored = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, summary)
            loaded = QLearningAgent.load(run_dir / summary["demo_model"], MazeEnv(max_steps=25))
            stats, _ = evaluate_greedy(loaded, MazeEnv(max_steps=25))
            self.assertEqual(stats["steps"], summary["per_trained_seed"][0]["steps"])

    def test_latest_lookup_requires_complete_run_and_contained_model_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                latest_demo_model(temp)
            run = Path(temp) / "qlearning-20200101"
            run.mkdir()
            manifest = run / "summary.json"
            manifest.write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                latest_demo_model(temp)
            manifest.write_text(json.dumps({"status": "complete", "demo_model": "../outside.json"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                latest_demo_model(temp)
            manifest.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                latest_demo_model(temp)


if __name__ == "__main__":
    unittest.main()
