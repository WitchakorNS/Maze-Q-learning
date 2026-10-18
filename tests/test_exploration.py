"""Count semantics, zero-beta control, reward accounting, and frozen evaluation."""

import copy
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from bonuses import VisitCountBonus
from curriculum import latest_comparison_model, run_comparison, train_budget
from environment import MazeEnv
from qlearning import QLearningAgent, evaluate_greedy


class ExplorationTest(unittest.TestCase):
    def test_bonus_uses_prior_count_and_decreases(self):
        bonus = VisitCountBonus(.5)
        bonus.begin_episode((0, 0))
        for visit in range(1, 5):
            reward = bonus.transition((0, 0), (0, 1), terminated=False)
            self.assertAlmostEqual(reward, .5 / math.sqrt(visit))
        self.assertEqual(bonus.counts, {(0, 0): 1, (0, 1): 4})

    def test_wall_and_goal_receive_no_bonus_and_counts_are_explicit(self):
        bonus = VisitCountBonus()
        bonus.begin_episode((0, 0))
        self.assertEqual(bonus.transition((0, 0), (0, 0), terminated=False), 0)
        self.assertEqual(bonus.counts[(0, 0)], 1)
        self.assertEqual(bonus.transition((0, 0), (0, 1), terminated=True), 0)
        self.assertEqual(bonus.counts[(0, 1)], 1)
        self.assertEqual(bonus.total_reward, 0)

    def test_reset_start_counts_without_bonus_and_history_is_preserved(self):
        bonus = VisitCountBonus()
        bonus.begin_episode((0, 0))
        bonus.transition((0, 0), (0, 1), terminated=False)
        bonus.begin_episode((0, 1))
        bonus.begin_episode((0, 0))
        self.assertEqual(bonus.counts, {(0, 0): 2, (0, 1): 2})
        self.assertEqual(bonus.total_reward, .5)
        self.assertAlmostEqual(bonus.transition((0, 0), (0, 1), terminated=False), .5 / math.sqrt(3))
        snapshot = bonus.snapshot()
        snapshot["visits"].clear()
        self.assertEqual(len(bonus.counts), 2)

    def test_invalid_beta_is_rejected(self):
        for beta in (-1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                train_budget(method="curriculum_bonus", bonus_beta=beta, budget=1)

    def test_beta_zero_reproduces_curriculum_exactly(self):
        options = dict(seed=3, budget=500, max_steps=12, maze=("S...G",),
                       easy_radius=1, medium_radius=2, window=3, stage_cap=40)
        a, ra, ea, sa = train_budget(method="curriculum", **options)
        b, rb, eb, sb = train_budget(method="curriculum_bonus", bonus_beta=0, **options)
        def without_method(rows):
            return [{k: v for k, v in row.items() if k != "method"} for row in rows]
        self.assertEqual(a.q, b.q)
        self.assertEqual(a.rng.getstate(), b.rng.getstate())
        for left, right in ((ra, rb), (ea, eb), (sa, sb)):
            self.assertEqual(without_method(left), without_method(right))

    def test_bonus_changes_learning_but_not_environment_reward_rules(self):
        options = dict(seed=0, budget=300, max_steps=12, maze=("S...G",),
                       easy_radius=1, medium_radius=2, window=3, stage_cap=40)
        plain, _, _, _ = train_budget(method="curriculum", **options)
        bonus, rows, evaluations, _ = train_budget(method="curriculum_bonus", **options)
        self.assertNotEqual(plain.q, bonus.q)
        self.assertTrue(any(row["intrinsic_reward"] > 0 for row in rows))
        self.assertEqual(sum(row["steps"] for row in rows), 300)
        for row in rows:
            self.assertEqual(row["env_reward"], -row["steps"] - 4 * row["wall_hits"]
                             + (101 if row["success"] else 0))
            self.assertAlmostEqual(row["learning_reward"], row["env_reward"] + row["intrinsic_reward"])
            self.assertLessEqual(row["intrinsic_reward"], .5 * (row["steps"] - row["wall_hits"]))
        for row in evaluations:
            self.assertEqual(row["intrinsic_reward"], 0)
            self.assertEqual(row["learning_reward"], row["env_reward"])

    def test_evaluation_cannot_change_bonus_counts_rng_or_training(self):
        options = dict(seed=2, method="curriculum_bonus", budget=500, max_steps=12,
                       maze=("S...G",), easy_radius=1, medium_radius=2, window=3, stage_cap=40)
        snapshots_a, snapshots_b = [], []
        def capture(target):
            return lambda agent, env, method, metadata: target.append(copy.deepcopy(metadata))
        a, ra, _, sa = train_budget(eval_every=5, checkpoint=capture(snapshots_a), **options)
        b, rb, _, sb = train_budget(eval_every=47, checkpoint=capture(snapshots_b), **options)
        self.assertEqual((a.q, a.rng.getstate(), ra, sa, snapshots_a),
                         (b.q, b.rng.getstate(), rb, sb, snapshots_b))
        self.assertEqual({row["stage"] for row in ra}, {"near", "middle", "real_start"})
        final = snapshots_a[-1]["exploration_bonus"]
        # Every episode start and actual move counted once, across all stages.
        self.assertEqual(sum(v["count"] for v in final["visits"]),
                         len(ra) + sum(r["steps"] - r["wall_hits"] for r in ra))
        self.assertAlmostEqual(final["total_intrinsic_reward"], sum(r["intrinsic_reward"] for r in ra))

    def test_three_method_artifacts_and_frozen_loaded_bonus_model(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path, summary = run_comparison(output_root=temp, seeds=(0, 2), budget=101,
                                           max_steps=15, eval_every=20, plot=False, include_bonus=True)
            self.assertTrue(path.name.startswith("exploration-"))
            self.assertEqual(summary["config"]["bonus_beta"], .5)
            self.assertEqual(len(summary["comparison"]), 3)
            self.assertEqual(json.loads((path / "summary.json").read_text(encoding="utf-8")), summary)
            with (path / "logs/training.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(int(r["steps"]) for r in rows), 3 * 2 * 101)
            for method in summary["config"]["methods"]:
                model = latest_comparison_model(temp, method)
                self.assertEqual(model, (path / summary["demo_models"][method]).resolve())
                agent = QLearningAgent.load(model, MazeEnv(max_steps=15))
                before = copy.deepcopy((agent.q, agent.updates, agent.rng.getstate(), agent.metadata))
                evaluate_greedy(agent, MazeEnv(max_steps=15))
                self.assertEqual(before, (agent.q, agent.updates, agent.rng.getstate(), agent.metadata))
                if method == "curriculum_bonus":
                    self.assertGreater(agent.metadata["exploration_bonus"]["total_intrinsic_reward"], 0)
                else:
                    self.assertIsNone(agent.metadata["exploration_bonus"])
            # A newer incomplete run cannot hide the last complete comparison.
            incomplete = Path(temp) / "exploration-99999999"
            incomplete.mkdir()
            (incomplete / "summary.json").write_text('{"status":"running"}', encoding="utf-8")
            self.assertEqual(latest_comparison_model(temp, "curriculum_bonus").parent.parent, path)
            report = (path / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Curriculum + Bonus", report)
            self.assertIn("beta=0.5", report)


if __name__ == "__main__":
    unittest.main()
