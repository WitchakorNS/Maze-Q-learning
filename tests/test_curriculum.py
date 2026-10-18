"""Curriculum starts, promotions, equal budgets, and evaluation isolation."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from curriculum import (DistanceCurriculum, latest_comparison_model,
                        learning_milestones, run_comparison, train_budget)
from environment import Action, MazeEnv
from qlearning import QLearningAgent, environment_spec, evaluate_greedy


class CurriculumTest(unittest.TestCase):
    def test_distance_is_walkable_path_distance_and_copy_is_private(self):
        env = MazeEnv()
        distances = env.distances_to_goal()
        self.assertEqual(distances[env.goal], 0)
        self.assertEqual(distances[env.start], 18)
        self.assertNotEqual(distances[env.start], sum(abs(a - b) for a, b in zip(env.start, env.goal)))
        distances.clear()
        self.assertEqual(env.distances_to_goal()[env.start], 18)

    def test_custom_reset_preserves_nominal_start_and_checkpoint_contract(self):
        env = MazeEnv(("S.G",))
        spec = environment_spec(env)
        state, info = env.reset(start=(0, 1))
        self.assertEqual((state, info["episode_start"]), ((0, 1), (0, 1)))
        self.assertEqual(env.start, (0, 0))
        self.assertEqual(environment_spec(env), spec)
        env.step(Action.RIGHT)
        self.assertTrue(env.terminated)
        self.assertEqual(env.total_reward, 100)
        state, _ = env.reset()
        self.assertEqual((state, env.episode_start), (env.start, env.start))
        self.assertEqual((env.steps, env.total_reward, env.wall_hits), (0, 0, 0))

    def test_invalid_reset_does_not_mutate_episode(self):
        env = MazeEnv(("S.G#", "####", ".###"))
        env.step(Action.RIGHT)
        before = (env.position, env.steps, env.total_reward, env.episode_start)
        for start in ((0, 2), (0, 3), (2, 0), (-1, 0), (True, 0), "bad", (0,)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                env.reset(start=start)
            self.assertEqual((env.position, env.steps, env.total_reward, env.episode_start), before)

    def test_text_render_marks_actual_episode_start(self):
        env = MazeEnv(("S.G",))
        env.reset(start=(0, 1))
        self.assertEqual(env.render(), ".AG")
        env.step(Action.LEFT)
        self.assertEqual(env.render(), "ASG")

    def test_start_bands_and_seed_reproducibility(self):
        env = MazeEnv()
        a, b = DistanceCurriculum(env, seed=11), DistanceCurriculum(env, seed=11)
        samples = [a.select_start() for _ in range(100)]
        self.assertEqual(samples, [b.select_start() for _ in range(100)])
        distances = env.distances_to_goal()
        self.assertTrue(all(1 <= distances[p] <= 4 for p in samples))
        self.assertTrue(all(5 <= distances[p] <= 10 for p in a.levels[1]["starts"]))
        self.assertEqual(a.levels[2]["starts"], (env.start,))
        self.assertNotIn(env.goal, samples)

    def test_training_success_window_promotes_and_clears_history(self):
        teacher = DistanceCurriculum(MazeEnv(), window=3, threshold=.6, stage_cap=100)
        self.assertIsNone(teacher.observe_episode(True, 1))
        self.assertIsNone(teacher.observe_episode(False, 1))
        event = teacher.observe_episode(True, 1)
        self.assertEqual((event["from_stage"], event["to_stage"], event["reason"]),
                         ("near", "middle", "success_threshold"))
        self.assertEqual(len(teacher.recent), 0)
        self.assertEqual(teacher.stage_actions, 0)
        self.assertIsNone(teacher.observe_episode(True, 1))
        self.assertEqual(teacher.name, "middle")

    def test_stage_cap_is_logged_and_final_stage_never_promotes(self):
        teacher = DistanceCurriculum(MazeEnv(), window=50, stage_cap=3)
        self.assertIsNone(teacher.observe_episode(False, 2))
        first = teacher.observe_episode(False, 2)
        self.assertEqual((first["reason"], first["stage_training_actions"]), ("stage_action_cap", 4))
        second = teacher.observe_episode(False, 3)
        self.assertEqual(second["to_stage"], "real_start")
        self.assertIsNone(teacher.observe_episode(True, 30))
        self.assertEqual(teacher.select_start(), MazeEnv().start)

    def test_partial_episode_is_not_a_promotion_success(self):
        teacher = DistanceCurriculum(MazeEnv(), window=1, stage_cap=1)
        self.assertIsNone(teacher.observe_episode(False, 5, complete=False))
        self.assertEqual((teacher.index, len(teacher.recent)), (0, 0))

    def test_exact_budget_and_partial_episode_are_visible(self):
        agent, rows, evaluations, transitions = train_budget(0, method="plain", budget=17, eval_every=7)
        self.assertEqual(agent.updates, 17)
        self.assertEqual(sum(row["steps"] for row in rows), 17)
        self.assertTrue(rows[-1]["budget_cutoff"])
        self.assertFalse(rows[-1]["episode_complete"])
        self.assertFalse(rows[-1]["success"])
        self.assertFalse(rows[-1]["truncated"])
        self.assertEqual(evaluations[-1]["training_env_steps"], 17)
        self.assertEqual(transitions, [])

    def test_both_methods_spend_same_budget_with_no_bonus(self):
        for method in ("plain", "curriculum"):
            agent, rows, evaluations, _ = train_budget(5, method=method, budget=217, eval_every=31)
            self.assertEqual(agent.updates, 217)
            self.assertEqual(sum(row["steps"] for row in rows), 217)
            for row in rows:
                self.assertEqual(row["env_reward"], -row["steps"] - 4 * row["wall_hits"]
                                 + (101 if row["success"] else 0))
            self.assertTrue(all(row["epsilon"] == 0 for row in evaluations))

    def test_eval_frequency_changes_no_training_data_or_stage_transitions(self):
        options = {"method": "curriculum", "budget": 250, "max_steps": 12,
                   "maze": ("S...G",), "easy_radius": 1, "medium_radius": 2,
                   "window": 3, "stage_cap": 40, "epsilon_decay": .99}
        a, rows_a, _, stages_a = train_budget(2, eval_every=5, **options)
        b, rows_b, _, stages_b = train_budget(2, eval_every=47, **options)
        self.assertEqual(a.q, b.q)
        self.assertEqual(a.rng.getstate(), b.rng.getstate())
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(stages_a, stages_b)
        self.assertEqual({r["stage"] for r in rows_a}, {"near", "middle", "real_start"})

    def test_milestones_are_observed_and_missing_values_stay_none(self):
        rows = [{"training_env_steps": i * 1000, "success": success}
                for i, success in enumerate((False, True, False, True, True, True, False))]
        self.assertEqual(learning_milestones(rows), (1000, 3000))
        self.assertEqual(learning_milestones(rows[:1]), (None, None))
        self.assertEqual(learning_milestones(rows[:2]), (1000, None))

    def test_comparison_artifacts_and_loaded_models_match(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path, summary = run_comparison(output_root=temp, seeds=(1, 3), budget=61,
                                           max_steps=15, eval_every=20, plot=False)
            self.assertEqual(summary["config"]["budget_unit"], "training_action")
            with (path / "logs/training.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(int(r["steps"]) for r in rows), 2 * 2 * 61)
            for row in summary["per_seed"]:
                self.assertEqual(row["training_env_steps"], 61)
                loaded = QLearningAgent.load(path / row["model"], MazeEnv(max_steps=15))
                stats, _ = evaluate_greedy(loaded, MazeEnv(max_steps=15))
                self.assertEqual((stats["steps"], stats["success"]), (row["steps"], row["success"]))
                self.assertEqual(loaded.metadata["epsilon_schedule_unit"], "training_action")
            for method in ("plain", "curriculum"):
                self.assertEqual(latest_comparison_model(temp, method),
                                 (path / summary["demo_models"][method]).resolve())
            self.assertEqual(json.loads((path / "summary.json").read_text(encoding="utf-8")), summary)
            self.assertTrue((path / "summary.md").is_file())
            self.assertTrue((path / "comparison.csv").is_file())
            self.assertIsNone(summary["learning_curve"])

    def test_model_lookup_rejects_external_path_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                latest_comparison_model(temp, "plain")
            run = Path(temp) / "curriculum-20200101"
            run.mkdir()
            (run / "summary.json").write_text(json.dumps({
                "status": "complete", "demo_models": {"plain": "../outside.json"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                latest_comparison_model(temp, "plain")

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"window": 0}, {"easy_radius": 10}, {"threshold": 0}, {"stage_cap": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DistanceCurriculum(MazeEnv(), **kwargs)
        for kwargs in ({"budget": 0}, {"method": "unknown"}, {"eval_every": 0},
                       {"epsilon_min": 1, "epsilon_start": .1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_budget(**kwargs)


if __name__ == "__main__":
    unittest.main()
