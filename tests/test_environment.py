"""Behavioral checks for the rules that future RL agents will depend on."""

from collections import deque
import unittest

from environment import Action, DEFAULT_MAZE, DELTAS, MazeEnv


def solution(env):
    queue = deque([(env.start, [])])
    seen = {env.start}
    while queue:
        position, path = queue.popleft()
        if position == env.goal:
            return path
        for action, (dr, dc) in DELTAS.items():
            candidate = position[0] + dr, position[1] + dc
            if candidate not in seen and env.is_walkable(candidate):
                seen.add(candidate)
                queue.append((candidate, path + [action]))
    raise AssertionError("Goal unreachable")


class MazeRulesTest(unittest.TestCase):
    def test_default_map_is_ten_by_ten_and_solvable(self):
        env = MazeEnv()
        self.assertEqual((env.rows, env.cols), (10, 10))
        route = solution(env)
        self.assertGreater(len(route), 0)
        self.assertLess(len(route), env.max_steps)

    def test_walk_and_collision_count_attempts_and_distinct_rewards(self):
        env = MazeEnv()
        state, reward, terminal, truncated, info = env.step(Action.UP)
        self.assertEqual((state, reward, terminal, truncated), ((1, 1), -5, False, False))
        self.assertEqual((info["steps"], info["wall_hits"], info["unique_cells"]), (1, 1, 1))
        state, reward, _, _, info = env.step(Action.RIGHT)
        self.assertEqual((state, reward, info["total_reward"]), ((1, 2), -1, -6))
        self.assertEqual(info["unique_cells"], 2)

    def test_goal_reward_and_post_terminal_guard(self):
        env = MazeEnv()
        route = solution(env)
        for action in route:
            state, reward, terminal, truncated, info = env.step(action)
        self.assertEqual((state, reward, terminal, truncated), (env.goal, 100, True, False))
        self.assertTrue(info["is_success"])
        self.assertEqual(info["total_reward"], 100 - (len(route) - 1))
        with self.assertRaises(RuntimeError):
            env.step(Action.LEFT)

    def test_timeout_including_wall_attempt_and_reset(self):
        env = MazeEnv(max_steps=2)
        env.step(Action.UP)
        _, reward, terminal, truncated, info = env.step(Action.UP)
        self.assertEqual((reward, terminal, truncated, info["steps"]), (-5, False, True, 2))
        with self.assertRaises(RuntimeError):
            env.step(Action.RIGHT)
        state, info = env.reset()
        self.assertEqual(state, env.start)
        self.assertFalse(env.done)
        self.assertEqual((env.steps, env.wall_hits, env.total_reward), (0, 0, 0))
        self.assertEqual(env.visited, {env.start})

    def test_goal_on_last_allowed_step_is_success(self):
        env = MazeEnv(("SG",), max_steps=1)
        _, reward, terminal, truncated, info = env.step(Action.RIGHT)
        self.assertEqual((reward, terminal, truncated, info["is_success"]), (100, True, False, True))

    def test_outside_map_is_collision_not_wraparound(self):
        env = MazeEnv(("SG",))
        for action in (Action.UP, Action.LEFT, Action.DOWN):
            state, reward, _, _, _ = env.step(action)
            self.assertEqual((state, reward), ((0, 0), -5))

    def test_invalid_action_does_not_change_episode(self):
        env = MazeEnv()
        for action in (-1, 4, "up", 1.0, True, None):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual((env.position, env.steps), (env.start, 0))

    def test_invalid_map_or_limit_is_rejected(self):
        for maze in ((), ("S.G", "##"), ("S#G",), ("SSG",), ("S?G",), ("...",)):
            with self.subTest(maze=maze), self.assertRaises(ValueError):
                MazeEnv(maze)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                MazeEnv(DEFAULT_MAZE, max_steps=limit)

    def test_text_render_moves_agent(self):
        env = MazeEnv()
        self.assertEqual(env.render().splitlines()[1][1], "A")
        env.step(Action.RIGHT)
        self.assertEqual(env.render().splitlines()[1][1:3], "SA")


if __name__ == "__main__":
    unittest.main()
