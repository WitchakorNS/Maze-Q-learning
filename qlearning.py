"""Tabular Q-learning and portable, validated JSON checkpoints."""

import json
import math
from pathlib import Path
import random

from environment import Action, MazeEnv


def unit_float(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not (0 <= value <= 1) or (positive and value == 0):
        raise ValueError(f"{name} must be {'> 0 and ' if positive else ''}between 0 and 1")
    return float(value)


def environment_spec(env):
    return {
        "maze": list(env.maze), "start": list(env.start), "goal": list(env.goal),
        "max_steps": env.max_steps,
        "rewards": {"walk": -1, "wall": -5, "goal": 100},
        "actions": [action.name for action in Action],
    }


class QLearningAgent:
    """Learns from (s, a, r, s') only; never receives the wall map or a route.

    epsilon=0 selects the first maximizer deterministically and consumes no RNG.
    With epsilon>0, exploitation breaks ties randomly; exploration samples all
    four actions, including those which collide with walls.
    """

    def __init__(self, rows, cols, *, seed=0, alpha=0.1, gamma=0.95):
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (rows, cols)):
            raise ValueError("rows and cols must be positive integers")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.rows, self.cols, self.seed = rows, cols, seed
        self.alpha = unit_float(alpha, "alpha", positive=True)
        self.gamma = unit_float(gamma, "gamma")
        self.rng = random.Random(seed)
        self.q = [[[0.0 for _ in Action] for _ in range(cols)] for _ in range(rows)]
        self.updates = 0
        self.metadata = {}

    def values(self, state):
        if (not isinstance(state, (list, tuple)) or len(state) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in state)):
            raise ValueError("state must be (row, column) integers")
        row, col = state
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError("state outside Q-table")
        return self.q[row][col]

    def act(self, state, epsilon=0.0):
        epsilon = unit_float(epsilon, "epsilon")
        values = self.values(state)
        if epsilon == 0:
            return Action(max(range(4), key=values.__getitem__))
        if self.rng.random() < epsilon:
            return Action(self.rng.randrange(4))
        best = max(values)
        return Action(self.rng.choice([a for a in range(4) if values[a] == best]))

    def update(self, state, action, reward, next_state, *, terminated, truncated=False):
        values, next_values = self.values(state), self.values(next_state)
        if isinstance(action, bool) or not isinstance(action, int) or action not in range(4):
            raise ValueError("action must be an integer from 0 to 3")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward)):
            raise ValueError("reward must be finite")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise ValueError("termination flags must be booleans")
        # Time limits are external to this position-only MDP: bootstrap on
        # truncation. Actual goal termination has no future-value term.
        target = reward if terminated else reward + self.gamma * max(next_values)
        values[action] += self.alpha * (target - values[action])
        self.updates += 1
        return values[action]

    def save(self, path, env, *, metadata=None):
        if (env.rows, env.cols) != (self.rows, self.cols):
            raise ValueError("Environment dimensions do not match Q-table")
        payload = {
            "schema_version": 1, "algorithm": "q_learning",
            "environment": environment_spec(env),
            "hyperparameters": {"alpha": self.alpha, "gamma": self.gamma},
            "seed": self.seed, "updates": self.updates,
            "metadata": metadata if metadata is not None else self.metadata,
            "q_table": self.q,
        }
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path, env):
        try:
            with Path(path).open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or payload.get("algorithm") != "q_learning"):
                raise ValueError("Unsupported model format")
            if payload.get("environment") != environment_spec(env):
                raise ValueError("Model map, rules, or action order differ from this environment")
            params = payload["hyperparameters"]
            agent = cls(env.rows, env.cols, seed=payload["seed"], **params)
            table = payload["q_table"]
            if (not isinstance(table, list) or len(table) != env.rows
                    or any(not isinstance(row, list) or len(row) != env.cols for row in table)):
                raise ValueError("Invalid Q-table dimensions")
            for row in table:
                for values in row:
                    if not isinstance(values, list) or len(values) != 4:
                        raise ValueError("Each state must contain four Q-values")
                    if any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values):
                        raise ValueError("Q-values must be finite numbers")
            updates = payload["updates"]
            if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
                raise ValueError("Invalid update counter")
            if not isinstance(payload.get("metadata"), dict):
                raise ValueError("Model metadata must be an object")
            agent.q = table
            agent.updates = updates
            agent.metadata = payload["metadata"]
            return agent
        except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Q-learning checkpoint: {exc}") from exc


def epsilon_at(completed_episodes, *, start=1.0, minimum=0.05, decay=0.995):
    start = unit_float(start, "epsilon_start")
    minimum = unit_float(minimum, "epsilon_min")
    decay = unit_float(decay, "epsilon_decay", positive=True)
    if minimum > start:
        raise ValueError("epsilon_min must not exceed epsilon_start")
    if isinstance(completed_episodes, bool) or not isinstance(completed_episodes, int) or completed_episodes < 0:
        raise ValueError("completed_episodes must be a nonnegative integer")
    return max(minimum, start * decay ** completed_episodes)


def latest_demo_model(results_dir):
    """Find the first-seed checkpoint from the latest completed experiment."""
    for manifest in sorted(Path(results_dir).glob("qlearning-*/summary.json"), reverse=True):
        with manifest.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict):
            raise ValueError("Training summary must be an object")
        if summary.get("status") != "complete":
            continue
        relative = summary.get("demo_model")
        if not isinstance(relative, str):
            raise ValueError("Completed run has no demo_model path")
        run_dir = manifest.parent.resolve()
        model = (run_dir / relative).resolve()
        if not model.is_relative_to(run_dir):
            raise ValueError("Demo model must stay inside its experiment directory")
        if not model.is_file():
            raise FileNotFoundError(f"Demo checkpoint missing: {model.name}")
        return model
    raise FileNotFoundError("No completed Q-learning run. Run train.cmd first.")


def evaluate_greedy(agent, env):
    """No update, no exploration, no RNG consumption. Returns stats and trajectory."""
    state, _ = env.reset()
    trajectory = [{"step": 0, "row": state[0], "col": state[1], "action": "", "reward": 0}]
    while not env.done:
        action = agent.act(state, epsilon=0.0)
        state, reward, _, _, info = env.step(action)
        trajectory.append({"step": env.steps, "row": state[0], "col": state[1],
                           "action": action.name, "reward": reward})
    stats = {
        "steps": env.steps, "env_reward": env.total_reward,
        "success": info["is_success"], "terminated": env.terminated,
        "truncated": env.truncated, "wall_hits": env.wall_hits,
        "unique_cells": len(env.visited), "final_row": state[0], "final_col": state[1],
    }
    return stats, trajectory


def train_one(seed=0, *, episodes=5000, max_steps=300, alpha=0.1, gamma=0.95,
              epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
              eval_interval=100, maze=None, checkpoint=None):
    """Train one independent agent; evaluation cannot influence its updates/RNG.

    checkpoint(agent, env, episode, metadata) persists predeclared snapshots.
    All runs finish their prescribed budget, regardless of evaluation scores.
    """
    for name, value in (("episodes", episodes), ("eval_interval", eval_interval)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    schedule = {"start": epsilon_start, "minimum": epsilon_min, "decay": epsilon_decay}
    epsilon_at(0, **schedule)
    env = MazeEnv(max_steps=max_steps, **({"maze": maze} if maze is not None else {}))
    evaluation_env = MazeEnv(maze=env.maze, max_steps=max_steps)
    agent = QLearningAgent(env.rows, env.cols, seed=seed, alpha=alpha, gamma=gamma)
    snapshots = {n for n in (0, 10, 100, 500, 1000, episodes) if n <= episodes}
    training, evaluations = [], []

    def observe(episode):
        metadata = {"episodes_completed": episode, "training_env_steps": agent.updates,
                    "epsilon_next": epsilon_at(episode, **schedule), "epsilon_schedule": schedule,
                    "training_budget_episodes": episodes, "snapshot_selection": "predeclared"}
        stats, _ = evaluate_greedy(agent, evaluation_env)
        evaluations.append({"seed": seed, "checkpoint_episode": episode,
                            "training_env_steps": agent.updates, "epsilon": 0.0, **stats})
        if episode in snapshots and checkpoint is not None:
            checkpoint(agent, env, episode, metadata)

    observe(0)
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        epsilon = epsilon_at(episode - 1, **schedule)
        while not env.done:
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated=terminated, truncated=truncated)
            state = next_state
        training.append({"seed": seed, "episode": episode, "epsilon": epsilon,
                         "next_epsilon": epsilon_at(episode, **schedule),
                         "steps": env.steps, "env_reward": env.total_reward,
                         "success": env.terminated, "terminated": env.terminated,
                         "truncated": env.truncated, "wall_hits": env.wall_hits,
                         "unique_cells": len(env.visited), "training_env_steps": agent.updates})
        if episode % eval_interval == 0 or episode in snapshots:
            observe(episode)
    return agent, training, evaluations
