"""Deterministic grid maze, independent of the interface and learning algorithm."""

from collections import deque
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DEFAULT_MAZE = (
    "##########",
    "#S..#....#",
    "###.#.##.#",
    "#...#..#.#",
    "#.####.#.#",
    "#......#.#",
    "#.####.#.#",
    "#.#....#.#",
    "#...##..G#",
    "##########",
)

DELTAS = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}


class MazeEnv:
    """A Gym-style API using only Python's standard library.

    Observation: (row, column). Actions: 0 up, 1 right, 2 down, 3 left.
    Each action gets exactly one reward: goal +100, collision -5, walk -1.
    Every attempted move counts as a step, including collisions.
    Reaching the goal terminates; exhausting the step budget truncates.
    """

    def __init__(self, maze=DEFAULT_MAZE, max_steps=300):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.maze = tuple(maze)
        if not self.maze or not isinstance(self.maze[0], str) or not self.maze[0]:
            raise ValueError("Maze must contain nonempty text rows")
        self.rows, self.cols = len(self.maze), len(self.maze[0])
        if any(not isinstance(row, str) or len(row) != self.cols for row in self.maze):
            raise ValueError("Maze rows must have equal length")
        if any(cell not in "#.SG" for row in self.maze for cell in row):
            raise ValueError("Maze cells must be #, ., S, or G")
        starts = self._find("S")
        goals = self._find("G")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Maze must contain exactly one S and one G")
        self.start, self.goal = starts[0], goals[0]
        if self.goal not in self._reachable(self.start):
            raise ValueError("There must be a path from S to G")
        self._goal_distances = self._distances_from(self.goal)
        self.max_steps = max_steps
        self.reset()

    def _find(self, marker):
        return [(r, c) for r, row in enumerate(self.maze)
                for c, cell in enumerate(row) if cell == marker]

    def is_walkable(self, position):
        row, col = position
        return (0 <= row < self.rows and 0 <= col < self.cols
                and self.maze[row][col] != "#")

    def _reachable(self, start):
        return set(self._distances_from(start))

    def _distances_from(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for dr, dc in DELTAS.values():
                candidate = row + dr, col + dc
                if candidate not in distances and self.is_walkable(candidate):
                    distances[candidate] = distances[(row, col)] + 1
                    queue.append(candidate)
        return distances

    def distances_to_goal(self):
        """Teacher/analysis information, never an observation given to an agent."""
        return dict(self._goal_distances)

    @property
    def done(self):
        return self.terminated or self.truncated

    def reset(self, *, start=None):
        position = self.start if start is None else start
        if (not isinstance(position, (tuple, list)) or len(position) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in position)):
            raise ValueError("Episode start must be (row, column) integers")
        position = tuple(position)
        if position == self.goal or position not in self._goal_distances:
            raise ValueError("Episode start must be a non-goal cell reachable from the goal")
        self.episode_start = position
        self.position = position
        self.steps = 0
        self.total_reward = 0
        self.wall_hits = 0
        self.terminated = False
        self.truncated = False
        self.visited = {position}
        self.last_reward = 0
        return self.position, self._info("reset")

    def _info(self, event):
        return {
            "event": event,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "wall_hits": self.wall_hits,
            "unique_cells": len(self.visited),
            "is_success": self.terminated,
            "episode_start": self.episode_start,
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; call reset() before stepping again")
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("Action must be an integer from 0 to 3")
        try:
            action = Action(action)
        except ValueError as exc:
            raise ValueError("Action must be an integer from 0 to 3") from exc
        dr, dc = DELTAS[action]
        candidate = self.position[0] + dr, self.position[1] + dc
        self.steps += 1
        if not self.is_walkable(candidate):
            self.wall_hits += 1
            reward, event = -5, "wall"
        else:
            self.position = candidate
            self.visited.add(candidate)
            self.terminated = candidate == self.goal
            reward, event = (100, "goal") if self.terminated else (-1, "move")
        self.truncated = self.steps >= self.max_steps and not self.terminated
        self.total_reward += reward
        self.last_reward = reward
        return self.position, reward, self.terminated, self.truncated, self._info(event)

    def render(self):
        """Return a text view; the GUI has its own renderer."""
        lines = [list(row) for row in self.maze]
        if self.episode_start != self.start:
            lines[self.start[0]][self.start[1]] = "."
            lines[self.episode_start[0]][self.episode_start[1]] = "S"
        row, col = self.position
        lines[row][col] = "A"
        return "\n".join("".join(line) for line in lines)
