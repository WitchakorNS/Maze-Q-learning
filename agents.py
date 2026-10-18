"""Policies independent of the maze rules and graphical interface."""

import random

from environment import Action


def episode_seed(seed, episode):
    """Independent, replayable random stream for each (seed, episode) pair."""
    return f"maze-random-v1:{seed}:{episode}"


class RandomAgent:
    """Uniformly choose from ALL four actions, without inspecting the map.

    Collisions remain possible. There is no learning, path planning, or
    action masking. A private RNG avoids changing Python's global RNG.
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, state):
        return Action(self.rng.randrange(4))
