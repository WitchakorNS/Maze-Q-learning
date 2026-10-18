"""Training-only count-based exploration; no map, goal distances, or RNG."""

import math


def nonnegative_beta(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("bonus_beta must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("bonus_beta must be a finite nonnegative number")
    return float(value)


class VisitCountBonus:
    """One counter per training run, shared across episodes and curriculum stages.

    Count each reset start and each actual arrival (including the goal).
    Collisions are not arrivals. Only nonterminal moves earn beta/sqrt(N+1),
    where N is the destination's count BEFORE this arrival. Evaluation must
    never call this object. Counts in snapshots are audit data, not a promise
    of exact training resumption (the learner's RNG is not checkpointed).
    """

    RULE = "beta/sqrt(prior_destination_visits+1); moved_nonterminal_only"

    def __init__(self, beta=0.5):
        self.beta = nonnegative_beta(beta)
        self.counts = {}
        self.total_reward = 0.0

    def begin_episode(self, start):
        self.counts[start] = self.counts.get(start, 0) + 1

    def transition(self, state, next_state, *, terminated):
        if next_state == state:
            return 0.0
        prior = self.counts.get(next_state, 0)
        self.counts[next_state] = prior + 1
        reward = 0.0 if terminated else self.beta / math.sqrt(prior + 1)
        self.total_reward += reward
        return reward

    def snapshot(self):
        return {"beta": self.beta, "rule": self.RULE,
                "reset_start_counts_as_visit": True, "goal_counts_as_visit": True,
                "collision_counts_as_visit": False,
                "counts_reset_on_episode_or_stage": False,
                "evaluation_bonus": 0, "evaluation_mutates_counts": False,
                "total_intrinsic_reward": self.total_reward,
                "visits": [{"row": p[0], "col": p[1], "count": count}
                           for p, count in sorted(self.counts.items())]}
