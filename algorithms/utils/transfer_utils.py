"""Shared utilities for the TRANSFER algorithm (reward-exchange / self-interest).

Each agent retains a proportion ``s`` of its own reward and distributes
``(1 - s)`` equally among the (n-1) co-players. The reward-exchange ratio is the
ratio of an agent's utility gain when a co-player receives one unit of reward
to its gain when it receives the same unit itself:

    ratio = (1 - s) / (s * (n - 1))

Solving for s:

    s = 1 / (1 + ratio * (n - 1))

Edge cases:
    ratio = 0  -> s = 1                 (pure individual reward)
    ratio = 1  -> s = 1 / num_agents    (common reward / full sharing)
"""


def s_from_ratio(ratio: float, num_agents: int) -> float:
    """Compute self-interest ``s`` from a reward-exchange ``ratio``."""
    if num_agents < 2:
        raise ValueError(f"num_agents must be >= 2, got {num_agents}")
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, got {ratio}")
    return 1.0 / (1.0 + ratio * (num_agents - 1))
