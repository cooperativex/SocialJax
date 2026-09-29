"""Every agent must see itself, exactly once, on its own cell, and never as "another agent".

Agents are stored on the grid as k + len(Items), and the observation one-hot is
built from grids - 1, so agent k lives in one-hot channel k + len(Items) - 1.
combine_channels used to read the self channel at x[agent] (the raw grid value),
which is agent k+1's channel. Agents 0..N-2 then saw agent k+1 marked as "self"
and their own body as "other agent"; only the last agent saw itself, because JAX
clamps the out-of-range index back onto its own channel.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_self_channel.py
"""

import sys

import jax
import numpy as onp

import socialjax

# every env whose observation is built by combine_channels, at its default num_agents
ENV_IDS = [
    "clean_up",
    "coin_game",
    "gift",
    "harvest_common_open",
    "mushrooms",
    "pd_arena",
    "territory_open",
]
STEPS = 40


def run(name, fn):
    fn()
    print(f"ok: {name}")


def check_env(env_id):
    env = socialjax.make(env_id)
    n = env.num_agents
    # obs channels: len(Items) - 1 item classes, then [self, other agent, ...]
    self_ch = len(sys.modules[type(env).__module__].Items) - 1
    other_ch = self_ch + 1

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    for t in range(STEPS):
        o = onp.stack([onp.asarray(obs[i]) for i in range(n)])  # (N, H, W, C)
        selfmap = o[..., self_ch] != 0

        per_agent = selfmap.reshape(n, -1).sum(1)
        assert (per_agent == 1).all(), (
            f"{env_id} step {t}: self-channel cells per agent {per_agent.tolist()}, expected all 1"
        )
        other_on_self = (o[..., other_ch] != 0) & selfmap
        assert not other_on_self.any(), (
            f"{env_id} step {t}: agents {onp.flatnonzero(other_on_self.any(axis=(1, 2))).tolist()} see themselves as another agent"
        )

        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        obs, state, _, _, _ = step(k_step, state, [acts[i] for i in range(n)])


def test_clean_up():
    check_env("clean_up")


def test_coin_game():
    check_env("coin_game")


def test_gift():
    check_env("gift")


def test_harvest_common_open():
    check_env("harvest_common_open")


def test_mushrooms():
    check_env("mushrooms")


def test_pd_arena():
    check_env("pd_arena")


def test_territory_open():
    check_env("territory_open")


if __name__ == "__main__":
    for env_id in ENV_IDS:
        run(f"self channel: {env_id}", lambda: check_env(env_id))
    print("ALL SELF-CHANNEL TESTS PASSED")
