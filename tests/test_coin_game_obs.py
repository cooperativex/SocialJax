"""Coin colours in the observation must be egocentric.

Agent 0 owns the red coins and agent 1 the green ones. A shared-parameter policy cannot tell
the two agents apart from their (symmetric) observations, so the red_apple channel must always
show the observer's own coins and the green_apple channel the other agent's coins.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_coin_game_obs.py
"""

import sys

import jax
import numpy as onp

import socialjax

STEPS = 100


def run(name, fn):
    fn()
    print(f"ok: {name}")


def test_own_coin_channel():
    env = socialjax.make("coin_game")
    Items = sys.modules[type(env).__module__].Items
    n, S, P = env.num_agents, env.OBS_SIZE, env.PADDING
    red, green = Items.red_apple - 1, Items.green_apple - 1
    own = [Items.red_apple, Items.green_apple]

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    seen = onp.zeros(n, dtype=int)
    for t in range(STEPS):
        grid = onp.pad(onp.asarray(state.grid), P, constant_values=int(Items.wall))
        for k in range(n):
            # the agent's raw view: same window and rotation as _get_obs
            x, y = (int(v) for v in env.get_obs_point(state.agent_locs[k]))
            view = onp.rot90(grid[x:x + S, y:y + S], k=int(state.agent_locs[k, 2]), axes=(0, 1))
            o = onp.asarray(obs[k])
            assert (o[..., red] == (view == own[k])).all(), f"step {t}: agent {k} own-coin channel"
            assert (o[..., green] == (view == own[1 - k])).all(), f"step {t}: agent {k} other-coin channel"
            seen[k] += int((view == own[k]).sum())

        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        obs, state, _, _, _ = step(k_step, state, [acts[i] for i in range(n)])
    assert (seen > 0).all(), f"own coins never in view {seen.tolist()}, test is vacuous"


if __name__ == "__main__":
    run("coin_game own-coin channel", test_own_coin_channel)
    print("ALL COIN GAME OBS TESTS PASSED")
