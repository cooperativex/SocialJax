"""Rollout property check: no two agents may ever share a (row, col) cell.

Runs random-action rollouts in every registered environment and asserts,
at every step, that agent positions are pairwise distinct (and that stored
reborn locations, where the env has them, are pairwise distinct too).

Run:
  ulimit -c 0
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/rollout_overlap_check.py [steps] [seeds]
"""

import sys

import jax
import jax.numpy as jnp

import socialjax
from socialjax.registration import REGISTERED_ENVS

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def agent_positions(state):
    if hasattr(state, "agent_locs"):
        return state.agent_locs[:, :2]
    return state.agent_positions[:, :2]


def check_unique(rc, env_id, seed, step, what):
    cells = set(map(tuple, jax.device_get(rc).tolist()))
    assert len(cells) == rc.shape[0], (
        f"{env_id} seed={seed} step={step}: duplicate {what}: "
        f"{jax.device_get(rc).tolist()}"
    )


def run_env(env_id):
    env = socialjax.make(env_id)
    step_fn = jax.jit(env.step)

    def sample_actions(key):
        keys = jax.random.split(key, env.num_agents)
        return [env.action_space(a).sample(keys[i])
                for i, a in enumerate(env.agents)]

    for seed in range(SEEDS):
        key = jax.random.PRNGKey(seed)
        key, k_reset = jax.random.split(key)
        _, state = env.reset(k_reset)
        check_unique(agent_positions(state), env_id, seed, -1, "reset positions")
        for step in range(STEPS):
            key, k_act, k_step = jax.random.split(key, 3)
            actions = sample_actions(k_act)
            _, state, _, _, _ = step_fn(k_step, state, actions)
            check_unique(agent_positions(state), env_id, seed, step, "positions")
            if hasattr(state, "reborn_locs"):
                check_unique(state.reborn_locs[:, :2], env_id, seed, step,
                             "reborn positions")
    print(f"ok: {env_id} ({env.num_agents} agents, {SEEDS}x{STEPS} steps)")


if __name__ == "__main__":
    failures = []
    for env_id in REGISTERED_ENVS:
        try:
            run_env(env_id)
        except AssertionError as e:
            failures.append(str(e))
            print(f"FAIL: {env_id}: {e}")
    if failures:
        sys.exit(f"{len(failures)} environment(s) violated the no-overlap invariant")
    print("ALL ENVIRONMENTS OVERLAP-FREE")
