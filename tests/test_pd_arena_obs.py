"""The per-agent resource vectors in the observation must be egocentric.

Rewards depend on each agent's own coop/defect inventory, but a shared-parameter policy cannot
tell which index it is from its (symmetric) observation. The coop/defect vectors are therefore
rotated so that entry 0 is always the observing agent.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_pd_arena_obs.py
"""

import sys

import jax
import numpy as onp

import socialjax

STEPS = 40


def run(name, fn):
    fn()
    print(f"ok: {name}")


def test_own_inventory_first():
    env = socialjax.make("pd_arena", num_agents=4)
    Items = sys.modules[type(env).__module__].Items
    n = env.num_agents
    # obs channels: len(Items) - 1 items, self, other, angle x4, coop x n, defect x n, ...
    coop = len(Items) - 1 + 6

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    step = jax.jit(env.step)
    for t in range(STEPS):
        key, k_act, k_step, k_c, k_d = jax.random.split(key, 5)
        # distinct inventories per agent, so a wrong offset is visible
        state = state.replace(
            coop_resources=jax.random.randint(k_c, (n,), 0, 9).astype(state.coop_resources.dtype),
            defect_resources=jax.random.randint(k_d, (n,), 0, 9).astype(state.defect_resources.dtype),
        )
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        obs, state, _, _, _ = step(k_step, state, [acts[i] for i in range(n)])

        cr, dr = onp.asarray(state.coop_resources), onp.asarray(state.defect_resources)
        for k in range(n):
            o = onp.asarray(obs[k])
            assert (o[..., coop:coop + n] == onp.roll(cr, -k)).all(), f"step {t}: agent {k} coop resources"
            assert (o[..., coop + n:coop + 2 * n] == onp.roll(dr, -k)).all(), f"step {t}: agent {k} defect resources"


if __name__ == "__main__":
    run("pd_arena own inventory first", test_own_inventory_first)
    print("ALL PD ARENA OBS TESTS PASSED")
