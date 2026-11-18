"""Agents must be able to start facing any of their four headings.

Agents turn through four headings (0-3, mod 4), but the starting heading was drawn with
randint(..., maxval=3) in seven environments, and resolve_respawn defaulted to
dir_maxval=3, so no agent ever started or respawned facing heading 3. Respawns are
checked in tests/test_movement.py::test_respawn.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_start_headings.py
"""

import jax
import numpy as onp

import socialjax

# every env whose reset draws the starting headings with randint
ENV_IDS = [
    "clean_up",
    "coin_game",
    "gift",
    "harvest_common_open",
    "mushrooms",
    "pd_arena",
    "territory_open",
]


def run(name, fn):
    fn()
    print(f"ok: {name}")


def check_start_headings(env_id, resets=256):
    env = socialjax.make(env_id)
    keys = jax.random.split(jax.random.PRNGKey(0), resets)
    _, states = jax.vmap(env.reset)(keys)
    headings = set(onp.asarray(states.agent_locs[..., 2]).ravel().tolist())
    assert headings == {0, 1, 2, 3}, f"{env_id}: start headings seen {sorted(headings)}"


def test_clean_up():
    check_start_headings("clean_up")


def test_coin_game():
    check_start_headings("coin_game")


def test_gift():
    check_start_headings("gift")


def test_harvest_common_open():
    check_start_headings("harvest_common_open")


def test_mushrooms():
    check_start_headings("mushrooms")


def test_pd_arena():
    check_start_headings("pd_arena")


def test_territory_open():
    check_start_headings("territory_open")


if __name__ == "__main__":
    for env_id in ENV_IDS:
        run(f"start headings 0-3: {env_id}", lambda: check_start_headings(env_id))
    print("ALL START-HEADING TESTS PASSED")
