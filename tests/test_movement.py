"""Standalone checks for socialjax.environments.movement (no pytest).

Run on a compute node:
  ulimit -c 0
  export PYTHONPATH=$PWD:$PYTHONPATH
  srun -n 1 env JAX_PLATFORMS=cpu ~/ENTER/envs/SocialJax/bin/python tests/test_movement.py
"""

import jax
import jax.numpy as jnp
import numpy as onp

from socialjax.environments.movement import (
    positions_unique,
    resolve_movement,
    resolve_respawn,
)


def locs(*rows):
    return jnp.array(rows, dtype=jnp.int16)


def run(name, fn):
    fn()
    print(f"ok: {name}")


def test_head_on_same_target():
    # Two movers contest (0,1); exactly one advances, and over many keys the
    # win rate is close to 50/50 (no index bias).
    old = locs([0, 0, 0], [0, 2, 0])
    prop = locs([0, 1, 0], [0, 1, 0])
    wins0 = 0
    trials = 1000
    resolved = jax.jit(resolve_movement)
    for i in range(trials):
        out = resolved(jax.random.PRNGKey(i), old, prop)
        a0_won = bool(jnp.all(out[0, :2] == prop[0, :2]))
        a1_won = bool(jnp.all(out[1, :2] == prop[1, :2]))
        assert a0_won != a1_won, "exactly one agent must win the cell"
        assert bool(positions_unique(out))
        wins0 += a0_won
    rate = wins0 / trials
    assert 0.4 <= rate <= 0.6, f"agent-0 win rate {rate} outside [0.4, 0.6]"


def test_mover_vs_stayer_and_rotator():
    # Agent 1 stays put; agent 0 tries to enter its cell -> 0 reverts.
    old = locs([0, 0, 0], [0, 1, 0])
    prop = locs([0, 1, 0], [0, 1, 0])
    out = resolve_movement(jax.random.PRNGKey(0), old, prop)
    assert jnp.array_equal(out, old)

    # Agent 1 only rotates (rc unchanged, dir changes): keeps its cell AND
    # its new heading; the mover targeting it reverts.
    prop = locs([0, 1, 0], [0, 1, 3])
    out = resolve_movement(jax.random.PRNGKey(0), old, prop)
    assert jnp.array_equal(out[0], old[0])
    assert jnp.array_equal(out[1], prop[1])


def test_swap_blocked():
    old = locs([0, 0, 0], [0, 1, 2])
    prop = locs([0, 1, 0], [0, 0, 2])
    for i in range(20):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert jnp.array_equal(out, old), "swap must revert both agents"


def test_train_allowed():
    # A vacates (0,0) -> (0,1); B follows into (0,0). Both succeed.
    old = locs([0, 0, 0], [1, 0, 0])
    prop = locs([0, 1, 0], [0, 0, 0])
    for i in range(20):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert jnp.array_equal(out, prop)
        assert bool(positions_unique(out))


def test_cascade_from_stayer():
    # N stays on (1,2); A and B both try to enter it -> both revert.
    # C had targeted B's old cell, D had targeted C's old cell -> cascade.
    old = locs([0, 2, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0], [3, 1, 0])
    #          A            B          N (stays)   C          D
    prop = locs([1, 2, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0], [2, 1, 0])
    for i in range(20):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert jnp.array_equal(out, old), "whole chain must revert"


def test_cascade_from_random_loser():
    # A (0,2)->(1,2) and B (1,1)->(1,2) contest; the loser reverts, and if B
    # loses, C (2,1)->(1,1) and D (3,1)->(2,1) must cascade-revert behind it.
    old = locs([0, 2, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0])
    prop = locs([1, 2, 0], [1, 2, 0], [1, 1, 0], [2, 1, 0])
    saw_b_lose = saw_b_win = False
    for i in range(200):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert bool(positions_unique(out))
        b_won = bool(jnp.all(out[1, :2] == prop[1, :2]))
        if b_won:
            saw_b_win = True
            # B advanced, so its old cell is free: C and D follow through.
            assert jnp.array_equal(out, prop.at[0].set(old[0]))
        else:
            saw_b_lose = True
            # B reverted onto (1,1): C loses that cell, D loses (2,1).
            assert jnp.array_equal(out, old.at[0].set(prop[0]))
    assert saw_b_win and saw_b_lose, "both outcomes should occur across keys"


def test_rotation_cycle_allowed():
    # 3-cycle A->B->C->A: pairwise distinct targets, no 2-swap -> all move.
    old = locs([0, 0, 0], [0, 1, 0], [1, 1, 0])
    prop = locs([0, 1, 0], [1, 1, 0], [0, 0, 0])
    for i in range(20):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert jnp.array_equal(out, prop)


def test_swap_plus_contest():
    # A<->B swap while C also targets A's proposed cell. Swap kills A and B;
    # B reverts onto (0,1), so C's contest is now against a stayer -> C
    # reverts too. No pass-through survivor, uniqueness holds.
    old = locs([0, 0, 0], [0, 1, 0], [1, 1, 0])
    prop = locs([0, 1, 0], [0, 0, 0], [0, 1, 0])
    for i in range(20):
        out = resolve_movement(jax.random.PRNGKey(i), old, prop)
        assert jnp.array_equal(out, old)
        assert bool(positions_unique(out))


def test_property_random():
    # Random boards: uniqueness, per-row ∈ {old, proposed}, non-movers keep
    # their proposal (rotation preserved), and no surviving swap pair.
    board = 10
    trials = 10_000
    deltas = jnp.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=jnp.int16
    )

    for n in (2, 4, 8, 16):
        def one(key, n=n):
            k_pos, k_act, k_dir, k_res = jax.random.split(key, 4)
            cells = jax.random.permutation(k_pos, board * board)[:n]
            rc = jnp.stack([cells // board, cells % board], -1).astype(jnp.int16)
            dirs = jax.random.randint(k_dir, (n, 1), 0, 4, dtype=jnp.int16)
            old = jnp.concatenate([rc, dirs], -1)
            act = jax.random.randint(k_act, (n,), 0, 5)
            new_rc = jnp.clip(rc + deltas[act], 0, board - 1)
            new_dirs = jnp.where(
                (act == 0)[:, None],
                jax.random.randint(k_dir, (n, 1), 0, 4, dtype=jnp.int16),
                dirs,
            )
            prop = jnp.concatenate([new_rc, new_dirs], -1)
            out = resolve_movement(k_res, old, prop)
            return old, prop, out

        keys = jax.random.split(jax.random.PRNGKey(1234 + n), trials)
        old, prop, out = jax.jit(jax.vmap(one))(keys)

        assert bool(jnp.all(jax.vmap(positions_unique)(out))), n
        is_old = jnp.all(out == old, axis=-1)
        is_prop = jnp.all(out == prop, axis=-1)
        assert bool(jnp.all(is_old | is_prop)), n
        moved = jnp.any(prop[..., :2] != old[..., :2], axis=-1)
        assert bool(jnp.all(jnp.where(~moved, is_prop, True))), \
            "non-movers/rotators must keep their proposal"
        # no surviving swap: both moved to each other's old cells
        final_rc, old_rc = out[..., :2], old[..., :2]
        adv = jnp.any(final_rc != old_rc, axis=-1)
        took = jnp.all(
            final_rc[:, :, None, :] == old_rc[:, None, :, :], axis=-1
        )
        swap = took & jnp.swapaxes(took, 1, 2) & adv[:, :, None] & adv[:, None, :]
        swap = swap & ~jnp.eye(n, dtype=bool)[None]
        assert not bool(jnp.any(swap)), "a swap survived resolution"


def test_respawn():
    spawns = jnp.array(
        [[0, 0], [0, 3], [3, 0], [3, 3], [1, 1], [2, 2]], dtype=jnp.int16
    )
    # Survivors 0 and 2 sit on spawn cells; agents 1 and 3 respawn.
    agent_locs = locs([0, 0, 1], [9, 9, 0], [3, 3, 2], [9, 8, 0])
    reborn = jnp.array([False, True, False, True])
    for i in range(200):
        out = resolve_respawn(jax.random.PRNGKey(i), agent_locs, reborn, spawns)
        assert jnp.array_equal(out[0], agent_locs[0])
        assert jnp.array_equal(out[2], agent_locs[2])
        assert bool(positions_unique(out))
        for j in (1, 3):
            cell = out[j, :2]
            on_spawn = jnp.any(jnp.all(cell == spawns, axis=-1))
            assert bool(on_spawn), "reborn agent must land on a spawn cell"
            assert not bool(jnp.all(cell == agent_locs[0, :2]))
            assert not bool(jnp.all(cell == agent_locs[2, :2]))
            assert 0 <= int(out[j, 2]) < 3

    # Edge case S == N with N-1 survivors parked on spawn cells: the single
    # reborn agent must take the one free spawn cell.
    spawns4 = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int16)
    agent_locs = locs([0, 0, 0], [0, 1, 0], [1, 0, 0], [5, 5, 0])
    reborn = jnp.array([False, False, False, True])
    for i in range(50):
        out = resolve_respawn(jax.random.PRNGKey(i), agent_locs, reborn, spawns4)
        assert jnp.array_equal(out[3, :2], jnp.array([1, 1], dtype=jnp.int16))
        assert bool(positions_unique(out))


if __name__ == "__main__":
    run("head-on same target + no index bias", test_head_on_same_target)
    run("mover vs stayer / rotator", test_mover_vs_stayer_and_rotator)
    run("swap blocked", test_swap_blocked)
    run("train allowed", test_train_allowed)
    run("cascade from stayer", test_cascade_from_stayer)
    run("cascade from random loser", test_cascade_from_random_loser)
    run("rotation 3-cycle allowed", test_rotation_cycle_allowed)
    run("swap + contest interaction", test_swap_plus_contest)
    run("randomized property test", test_property_random)
    run("occupancy-aware respawn", test_respawn)
    print("ALL MOVEMENT TESTS PASSED")
