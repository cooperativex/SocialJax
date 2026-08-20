"""Shared, jit-safe agent-movement conflict resolution for all environments.

Semantics (Melting Pot 2.0 style):
  - Trains/following allowed: agent B may enter the cell agent A vacates in
    the same step, provided A's own move ultimately succeeds. If A is
    reverted, the revert cascades to B (and onward) until stable.
  - Same-target conflicts: an agent staying on the contested cell always
    keeps it; otherwise one moving contender is chosen uniformly at random
    and the rest revert to their old cells.
  - Swaps (2-cycles, A and B exchanging cells) are blocked: both revert.
  - Pure rotation cycles of length >= 3 (A->B->C->A) are allowed: they have
    no same-cell conflict, contain no 2-swap, and permute already-unique
    cells, so they cannot create an overlap.

Postcondition of both resolvers: the returned (row, col) pairs are pairwise
distinct, so stamping agents into a grid with ``grid.at[r, c].set(...)`` is
safe (no duplicate indices, no agent silently vanishing from observations).

The seven SocialJax-lineage environments bind ``step_env`` as
``__init__``-local jitted closures, so these are deliberately module-level
functions the closures call, not base-class methods.
"""

import jax
import jax.numpy as jnp


def resolve_movement(key, old_locs, proposed_locs):
    """Resolve agent-agent movement conflicts.

    Args:
      key: PRNGKey, consumed by one ``jax.random.permutation`` draw; the
        caller must pass a dedicated subkey.
      old_locs: (N, C) int array with C >= 2; columns 0,1 are (row, col).
        Precondition: the (row, col) pairs are pairwise distinct.
      proposed_locs: (N, C), same shape/dtype as ``old_locs``. Proposals
        must ALREADY have bounds clipping and wall/static-obstacle blocking
        applied (a blocked agent's proposed (row, col) equals its old one).
        Column 2+ (heading) may change freely on rotation-only actions.

    Returns:
      (N, C) array, same dtype as the inputs. Row i is either
      ``proposed_locs[i]`` (move/rotation succeeded) or ``old_locs[i]``
      (move reverted). Non-movers (including pure rotators) are never
      displaced. Final (row, col) pairs are pairwise distinct.

    Reverting the full row (heading included) is safe: only agents whose
    (row, col) changed can be reverted, and movement actions never change
    heading, so old and proposed headings agree for every revertible agent.
    """
    N = old_locs.shape[0]
    old_rc = old_locs[:, :2]
    prop_rc = proposed_locs[:, :2]
    moved = jnp.any(prop_rc != old_rc, axis=-1)  # (N,) movers only

    # One uniform random permutation acts as a random total order over the
    # agents: within any same-target group the winner is the perm-maximum,
    # making each contender equally likely to win — no agent-index bias.
    # (Outcomes of distinct groups are correlated through the shared
    # permutation, but each individual contest is uniform.)
    perm = jax.random.permutation(key, N)

    # Iterated revert to a fixed point. `alive[i]` means mover i is still
    # attempting its move; the body only ever clears bits, so the alive-set
    # is monotonically decreasing and stabilizes within N iterations —
    # a fixed trip count keeps this jit-friendly, and iterations past the
    # fixed point are no-ops.
    #
    # Correctness sketch:
    #  - An agent claiming its own (unique) old cell gets priority N, above
    #    every perm value, so a stayer always keeps its cell; distinct old
    #    cells mean two priority-N agents can never contest one cell.
    #  - When a mover reverts, next iteration it is "staying" on its old
    #    cell; any mover that had targeted that cell now loses the contest
    #    and reverts too — the cascade that fixes "following" overlaps.
    #    Each propagation kills >= 1 alive mover, bounding chains by N.
    #  - At the fixed point no two agents share a cell: two stayers cannot
    #    (old cells unique); a mover contesting a stayer's cell would have
    #    lost and left `alive`; two alive movers on one cell have distinct
    #    integer priorities, so one fails `wins_cell` — contradiction.
    def body(_, alive):
        claims = jnp.where(alive[:, None], prop_rc, old_rc)  # (N, 2)
        staying = jnp.all(claims == old_rc, axis=-1)  # non-movers + reverted
        pri = jnp.where(staying, N, perm)  # (N,)

        same = jnp.all(claims[:, None, :] == claims[None, :, :], axis=-1)
        group_max = jnp.max(jnp.where(same, pri[None, :], -1), axis=-1)
        wins_cell = pri == group_max

        # 2-cycle (swap) detection among still-alive movers: i proposes j's
        # old cell and vice versa -> both revert. A mover targeting the old
        # cell of a *staying* agent is not a swap; that is a same-cell
        # contest the stayer wins above.
        fwd = jnp.all(prop_rc[:, None, :] == old_rc[None, :, :], axis=-1)
        swap = (fwd & fwd.T) & alive[:, None] & alive[None, :]
        swap = swap.at[jnp.arange(N), jnp.arange(N)].set(False)
        in_swap = jnp.any(swap, axis=-1)

        return alive & wins_cell & ~in_swap

    alive = jax.lax.fori_loop(0, N, body, moved)
    reverted = moved & ~alive
    return jnp.where(reverted[:, None], old_locs, proposed_locs)


def resolve_respawn(key, agent_locs, reborn_mask, spawn_points, dir_maxval=3):
    """Occupancy-aware respawn: reborn agents land on unoccupied spawn cells.

    Args:
      key: PRNGKey (consumed; caller must pass a dedicated subkey).
      agent_locs: (N, 3) post-movement locations — survivors' final cells.
      reborn_mask: (N,) bool, True for agents to respawn this step.
      spawn_points: (S, 2) static spawn cells, S >= N, pairwise distinct.
      dir_maxval: exclusive upper bound for the random spawn heading.
        Defaults to 3 to preserve the existing environments' quirk of never
        respawning with heading 3.

    Returns:
      (N, 3) array, same dtype as ``agent_locs``. Survivors keep their rows
      verbatim; reborn agents get (spawn_row, spawn_col, random_dir) on
      cells unoccupied by any survivor. With S >= N, survivors occupy at
      most N - R spawn cells (R = number reborn), leaving >= R free ranked
      cells, so all N final (row, col) pairs are pairwise distinct.
    """
    k_perm, k_dir = jax.random.split(key)
    N = agent_locs.shape[0]
    S = spawn_points.shape[0]

    # Spawn cells occupied by survivors only (reborn agents vacate theirs).
    live_rc = agent_locs[:, :2]
    occupied = jnp.any(
        jnp.all(spawn_points[:, None, :] == live_rc[None, :, :], axis=-1)
        & ~reborn_mask[None, :],
        axis=-1,
    )  # (S,)

    # Random order over spawn points with unoccupied cells ranked first.
    # Integer sort keys (occupied * S + perm) have no ties, so the order is
    # exact and jit-safe — no boolean fancy-indexing needed.
    perm = jax.random.permutation(k_perm, S)
    order = jnp.argsort(occupied * S + perm)

    # The r-th reborn agent (in index order) takes the r-th ranked cell.
    reborn_rank = jnp.cumsum(reborn_mask) - 1  # valid where reborn
    chosen = spawn_points[order[jnp.clip(reborn_rank, 0, S - 1)]]  # (N, 2)

    dirs = jax.random.randint(k_dir, (N,), 0, dir_maxval, dtype=agent_locs.dtype)
    re_locs = jnp.concatenate(
        [chosen.astype(agent_locs.dtype), dirs[:, None]], axis=-1
    )
    return jnp.where(reborn_mask[:, None], re_locs, agent_locs)


def positions_unique(locs):
    """(N, >=2) -> bool scalar: True iff all (row, col) pairs are distinct."""
    rc = locs[:, :2]
    same = jnp.all(rc[:, None, :] == rc[None, :, :], axis=-1)
    n = rc.shape[0]
    return jnp.sum(same) == n
