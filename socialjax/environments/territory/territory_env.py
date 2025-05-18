from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces


from socialjax.environments.territory.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 4  # empty (0), wall (1), resource unclaimed (2), resource claimed (3)
NUM_COIN_TYPES = 1
INTERACT_THRESHOLD = 0

@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    resources: jnp.ndarray
    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray
    claimed_resources: jnp.ndarray
    claimed_coord: jnp.ndarray
    claimed_indicator_time_matrix: jnp.ndarray

@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6  # Use claim beam
    zap_forward = 7    # Use zap beam
    claim = 8

class Items(IntEnum):
    empty = 0
    wall = 1
    spawn_point = 2
    wall_highlight_nw = 3
    wall_highlight_ne = 4
    wall_highlight_se = 5
    wall_highlight_sw = 6
    resource_unclaimed = 7
    resource_claimed = 8
    interact = 9

char_to_int = {
    " ": 0,
    "W": 1,
    "P": 2,
    "F": 3,
    "T": 4,
    "J": 5,
    "L": 6,
    "R": 7,
}

ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
        # [0, 0, 0],
        # [0, 0, 0],
        # [0, 0, 0],
        # [0, 0, 0],
        # [0, 0, 0],
        # [0, 0, 1], # turn left
        # [0, 0, -1],  # turn right
        # [0, 0, 0]
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],  
        [0, -1, 0],  
        [1, 0, 0],  
        [-1, 0, 0],  
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=jnp.int8,
)



def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)

class Territory_open(MultiAgentEnv):
    """Territory environment where agents claim resources and can zap each other."""
    tile_cache: Dict[Tuple[Any, ...], Any] = {}
    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=9,
        shared_rewards=True,
        grid_size=(23,39),
        obs_size=11,
        cnn=True,
        jit=True,
        # map_ASCII = [
        #         "JRRRRRLJRRRRRLJRRRRRL",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,P,,RR,,P,,RR,,P,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "TRRRRRFTRRRRRFTRRRRRF",
        #         "JRRRRRLJRRRRRLJRRRRRL",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,P,,RR,,P,,RR,,P,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "TRRRRRFTRRRRRFTRRRRRF",
        #         "JRRRRRLJRRRRRLJRRRRRL",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,P,,RR,,P,,RR,,P,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "R,,,,,RR,,,,,RR,,,,,R",
        #         "TRRRRRFTRRRRRFTRRRRRF"
        #     ]
        map_ASCII = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W,,RRRRR,,RR,,RR,,,,,,RR,,,,,,RR,,,,,,W",
    "W,,,,,RR,,,,,,RR,,,,,,RR,,,,,,,,,,,,,,W",
    "W,,,,,RR,,,,,,RR,,,,,,,,,,,,,,,,,,,,,,W",
    "W,RR,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,RR,W",
    "W,,,,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,,,,W",
    "W,,,,,RR,,,,,,,,,,RRRR,,,,R,,,,,,,,,,,W",
    "W,,RR,RR,,,,,,,,,,,,,,,,,,R,,,,,,,,,,,W",
    "W,,,,,RR,,,,,,,RR,,,,,,,,,R,,,,,,,,,,,W",
    "W,,,,,RRRR,,,,,,,,,,,,,,,,,,,,,RR,,,,,W",
    "W,,,,,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,W",
    "W,,RRRR,,,RRRRRR,,,,,,,,,,,RR,,,,R,,,,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,R,,,,W",
    "W,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,,P,,W",
    "W,,,,RR,,,,,,,,,,,,,,,,RR,,,,,,,P,,,,,W",
    "W,,,,,,,,,RR,,,,,,,,,,,,,,,,,,,,,P,,P,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,P,,,,,,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,,P,,,W",
    "W,,P,,,,P,,,P,,P,,,P,,,,P,P,,P,,P,,P,,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,W",
    "W,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
]
    ):
        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)
        self.shared_rewards = shared_rewards

        # self.agents = [str(i) for i in list(range(num_agents))]

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = grid_size[0]
        self.GRID_SIZE_COL = grid_size[1]
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1


        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(5)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            return a_positions

        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.SPAWN_PLAYER = find_positions(nums_map, 2)
        self.SPAWNS_WALL_HIGHLIGH_NW = find_positions(nums_map, 3)
        self.SPAWNS_WALL_HIGHLIGH_NE = find_positions(nums_map, 4)
        self.SPAWNS_WALL_HIGHLIGH_SE = find_positions(nums_map, 5)
        self.SPAWNS_WALL_HIGHLIGH_SW = find_positions(nums_map, 6)
        self.SPAWNS_RESOURCE_UNCLAIMED = find_positions(nums_map, 7)
        self.SPAWNS_WALL = find_positions(nums_map, 1)

        def rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            def scan_fn(
                    state,
                    idx
            ):

                key, conflicts, conflicts_matrix, step_arr = state

                return jax.lax.cond(
                    conflicts[idx] > 0,
                    lambda: _rand_interaction(
                        key,
                        conflicts,
                        conflicts_matrix,
                        step_arr
                    ),
                    lambda: (state, step_arr.astype(jnp.int16))
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, conflicts, conflicts_matrix, step_arr.astype(jnp.int16)),
                jnp.arange(self.num_agents)
            )

            final_itxs = ys[-1]
            return final_itxs

        def _rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            conflict_idx = jnp.nonzero(
                conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0][0]

            agent_conflicts = conflicts_matrix[conflict_idx]

            agent_conflicts_idx = jnp.nonzero(
                agent_conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0]
            max_rand = jnp.sum(agent_conflicts_idx > -1)

            # preparing random agent selection
            k1, k2 = jax.random.split(key, 2)
            random_number = jax.random.randint(
                k1,
                (1,),
                0,
                max_rand
            )

            # index in main matrix of agent of successful interaction
            rand_agent_idx = agent_conflicts_idx[random_number]

            # set that agent's bool to False, for inversion later
            new_agent_conflict = agent_conflicts.at[rand_agent_idx].set(False)

            # set all remaining True agents' values as the "empty" item
            step_arr = jnp.where(
                new_agent_conflict,
                Items.empty,
                step_arr
            ).astype(jnp.int16)

            # update conflict bools to reflect the post-conflict state
            _conflicts = conflicts.at[agent_conflicts_idx].set(0)
            conflicts = jnp.where(
                agent_conflicts,
                0,
                conflicts
            )
            conflicts = conflicts.at[conflict_idx].set(0)
            conflicts_matrix = jax.vmap(
                lambda c, x: jnp.where(c, x, jnp.array([False]*conflicts.shape[0]))
            )(conflicts, conflicts_matrix)

            # deal with next conflict
            return ((
                k2,
                conflicts,
                conflicts_matrix,
                step_arr
            ), step_arr)
        
        def fix_interactions(
            key: jnp.ndarray,
            all_interacts: jnp.ndarray,
            actions: jnp.ndarray,
            state: State,
            one_step: jnp.ndarray,
            two_step: jnp.ndarray,
            right: jnp.ndarray,
            left: jnp.ndarray
        ) -> jnp.ndarray:
            '''
            Function defining multi-interaction logic.
            
            Args:
                - key: jax key for randomisation
                - all_interacts: jnp.ndarray of bools, provisional interaction
                indicators, subject to interaction logic
                - actions: jnp.ndarray of ints, actions taken by the agents
                - state: State, env state object
                - one_step, two_step, right, left: jnp.ndarrays, all of the
                same length, where each index is the index of an agent, and
                the element at each index is the item found at that agent's
                respective target location in the grid.

                
            Returns:
                - (jnp.ndarray, jnp.ndarray) - Tuple, where index 0 contains
                the array of the final agent interactions and index 1 contains
                a multi-dimensional array of the updated freeze penalty
                matrix.
            '''
            # distance-based priority, else random

            # fix highest priority interactions; 2 one-step zaps on the same
            # agent - random selection of winner
            forward = jnp.logical_and(
                actions == Actions.zap,
                all_interacts
            )

            forward = jnp.where(
                forward,
                one_step,
                Items.empty
            )

            conflicts_matrix = check_interaction_conflict(forward) * (forward != Items.empty)[:, None]

            conflicts = jnp.clip(
                jnp.sum(conflicts_matrix, axis=-1),
                0,
                1
            )

            k1, k2, k3  = jax.random.split(key, 3)
            one_step = rand_interaction(
                k2,
                conflicts,
                conflicts_matrix,
                forward
            )

            # if an interaction 2 steps away (diagonally or straight ahead)
            # is simultaneously another agent's one-step interaction, clear
            # it for the agent that is further:

            # right targets
            _right = jnp.logical_and(
                actions == Actions.zap,
                all_interacts
            )

            _right = jnp.where(
                _right,
                right,
                Items.empty
            )

            right = jnp.where(
                jnp.logical_and(
                    _right == one_step,
                    jnp.isin(_right, self._agents)
                ),
                Items.empty,
                _right
            )

            # left targets
            _left = jnp.logical_and(
                actions == Actions.zap_left,
                all_interacts
            )

            _left = jnp.where(
                _left,
                left,
                Items.empty
            )

            left = jnp.where(
                jnp.logical_and(
                    _left == one_step,
                    jnp.isin(_left, self._agents)
                ),
                Items.empty,
                _left
            )

            # two step targets
            _ahead = jnp.logical_and(
                actions == Actions.zap_ahead,
                all_interacts
            )

            _ahead = jnp.where(
                _ahead,
                two_step,
                Items.empty
            )
            two_step = jnp.where(
                jnp.logical_and(
                    _ahead == one_step,
                    jnp.isin(_ahead, self._agents)
                ),
                Items.empty,
                _ahead
            )

            # if any interactions occur between equi-distant pairs of agents,
            # choose randomly:
            all_two_step = jnp.concatenate(
                [
                    jnp.expand_dims(two_step, axis=-1),
                    jnp.expand_dims(right, axis=-1),
                    jnp.expand_dims(left, axis=-1),
                ],
                axis=-1
            )
            same_dist = jnp.logical_and(
                jnp.isin(
                    actions,
                    jnp.array([
                        Actions.zap_ahead,
                        Actions.zap_right,
                        Actions.zap_left
                    ]),
                ),
                all_interacts
            )

            same_dist_targets = jax.vmap(
                lambda s, ats, a: jnp.where(
                    s,
                    ats[a-Actions.zap_ahead],
                    Items.empty
                ).astype(jnp.int16)
            )(same_dist, all_two_step, actions)

            two_step_conflicts_matrix = check_interaction_conflict(
                same_dist_targets
            ) * (same_dist_targets != Items.empty)[:, None]

            two_step_conflicts = jnp.clip(
                jnp.sum(two_step_conflicts_matrix, axis=-1),
                0,
                1
            )

            k1, k2  = jax.random.split(k3, 2)
            two_step_interacts = rand_interaction(
                k2,
                two_step_conflicts,
                two_step_conflicts_matrix,
                same_dist_targets
            )

            # combine one & two step interactions
            final_interactions = jnp.where(
                actions == Actions.zap_forward, # if action was a 1-step zap
                one_step, # use the one-step item list after it was filtered
                two_step_interacts # else use two-step item after filtering
            ) * (state.freeze.max(axis=-1) <= 0)
            
            # note, both arrays were
            # qualified for:
            # 1. having actually zapped,
            # 2. zapping an actual agent,
            # 3. the zapping agent being not-frozen,
            # 4. the zapping agent meets
            # the zapping inventory threshold,
            # 5. the target agent meets the inventory threshold,
            # 6. agents 2 steps away surrender to agents 1 step away from the
            # same target
            # 7. equidistant agents succeed randomly

            # if the zapped agent is frozen, disallow the interaction
            itx_bool = jnp.isin(final_interactions, self._agents)
            frzs = state.freeze.max(axis=-1)
            final_interactions = jax.vmap(lambda i, itx: jnp.where(
                    jnp.logical_and(
                        i, frzs[itx-len(Items)] > 0
                    ),
                    Items.empty,
                    itx
                ).astype(jnp.int16)
            )(itx_bool, final_interactions)

            # update freeze matrix
            new_freeze_idx_bool = jnp.isin(
                    final_interactions,
                    self._agents
            )

            new_freeze = jax.vmap(lambda i, i_t, itx: jnp.where(
                    i_t,
                    state.freeze[i].at[itx-len(Items)].set(freeze_penalty),
                    state.freeze[i]
                ).astype(jnp.int16)
            )(self._agents-len(Items), new_freeze_idx_bool, final_interactions)

            new_freeze = new_freeze.transpose()
            new_freeze = jax.vmap(lambda i, i_t, itx: jnp.where(
                    i_t,
                    new_freeze[i].at[itx-len(Items)].set(freeze_penalty),
                    new_freeze[i]
                ).astype(jnp.int16)
            )(self._agents-len(Items), new_freeze_idx_bool, final_interactions)

            return jnp.concatenate(
                [jnp.expand_dims(final_interactions, axis=0),
                new_freeze.transpose()],
                axis=0
            )

        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_interaction_conflict(
                items: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check conflicting interaction targets.
            
            Args:
                - items: jnp.ndarray, the agent itemss at the interaction
                targets.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.logical_and(
                    jnp.all(x == y),
                    jnp.isin(x, self._agents)
                ),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(items, items)

            return collisions

        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> Tuple[Tuple, jnp.ndarray]:
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (1,),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False) # <<<< PROBLEM LINE        
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)

        def to_dict(
                agent: int,
                obs: jnp.ndarray,
                agent_invs: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                inv_to_show: jnp.ndarray
            ) -> dict:
            '''
            Function to produce observation/state dictionaries.
            
            Args:
                - agent: int, number identifying agent
                - obs: jnp.ndarray, the combined grid observations for each
                agent
                - agent_invs: jnp.ndarray of current agents' inventories
                - agent_pickups: boolean indicators of interaction
                - inv_to_show: jnp.ndarray inventory to show to other agents
                
            Returns:
                - dictionary of full state observation.
            '''
            idx = agent - len(Items)
            state_dict = {
                "observation": obs,
                "inventory": {
                    "agent_inv": agent_invs,
                    "agent_pickups": agent_pickups,
                    "invs_to_show": jnp.delete(
                        inv_to_show,
                        idx,
                        assume_unique_indices=True
                    )
                }
            }

            return state_dict
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
            ):
            '''
            Function to enforce symmetry in observations & generate final
            feature representation; current agent is permuted to first 
            position in the feature dimension.
            
            Args:
                - grid: jax ndarray of current agent's obs grid
                - agent: int, an index indicating current agent number
                - angles: jnp.ndarray of current agents' relative orientation
                in current agent's obs grid
                - agent_pickups: jnp.ndarray of agents able to interact
                - state: State, the env state obj
            Returns:
                - grid with current agent [x] permuted to 1st position (after
                the first 4 "Items" features, so 5th position overall) in the
                feature dimension, "other" [x] agent 2nd, angle [x, x, x, x]
                3rd, pick-up [x] bool 4th, inventory [x, x] 5th, frozen [x]
                6th, for a final obs grid of shape (5, 5, 14) (additional 4
                one-hot places for 5 possible items)
            '''
            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent-len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= len(Items)
                )

                show_inv_idxs = jnp.where(
                    state.freeze[agent],
                    size=12, # since, in a setting where simultaneous interac-
                    fill_value=-1 # -tions can happen, only a max of 12 can
                )[0] # happen at once (zap logic), regardless of pop size

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.isin(item_idx-len(Items), show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[item_idx - len(Items)],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # check if agent is not the subject & is frozen & therefore
                # not possible to interact with
                frozen = jnp.where(
                    other_agent,
                    state.freeze[
                        item_idx-len(Items)
                    ].max(axis=-1) > 0,
                    0
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # build extension
                extension = jnp.concatenate(
                    [
                        agent_element,
                        other_agent,
                        angle,
                        picked_up,
                        inv_to_show,
                        frozen
                    ],
                    axis=-1
                )

                # build final feature vector
                final_vec = jnp.concatenate(
                    [x[:len(Items)-1], extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            x, y = x + self.PADDING, y + self.PADDING

            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)


            # x = jnp.where(direction == 3, x - (self.OBS_SIZE // 2), x)
            # y = jnp.where(direction == 3, y - (self.OBS_SIZE // 2), y)

            x = jnp.where(direction == 0, x + (self.OBS_SIZE//2)-1, x)
            y = jnp.where(direction == 0, y, y)

            x = jnp.where(direction == 1, x, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE//2)-1, y)


            x = jnp.where(direction == 2, x - (self.OBS_SIZE//2)+1, x)
            y = jnp.where(direction == 2, y, y)


            x = jnp.where(direction == 3, x, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE//2)+1, y)

            # x = jnp.where(direction == 1, x - (self.OBS_SIZE // 2), x)
            # y = jnp.where(direction == 1, y - (self.OBS_SIZE - 1), y)

            # x = jnp.where(direction == 2, x - (self.OBS_SIZE - 1), x)
            # y = jnp.where(direction == 2, y - (self.OBS_SIZE // 2), y)

            # x = jnp.where(direction == 2, x - (self.OBS_SIZE // 2), x)
            # x = jnp.where(direction == 3, x - (self.OBS_SIZE - 1), x)

            # y = jnp.where(direction == 0, y - (self.OBS_SIZE // 2), y)
            # y = jnp.where(direction == 2, y - (self.OBS_SIZE - 1), y)
            # y = jnp.where(direction == 3, y - (self.OBS_SIZE // 2), y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state
            )

            return grids



        def _interact(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_forward,
                        # Actions.zap_ahead
                    ]
                )
            )

            interact_idx = jnp.int16(Items.interact)

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))

            # calculate pickups
            # agent_pickups = state.agent_invs.sum(axis=-1) > -100

            def check_valid_zap(t, i):
                '''
                Check target agent exists, isn't frozen, can interact, and
                is not the zapping-agent's self.
                '''
                agnt_bool = jnp.isin(state.grid[t[0], t[1]], self._agents)
                self_bool = state.grid[t[0], t[1]] != i

                return jnp.logical_and(agnt_bool, self_bool)

            # check 1 ahead
            clip_row = partial(jnp.clip, a_min=0, a_max=self.GRID_SIZE_ROW - 1)
            clip_col = partial(jnp.clip, a_min=0, a_max=self.GRID_SIZE_COL - 1)

            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            # one_step_targets = jax.vmap(clip)(one_step_targets)
            # one_step_targets[:,0] = jax.vmap(clip_row)(one_step_targets[:,0])
            # one_step_targets[:,1] = jax.vmap(clip_col)(one_step_targets[:,1])

            one_step_interacts = jax.vmap(
                check_valid_zap
            )(t=one_step_targets, i=self._agents)

            # check 2 ahead
            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)

            # two_step_targets = jax.vmap(clip)(two_step_targets)
            # two_step_targets[:,0] = jax.vmap(clip_row)(two_step_targets[:,0])
            # two_step_targets[:,1] = jax.vmap(clip_col)(two_step_targets[:,1])

            two_step_interacts = jax.vmap(
                check_valid_zap
            )(t=two_step_targets, i=self._agents)

            # check forward-right & manually check out-of-bounds
            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )

            right_interacts = jax.vmap(
                check_valid_zap
            )(t=target_right, i=self._agents)

            # check forward-left & manually check out-of-bounds
            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )

            left_interacts = jax.vmap(
                check_valid_zap
            )(t=target_right, i=self._agents)


            # one_step_targets_exd = jnp.expand_dims(one_step_targets, axis=0)
            # two_step_targets_exd = jnp.expand_dims(two_step_targets, axis=0)
            # target_right_exd = jnp.expand_dims(target_right, axis=0)
            # target_left_exd = jnp.expand_dims(target_left, axis=0)


            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            zaps_4_locs = jnp.concatenate((zaps, zaps, zaps, zaps), 0)

            resources_zaped = jnp.concatenate((one_step_targets, two_step_targets), 0)
            zaps_2_resources_locs = jnp.concatenate((zaps,zaps), 0)
            resources_zaped = resources_zaped.at[:, 2].set(jnp.concatenate([jnp.arange(1000, 1000+self.num_agents), jnp.arange(1000, 1000+self.num_agents)]))

            # resources_zaped = one_step_targets
            # zaps_2_resources_locs = zaps
            # resources_zaped = resources_zaped.at[:, 2].set(jnp.arange(1000, 1000+self.num_agents))
            
            # all_zaped_locs = jax.vmap(filter_zaped_locs)(all_zaped_locs)

            def check_coordinates(search_array, bool_matrix):
                # 先从bool矩阵获取True的坐标
                target_coords = jnp.stack(jnp.where(bool_matrix), axis=1)  # (N, 2)
                
                # 获取要检查的坐标
                coords_to_check = search_array[:, :2]  # (18, 2)
                
                # 比较坐标
                matches = (coords_to_check[:, None, :] == target_coords[None, :, :]).all(axis=-1)
                return matches.any(axis=1)
            def check_coordinates(search_array, bool_matrix):
                # 直接检查search_array中的每个坐标在bool_matrix中对应位置的值
                return bool_matrix[search_array[:, 0], search_array[:, 1]]

            def update_grid_values(grid, coords, mark_values, set_value):
                """
                更新二维网格值的函数，支持 JAX jit 编译。

                Args:
                    grid: jax.numpy 数组表示的二维网格。
                    coords: (N, 3) 的坐标矩阵，仅使用前两列。
                    mark_values: (N,) 的布尔数组，指示是否更新。
                    set_value: 如果需要更新时设置的标量值。

                Returns:
                    更新后的网格。
                """
                # 使用前两列 (x, y) 作为索引
                x, y = coords[:, 0], coords[:, 1]
                # 更新指定位置的值
                return grid.at[x, y].set(jnp.where(mark_values, set_value, grid[x, y]))

            break_pos = check_coordinates(resources_zaped,state.grid==999)
            break_pos = jnp.logical_and(break_pos, zaps_2_resources_locs.squeeze())
            grid = update_grid_values(state.grid, resources_zaped, break_pos,0)
            state = state.replace(grid=grid)



            exists_claimed_pos = check_coordinates(resources_zaped,state.claimed_coord)
            exists_claimed_pos = jnp.logical_and(exists_claimed_pos, zaps_2_resources_locs.squeeze())


            grid = update_grid_values(state.grid, resources_zaped, exists_claimed_pos,999)
            state = state.replace(grid=grid)
            


            def zaped_gird(a, z):
                return jnp.where(z, state.grid[a[0], a[1]], -1)

            all_zaped_gird = jax.vmap(zaped_gird)(all_zaped_locs, zaps_4_locs)
            # jax.debug.print("all_zaped_gird {all_zaped_gird} 🤯", all_zaped_gird=all_zaped_gird)

            def check_reborn_player(a):
                return jnp.isin(a, all_zaped_gird)
            
            reborn_players = jax.vmap(check_reborn_player)(self._agents)
            # jax.debug.print("reborn_players {reborn_players} 🤯", reborn_players=reborn_players)

            # all interacts = whether there is an agent at target & whether
            # agent is not frozen already & is qualified to interact
            # all_interacts = jnp.concatenate(
            #     [
            #         jnp.expand_dims(one_step_interacts, axis=-1),
            #         jnp.expand_dims(two_step_interacts, axis=-1),
            #         jnp.expand_dims(right_interacts, axis=-1),
            #         jnp.expand_dims(left_interacts, axis=-1)
            #     ],
            #     axis=-1
            # )
            # jax.debug.print("all_interacts {all_interacts} 🤯", all_interacts=all_interacts)

            # def check_reborn_players(i):
            #     return jnp.any(all_interacts[i,:] == True)
            
            # exists_vector = jax.vmap(check_reborn_players)(self._agents-6)

            # reborn_players = jnp.where(exists_vector is True, self._agents, -1)


            # interact_idxs = jnp.clip(actions - Actions.zap_forward, 0, 3)

            # all_interacts = all_interacts[jnp.arange(all_interacts.shape[0]), interact_idxs]# * zaps * agent_pickups * (state.freeze.max(axis=-1) <= 0)

                        # update grid with all zaps
            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )
            break_items_first = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        jnp.array(999).astype(jnp.int16)
                    )
            break_items_second = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        jnp.array(999).astype(jnp.int16)
                    )

            empty_item = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        jnp.array(Items.empty).astype(jnp.int16)
                    )
            qualified_to_zap = zaps.squeeze()
            # jax.debug.print("qualified_to_zap {qualified_to_zap} 🤯", qualified_to_zap=qualified_to_zap)
            # update grid
            def update_grid(a_i, t, i, grid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )
            # def update_grid(a_i, t, i, grid):
            #     return grid.at[t[:, 0], t[:, 1]].set(2)


            # jax.debug.print("one_step_targets {one_step_targets} 🤯", one_step_targets=one_step_targets)
            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid)

            # qualified_break_items = ((state.resources[:, None] == resources_zaped[:, :2]).all(axis=2)).any(axis=0)
            # qualified_break_items = jnp.logical_and(qualified_break_items, qualified_to_zap)
            # qualified_empty_items = state.grid[resources_zaped[:, 0], resources_zaped[:, 1]] == resources_zaped[:, 2]
            # aux_grid = update_grid(qualified_break_items, one_step_targets, break_items_first, aux_grid)
            # aux_grid = update_grid(qualified_break_items, two_step_targets, break_items_second, aux_grid)
            # aux_grid = update_grid(qualified_empty_items, one_step_targets, empty_item, aux_grid)
            # jax.debug.print("aux_grid {aux_grid} 🤯", aux_grid=aux_grid)
            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )

            # state = state.replace(grid=aux_grid)

            # jax.debug.print("state.grid {grid} 🤯", grid=state.grid)

            return reborn_players, state
        def _interact_fire_cleaning(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.claim,
                    ]
                )
            )

            # remove old interacts
            interact_index = jnp.arange(1000, 1000+self.num_agents)
            # claimed_resources_indicator = jnp.zeros((21, 21), dtype=bool).at[state.resources[:, 0], state.resources[:, 1]].set(True)
            claimed_resources_indicator = jnp.zeros((23, 39), dtype=bool).at[state.resources[:, 0], state.resources[:, 1]].set(True)
            mask = jnp.logical_and(jnp.isin(state.grid, interact_index),~claimed_resources_indicator)
            new_grid = jnp.where(mask, jnp.int16(Items.empty), state.grid)
            state = state.replace(grid=new_grid)
            # state = state.replace(grid=jnp.where(
            #     state.grid == interact_idx, state.init_grid, state.grid))
            


            # calculate pickups
            # agent_pickups = state.agent_invs.sum(axis=-1) > -100

            def check_valid_claim(beam):
                '''
                Check target agent exists, isn't frozen, can interact, and
                is not the zapping-agent's self.
                '''
                return jnp.any(jnp.all(state.resources == beam, axis=1))

            # check 1 ahead
            clip_row = partial(jnp.clip, a_min=0, a_max=self.GRID_SIZE_ROW - 1)
            clip_col = partial(jnp.clip, a_min=0, a_max=self.GRID_SIZE_COL - 1)

            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)


            # check 2 ahead
            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)

            # two_step_targets = jax.vmap(clip)(two_step_targets)
            # two_step_targets[:,0] = jax.vmap(clip_row)(two_step_targets[:,0])
            # two_step_targets[:,1] = jax.vmap(clip_col)(two_step_targets[:,1])

            # check forward-right & manually check out-of-bounds
            # target_right = jax.vmap(
            #     lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            # )(state.agent_locs)

            # right_oob_check = jax.vmap(
            #     lambda t: jnp.logical_or(
            #         jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
            #         (t < 0).any(),
            #     )
            # )(target_right)

            # target_right = jnp.where(
            #     right_oob_check[:, None],
            #     one_step_targets,
            #     target_right
            # )

            # right_interacts = jax.vmap(
            #     check_valid_zap
            # )(t=target_right, i=self._agents)

            # # check forward-left & manually check out-of-bounds
            # target_left = jax.vmap(
            #     lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            # )(state.agent_locs)

            # left_oob_check = jax.vmap(
            #     lambda t: jnp.logical_or(
            #         jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
            #         (t < 0).any(),
            #     )
            # )(target_left)

            # target_left = jnp.where(
            #     left_oob_check[:, None],
            #     one_step_targets,
            #     target_left
            # )

            # left_interacts = jax.vmap(
            #     check_valid_zap
            # )(t=target_right, i=self._agents)


            # one_step_targets_exd = jnp.expand_dims(one_step_targets, axis=0)
            # two_step_targets_exd = jnp.expand_dims(two_step_targets, axis=0)
            # target_right_exd = jnp.expand_dims(target_right, axis=0)
            # target_left_exd = jnp.expand_dims(target_left, axis=0)


            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            zaps_4_locs_judge = jnp.concatenate((zaps, zaps), 0)

            agent_index = jnp.concatenate([jnp.arange(1000,1000+self.num_agents), jnp.arange(1000,1000+self.num_agents)])
            all_zaped_locs = all_zaped_locs.at[:, 2].set(jnp.concatenate([jnp.arange(1000, 1000+self.num_agents), jnp.arange(1000, 1000+self.num_agents)]))

            # def check_coordinates(search_array, coordinates_180):
            #     coords_to_check = search_array[:, :2]    
            #     target_coords = coordinates_180       
            #     matches = (coords_to_check[:, None, :] == target_coords[None, :, :]).all(axis=-1)
            #     return matches.any(axis=1)
            def combined_check(grid, search_array, coordinates_180):
                """
                检查grid中的值并与坐标匹配
                
                Args:
                    grid: shape (24, 25) 的网格
                    search_array: shape (18, 3) 的搜索坐标
                    coordinates_180: shape (180, 3) 的目标坐标
                
                Returns:
                    shape (18,) 的布尔数组
                """
                # 检查18个坐标与180个坐标的匹配
                coords_to_check = search_array[:, :2]  # (18, 2)
                target_coords = coordinates_180[:, :2]  # (180, 2)
                matches = (coords_to_check[:, None, :] == target_coords[None, :, :]).all(axis=-1)  # (18, 180)
                coord_matches = matches.any(axis=1)  # (18,)
                
                # 检查grid中对应位置的值
                grid_values = grid.at[search_array[:, 0], search_array[:, 1]].get()
                grid_bool = grid_values != 0  # (18,)
                
                # 结合两个条件
                return jnp.logical_and(grid_bool, coord_matches)  # (18,)
            def combine_bool_masks(grid, coords, bool_array):
                # 找到grid中999的位置
                grid_bool = (grid != 999)
                
                # 根据coords获取这些位置在grid_bool中的值
                grid_values = grid_bool[coords[:, 0], coords[:, 1]]
                
                # 与bool_array做and操作
                return jnp.logical_and(grid_values, bool_array)
            
            exists_claimed_pos = combined_check(state.grid,all_zaped_locs,state.resources)
            exists_claimed_pos = jnp.logical_and(exists_claimed_pos, zaps_4_locs_judge.squeeze())
            exists_claimed_pos = combine_bool_masks(state.grid, all_zaped_locs, exists_claimed_pos)
            # potential_dirt_all_zap = all_zaped_locs[zaps_4_locs_judge]
            # exists = jnp.any(jnp.all(state.resources[:, None, :] == potential_dirt_all_zap[None, :, :2], axis=2), axis=0)
            # claimed_grid = potential_dirt_all_zap[exists]
            # agent_index = agent_index[exists]
            
            def claim_grid(search_array, exists, grid):
                pairs = jnp.stack([exists, search_array[:, 0], search_array[:, 1], search_array[:, 2]], axis=1)
                
                def step_fn(grid, pair):
                    flag, x, y, value = pair
                    return jax.lax.select(
                        flag,
                        grid.at[x, y].set(value),
                        grid
                    )
                return jax.lax.scan(lambda g, p: (step_fn(g, p), None), grid, pairs)[0]
            # def claim_grid(search_array, exists, grid, state):
            #     def update_single(exist_flag, coord):
            #         # 在resources中找到对应坐标的索引
            #         is_match = jnp.all(state.resources == coord[:2], axis=1)
            #         resource_idx = jnp.argmax(is_match)  # 获取匹配的索引
            #         return exist_flag, coord[0], coord[1], coord[2], resource_idx
                
            #     # vmap处理所有更新，返回mask、坐标和resource索引
            #     flags, x, y, values, resource_indices = jax.vmap(update_single)(exists, search_array)
            #     valid_coords = jnp.stack([x[flags], y[flags], values[flags]], axis=1)
            #     valid_indices = resource_indices[flags]
                
            #     # 更新grid
            #     new_grid = jax.lax.cond(
            #         flags.any(),
            #         lambda g: g.at[x[flags], y[flags]].set(values[flags]),
            #         lambda g: g,
            #         grid
            #     )
                
            #     # 更新claimed_resources在对应的resource索引位置
            #     new_claimed = jax.lax.cond(
            #         flags.any(),
            #         lambda cr: cr.at[valid_indices].set(valid_coords),
            #         lambda cr: cr,
            #         state.claimed_resources
            #     )
                
            #     return new_claimed, new_grid
            # new_claimed,new_grid = claim_grid(all_zaped_locs, exists_claimed_pos, state.grid, state)
            state = state.replace(grid=claim_grid(search_array=all_zaped_locs, exists=exists_claimed_pos, grid=state.grid))
            # state = state.replace(grid=new_grid, claimed_resources=new_claimed)
            return state
        
        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the environment."""
            # actions = self.action_set.take(indices=jnp.array([actions["0"], actions["agent_1"]]))
            actions = jnp.array(actions)

            

            # moving all agents

            new_grid = state.grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            x, y = state.reborn_locs[:, 0], state.reborn_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)
            state = state.replace(agent_locs=state.reborn_locs)

            # state = state.replace(reborn_locs=state.agent_locs)

            key, subkey = jax.random.split(key)
            all_new_locs = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array([self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], dtype=jnp.int16))(p=state.agent_locs, a=actions).squeeze()

            agent_move = (actions == Actions.up) | (actions == Actions.down) | (actions == Actions.right) | (actions == Actions.left)
            all_new_locs = jax.vmap(lambda m, n, p: jnp.where(m, n + STEP_MOVE[p], n))(m=agent_move, n=all_new_locs, p=actions)
            
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int16
                ),
            ).squeeze()

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    key,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    all_new_locs
                ),
                lambda: all_new_locs
            )

            # fix collision with the resources
            def check_wall_positions(grid, positions):
                def check_single_position(pos):
                    is_wall = grid[pos[0], pos[1]] != Items.empty
                    return jnp.where(is_wall, pos, jnp.array([-1, -1]))
                return jax.vmap(check_single_position)(positions)
            resources_position = check_wall_positions(state.grid, state.resources)
            wall_position = check_wall_positions(state.grid, self.SPAWNS_WALL)
            def handle_wall_collisions(original_positions, agent_positions, wall_positions):
                """
                agent_positions: shape (n_agents, 3) - (x, y, angle)
                wall_positions: shape (n_walls, 2) - (x, y)
                """
                def check_single_agent(original_pos,agent_pos):
                    # 只检查前两个坐标 (x,y)
                    agent_xy = agent_pos[:2]
                    # 检查是否与任何墙重叠
                    collisions = jnp.all(agent_xy == wall_positions, axis=1)
                    any_collision = jnp.any(collisions)
                    
                    # 如果碰撞，返回原位置；否则返回新位置
                    return jax.lax.cond(
                        any_collision,
                        lambda _: original_pos,  # 保持角度不变
                        lambda _: agent_pos,
                        operand=None
                    )
                
                # 使用 vmap 并行处理所有 agent
                return jax.vmap(check_single_agent)(original_positions, agent_positions)
            new_locs = handle_wall_collisions(state.agent_locs, all_new_locs, resources_position)
            new_locs = handle_wall_collisions(state.agent_locs, new_locs, wall_position)
            # new_locs = handle_wall_collisions(all_new_locs, new_locs, wall_position)


            def check_resource_interaction(
                agent_locs: jnp.ndarray,  # Shape: [num_agents, 3] - (x, y, direction)
                resource_locs: jnp.ndarray,  # Shape: [num_resources, 2] - (x, y)
                interaction_range: int = 1
            ) -> jnp.ndarray:  # Shape: [N, 3] where N is number of interactions, third column is agent index
                """
                Check which resources each agent is facing within range.
                Returns array of [resource_x, resource_y, agent_index] for each valid interaction.
                Returns empty array if no resources are faced.
                """
                directions = jnp.array([
                    [1, 0],  # North (0)
                    [0, 1],  # West  (1) 
                    [-1, 0],   # South (2)
                    [0, -1],   # East  (3)
                ])
                
                agent_pos = agent_locs[:, :2]
                agent_dirs = agent_locs[:, 2]
                agent_direction_vectors = directions[agent_dirs]
                
                def check_facing(pos, dir_vec, resource_pos):
                    return jnp.all(pos + dir_vec == resource_pos)
                
                # Vectorize the check across all agents and resources
                is_facing = jax.vmap(
                    jax.vmap(check_facing, in_axes=(None, None, 0)), 
                    in_axes=(0, 0, None)
                )(agent_pos, agent_direction_vectors, resource_locs)
                
                # Calculate distances
                rel_positions = jnp.expand_dims(resource_locs, 0) - jnp.expand_dims(agent_pos, 1)
                distances = jnp.sqrt(jnp.sum(rel_positions ** 2, axis=-1))
                in_range = distances <= interaction_range
                
                # Combine conditions
                can_interact = jnp.logical_and(in_range, is_facing)
                
                # Get indices where interactions are possible
                # agent_indices, resource_indices = jnp.where(can_interact)
                
                
                return can_interact

            def process_matches(bool_matrix, coord_matrix):
                # 对每一列，找出其中是否有True以及True在哪一行
                has_true = jnp.any(bool_matrix, axis=0)  # shape: (180,)
                true_rows = jnp.argmax(bool_matrix, axis=0)  # shape: (180,)
                
                # 构建结果数组
                return jnp.where(
                    has_true[:, None],  # 扩展维度以广播
                    jnp.column_stack((coord_matrix, true_rows)),  # True时：实际坐标和agent_index
                    # jnp.full((180, 3), 1000)  # False时：[1000,1000,1000]
                    jnp.full((88, 3), 1000)  # False时：[1000,1000,1000]
                )
            can_interact = check_resource_interaction(new_locs,state.resources)
            claimed_resources_matrix = process_matches(can_interact,state.resources)
            state = state.replace(claimed_resources=claimed_resources_matrix.astype(jnp.int16))


            # rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
            # rewards_sum = jnp.sum(rewards)
            # rewards_sum_all_agents += rewards_sum
            # rewards = rewards_sum_all_agents

            # new_invs = state.agent_invs + apple_matches

            # state = state.replace(
            #     agent_invs=new_invs
            # )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )
            x, y = new_locs[:, 0], new_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)


            exists_claimed_resources = ~jnp.all(state.claimed_resources == 1000, axis=1)
            claimed_resources_color_array = jnp.where(
                            state.claimed_resources[:, 2:3] == 1000,
                            state.claimed_resources,
                            jnp.column_stack((state.claimed_resources[:, :2], jnp.arange(1000,1000+self.num_agents)[state.claimed_resources[:, 2]]))
                            )
            def update_close_grid(search_array, exists, grid):
                def scan_fn(g, inputs):
                    coord, flag = inputs
                    return (
                        jax.lax.select(
                            flag,
                            g.at[coord[0], coord[1]].set(coord[2]),
                            g
                        ),
                        None
                    )
                return jax.lax.scan(scan_fn, grid, (search_array, exists))[0]
            state = state.replace(grid=update_close_grid(search_array=claimed_resources_color_array, exists=exists_claimed_resources,grid=state.grid))

            claimed_indicator_old_grid = state.grid

            # update agent locations
            state = state.replace(agent_locs=new_locs)
            
            color = jnp.arange(1000, 1000+self.num_agents)
            prev_claimed_resources = jnp.equal(state.grid[None, ...], color[:, None, None])
            
            state = _interact_fire_cleaning(key, state, actions)
            
            
            coord_claimed_resources =  jnp.isin(state.grid, color)
            state = state.replace(claimed_coord=coord_claimed_resources)

            reborn_players, state = _interact(key, state, actions)
            
            claimed_indicator_new_grid = state.grid


            def compare_grids(grid1, grid2, result):
                return jnp.where(jnp.equal(grid1, grid2), result + 1, 0)
            
            present_claimed_resources = jnp.equal(state.grid[None, ...], color[:, None, None])
            owner = jnp.argmax(prev_claimed_resources, axis=0)
            owner_expanded = jnp.expand_dims(owner, axis=0)
            agents = jnp.arange(prev_claimed_resources.shape[0]).reshape(prev_claimed_resources.shape[0], 1, 1)
            present_claimed_resources_updated = jnp.where(agents == owner_expanded, present_claimed_resources, False)

            claimed_indicator_time_matrix = compare_grids(claimed_indicator_old_grid, claimed_indicator_new_grid, state.claimed_indicator_time_matrix)
            rewards_matrix = jnp.logical_and(present_claimed_resources_updated,claimed_indicator_time_matrix[None, ...] >= 100)
            # rewards_matrix = jnp.logical_and(jnp.equal(state.grid[None, ...], color[:, None, None]),claimed_indicator_time_matrix[None, ...] >= 25)
            # rewards = jnp.sum(jnp.equal(state.grid[None, ...], color[:, None, None]), axis=(1, 2)) * 0.01
            # rewards = jnp.sum(rewards_matrix, axis=(1, 2)) * 0.01
            if self.shared_rewards:
                original_rewards = jnp.sum(rewards_matrix, axis=(1, 2)) * 0.01

                reward_sum = jnp.sum(original_rewards)
                rewards = jnp.full_like(original_rewards, reward_sum)/self.num_agents
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.inequity_aversion:
                original_rewards = jnp.sum(rewards_matrix, axis=(1, 2)) * 0.01
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
                else:
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.svo:      
                original_rewards = jnp.sum(rewards_matrix, axis=(1, 2)) * 0.01
                rewards, theta = self.get_svo_rewards(original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            else:
                rewards = jnp.sum(rewards_matrix, axis=(1, 2)) * 0.01
                info = {}
            

            # reward_sum = jnp.sum(rewards)
            # rewards = jnp.full_like(rewards, reward_sum)/self.num_agents

            reborn_players_3d = jnp.stack([reborn_players, reborn_players, reborn_players], axis=-1)


            re_agents_pos = jax.random.permutation(subkey, self.SPAWN_PLAYER)[:num_agents]

            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            re_agent_locs = jnp.array(
                [re_agents_pos[:, 0], re_agents_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T


            # jax.debug.print("reborn_players_3d {reborn_players_3d} 🤯", reborn_players_3d=reborn_players_3d)
            # jax.debug.print("new_locs {new_locs} 🤯", new_locs=new_locs)

            new_re_locs = jnp.where(reborn_players_3d == False, new_locs, re_agent_locs)
            new_re_locs = jnp.where(reborn_players_3d == False, new_locs, re_agent_locs)
            # jax.debug.print("new_re_locs {new_re_locs} 🤯", new_re_locs=new_re_locs)

            # new_grid = state.grid.at[
            #     state.agent_locs[:, 0],
            #     state.agent_locs[:, 1]
            # ].set(
            #     jnp.int16(Items.empty)
            # )

            # x, y = new_re_locs[:, 0], new_re_locs[:, 1]
            # new_grid = new_grid.at[x, y].set(self._agents)
            # state = state.replace(grid=new_grid)

            new_re_locs = jnp.where(reborn_players.any(), new_re_locs, state.agent_locs)
            # jax.debug.print("reborn_players.all() {reborn_players} 🤯", reborn_players=reborn_players.any())
            # jax.debug.print("new_re_locs111111111 {new_re_locs} 🤯", new_re_locs=new_re_locs)
            state = state.replace(reborn_locs=new_re_locs)

            # state.reborn_locs = new_re_locs
            # jax.debug.print("state.grid🤯 {state} 🤯", state=state.grid)

            # if self.shared_rewards:
            #     reward_sum = jnp.sum(rewards)
            #     rewards = jnp.full_like(rewards, reward_sum)/self.num_agents

            # else:
            #     rewards = rewards
          
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                resources=state.resources,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                claimed_resources = state.claimed_resources,
                claimed_coord = state.claimed_coord,
                claimed_indicator_time_matrix = claimed_indicator_time_matrix,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)

            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree_map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            # done = [reset_outer for _ in self.agents]
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards, dtype=jnp.int16),
                rewards
            )

            return (
                obs,
                state,
                rewards.squeeze(),
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)

            # all_positions = jax.random.permutation(subkey, self.SPAWNS)
            # player_reborn_pos = 
            # total_items = num_agents + NUM_COIN_TYPES * self.NUM_COINS

            # CHECK: doesn't use insider players, since the territory room has fix spawn location
            # inside_players_pos = jax.random.permutation(subkey, self.SPAWN_PLAYER)
            player_positions = self.SPAWN_PLAYER
            agent_pos = jax.random.permutation(subkey, player_positions)[:num_agents]
            wall_pos = self.SPAWNS_WALL
            wall_ne_pos = self.SPAWNS_WALL_HIGHLIGH_NE
            wall_se_pos = self.SPAWNS_WALL_HIGHLIGH_SE
            wall_sw_pos = self.SPAWNS_WALL_HIGHLIGH_SW
            wall_nw_pos = self.SPAWNS_WALL_HIGHLIGH_NW
            resource_pos = self.SPAWNS_RESOURCE_UNCLAIMED

            # resouces location
            grid = grid.at[
                resource_pos[:, 0],
                resource_pos[:, 1]
            ].set(jnp.int16(Items.resource_unclaimed))

            # wall location
            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))

            # wall highlight location
            grid = grid.at[
                wall_ne_pos[:, 0],
                wall_ne_pos[:, 1]
            ].set(jnp.int16(Items.wall_highlight_ne))

            # wall highlight location
            grid = grid.at[
                wall_se_pos[:, 0],
                wall_se_pos[:, 1]
            ].set(jnp.int16(Items.wall_highlight_se))
            
            # wall highlight location
            grid = grid.at[
                wall_sw_pos[:, 0],
                wall_sw_pos[:, 1]
            ].set(jnp.int16(Items.wall_highlight_sw))

            # wall highlight location
            grid = grid.at[
                wall_nw_pos[:, 0],
                wall_nw_pos[:, 1]
            ].set(jnp.int16(Items.wall_highlight_nw))

            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                resources=resource_pos,
                freeze=freeze,
                reborn_locs = agent_locs,
                # claimed_resources = jnp.ones((180,3), dtype=jnp.int16) * 1000,
                # claimed_coord = jnp.zeros((21,21), dtype=bool),
                # claimed_indicator_time_matrix = jnp.zeros((21,21), dtype=jnp.int16),
                claimed_resources = jnp.ones((88,3), dtype=jnp.int16) * 1000,
                claimed_coord = jnp.zeros((23,39), dtype=bool),
                claimed_indicator_time_matrix = jnp.zeros((23,39), dtype=jnp.int16),
            )

        def reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
        ################################################################################

        # for debugging
        self.get_obs = jax.jit(_get_obs)
        self.cnn = cnn

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE, (len(Items)-1) + 10)
            if self.cnn
            else (self.OBS_SIZE**2 * ((len(Items)-1) + 10),)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=onp.uint8,
        )

    # class Items(IntEnum):
        claimed_resources_color_array = jnp.arange(1000,1000+self.num_agents)
        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.resource_unclaimed:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))
        elif obj == Items.wall_highlight_ne or obj == Items.wall_highlight_se or obj == Items.wall_highlight_sw or obj == Items.wall_highlight_nw or obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (200.0, 200.0, 200.0))
        elif obj in claimed_resources_color_array:
            color_index = jnp.where(obj==claimed_resources_color_array)[0]
            fill_coords(img, point_in_rect(0, 1, 0, 1), self.PLAYER_COLOURS[int(color_index)])
        elif obj == 999:
            fill_coords(img, point_in_rect(0.1, 0.9, 0.3, 0.9), (117, 88, 71))
        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)

                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_locs[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False
                # for a in range(self.num_agents):
                #     agent_hat = (
                #         bool(state.agent_invs[a].sum() > INTERACT_THRESHOLD)
                #         if agent_here[a]
                #         else agent_hat
                #     )
                # if jnp.any(jnp.all(jnp.array([i,j]) == state.claimed_resources[:, :2], axis=1)):
                #     claimed_resources_array = jnp.arange(1000,1000+self.num_agents)
                #     agent_index = state.claimed_resources[jnp.where(jnp.all(jnp.array([i,j]) == state.claimed_resources[:, :2], axis=1))][0][-1]
                #     cell = claimed_resources_array[agent_index].item()
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        return img

    def render_inventory(self, inventory, width_px) -> onp.array:
        tile_height = 32
        height_px = NUM_COIN_TYPES * tile_height
        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // self.NUM_COINS
        for j in range(0, NUM_COIN_TYPES):
            num_coins = inventory[j]
            for i in range(int(num_coins)):
                cell = None
                if j == 0:
                    cell = 99
                elif j == 1:
                    cell = 100
                tile_img = self.render_tile(cell, tile_size=tile_height)
                ymin = j * tile_height
                ymax = (j + 1) * tile_height
                xmin = i * tile_width
                xmax = (i + 1) * tile_width
                img[ymin:ymax, xmin:xmax, :] = onp.resize(
                    tile_img, (tile_height, tile_width, 3)
                )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img

    def get_inequity_aversion_rewards_immediate(self, array, inner_t, target_agents=None, alpha=5, beta=0.05):
        """
        Calculate inequity aversion rewards using immediate rewards, based on equation (3) in the paper
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i^t for each agent
            target_agents: list of agent indices to apply inequity aversion
            alpha: inequity aversion coefficient (when other agents' rewards are greater than self)
            beta: inequity aversion coefficient (when self's rewards are greater than others)
        Returns:
            subjective_rewards: adjusted subjective rewards u_i^t after inequity aversion
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Calculate inequality using immediate rewards
        r_i = array  # [num_agents, 1]
        r_j = jnp.transpose(array)  # [1, num_agents]
        
        # Calculate inequality
        disadvantageous = jnp.maximum(r_j - r_i, 0)  # when other agents' rewards are higher
        advantageous = jnp.maximum(r_i - r_j, 0)     # when self's rewards are higher
        
        # Create mask to exclude self-comparison
        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask
        
        # Calculate inequality penalty
        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        # Calculate subjective rewards u_i^t = r_i^t - inequality penalty
        subjective_rewards = array - inequity_penalty

        subjective_rewards = jnp.where(jnp.all(array == 0), -(alpha + beta) * n_others, subjective_rewards)
        
        # Apply inequity aversion only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, subjective_rewards, array),jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)
        else:
            return subjective_rewards,jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)

    def get_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i for each agent
            w: SVO weight to balance self-reward and social value (0 <= w <= 1)
               w=0 means completely selfish, w=1 means completely altruistic
            ideal_angle_degrees: ideal angle in degrees
               - 45 degrees means complete equality
               - 0 degrees means completely selfish
               - 90 degrees means completely altruistic
            target_agents: list of agent indices to apply SVO
        
        Returns:
            shaped_rewards: rewards adjusted by SVO
            theta: reward angle in radians
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)  # [num_agents, num_agents]
        # Modified: use matrix multiplication to calculate other agents' rewards
        others_rewards = jnp.matmul(mask, array)  # [num_agents, 1]
        mean_others = others_rewards / (self.num_agents - 1)  # divide by number of other agents
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array  # [num_agents, 1]
        r_j = mean_others  # [num_agents, 1]
        theta = jnp.arctan2(r_j, r_i)
        
        # Calculate social value oriented utility
        # U(r_i, r_j) = r_i - w * |θ(R) - ideal_angle|
        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_standardized_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, array)
        mean_others = others_rewards / (self.num_agents - 1)
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)
        
        # Convert angle to [0, 2π] range
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
        
        # Calculate angle deviation and normalize to [0, 1] range
        angle_deviation = jnp.abs(theta - ideal_angle)
        angle_deviation = jnp.minimum(angle_deviation, 2 * jnp.pi - angle_deviation)  # take minimum deviation
        normalized_deviation = angle_deviation / jnp.pi  # normalize to [0, 1]
        
        # Use multiplicative form of penalty instead of subtraction
        svo_utility = r_i * (1 - w * normalized_deviation)
        
        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta