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


from socialjax.environments.common_harvest.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 4  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COIN_TYPES = 1
INTERACT_THRESHOLD = 0


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    apples: jnp.ndarray

    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray
    smooth_rewards: jnp.ndarray

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
    stay = 6


class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    red_apple = 3
    green_apple = 4

    
char_to_int = {
    'W': 1,
    ' ': 0,  # space 0
    'C': 3,
    'P': 4,
    'Q': 5
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
###################################################

class CoinGame(MultiAgentEnv):

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=2,
        shared_rewards=True,
        payoff_matrix=[[1, 1, -2], [1, 1, -2]],
        regrow_rate=0.0005,
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        enable_smooth_rewards=False,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        jit=True,
        
        grid_size=(16,11),
        obs_size=11,
        cnn=True,
        map_ASCII = [
                "CCCCCCCCCCC",
                "CPCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCCC",
                "CCCCCCCCCPC",
                "CCCCCCCCCCC",
            ]
    ):

        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        # self.agents = [str(i) for i in list(range(num_agents))]

        self.payoff_matrix = payoff_matrix
        self.shared_rewards = shared_rewards
        self.cnn = cnn
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.enable_smooth_rewards = enable_smooth_rewards
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = grid_size[0]
        self.GRID_SIZE_COL = grid_size[1]
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1
        self.regrow_rate = regrow_rate
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
        self.SPAWNS_APPLE = find_positions(nums_map, 3)
        self.SPAWNS_PLAYERS = find_positions(nums_map, 4)


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

            x = jnp.where(direction == 0, x + (self.OBS_SIZE//2)-1, x)
            y = jnp.where(direction == 0, y, y)

            x = jnp.where(direction == 1, x, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE//2)-1, y)


            x = jnp.where(direction == 2, x - (self.OBS_SIZE//2)+1, x)
            y = jnp.where(direction == 2, y, y)


            x = jnp.where(direction == 3, x, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE//2)+1, y)

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


        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the environment."""
            # actions = self.action_set.take(indices=jnp.array([actions["0"], actions["agent_1"]]))
            actions = jnp.array(actions)

            # freeze check
            # actions = jnp.where(
            #     state.freeze.max(axis=-1) > 0,
            #     Actions.stay,
            #     actions
            # )
            key, subkey = jax.random.split(key)
            # regrow apple
            grid_apple = state.grid
            probability = self.regrow_rate
            def regrow_green_apple(apple_locs, p):
                new_apple = jnp.where((((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (p < probability)) 
                                       | ((grid_apple[apple_locs[0], apple_locs[1]] == Items.green_apple))),  
                                      Items.green_apple, grid_apple[apple_locs[0], apple_locs[1]])
                return new_apple
            prob = jax.random.uniform(key, shape=(len(self.SPAWNS_APPLE),))
            new_apple = jax.vmap(regrow_green_apple)(self.SPAWNS_APPLE, prob)
            new_apple_grid = grid_apple.at[self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]].set(new_apple[:])
            state = state.replace(grid=new_apple_grid)


            grid_apple = state.grid
            def regrow_red_apple(apple_locs, p):
                new_apple = jnp.where((((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (p < probability)) 
                                       | ((grid_apple[apple_locs[0], apple_locs[1]] == Items.red_apple))),  
                                      Items.red_apple, grid_apple[apple_locs[0], apple_locs[1]])
                return new_apple
            prob = jax.random.uniform(subkey, shape=(len(self.SPAWNS_APPLE),))
            new_apple = jax.vmap(regrow_red_apple)(self.SPAWNS_APPLE, prob)
            new_apple_grid = grid_apple.at[self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]].set(new_apple[:])
            state = state.replace(grid=new_apple_grid)


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

            # update inventories
            def red_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.red_apple
                    ])
                return c_matches
            
            def green_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.green_apple
                    ])
                return c_matches


            red_apple_matches = jax.vmap(red_matcher)(p=new_locs)
            green_apple_matches = jax.vmap(green_matcher)(p=new_locs)


            red_red_reward = self.payoff_matrix[0][0]
            red_green_reward = self.payoff_matrix[0][1]
            red_penalty = self.payoff_matrix[0][2]
            green_red_reward = self.payoff_matrix[1][0]
            green_green_reward = self.payoff_matrix[1][1]
            green_penalty = self.payoff_matrix[1][2]

            red_reward, green_reward = 0, 0

            red_red_matches = red_apple_matches[0, :]
            red_green_matches = green_apple_matches[0, :]
            
            # jnp.all(
            #     new_red_pos == state.blue_coin_pos, axis=-1
            # )

            green_red_matches = red_apple_matches[1, :]
            # jnp.all(
            #     new_blue_pos == state.red_coin_pos, axis=-1
            # )
            green_green_matches = green_apple_matches[1, :]
            # jnp.all(
            #     new_blue_pos == state.blue_coin_pos, axis=-1
            # )

            red_reward = jnp.where(
                red_red_matches, red_reward + red_red_reward, red_reward
            )
            red_reward = jnp.where(
                red_green_matches, red_reward + red_green_reward, red_reward
            )
            red_reward = jnp.where(
                green_red_matches, red_reward + red_penalty, red_reward
            )

            green_reward = jnp.where(
                green_red_matches, green_reward + green_red_reward, green_reward
            )
            green_reward = jnp.where(
                green_green_matches, green_reward + green_green_reward, green_reward
            )
            green_reward = jnp.where(
                red_green_matches, green_reward + green_penalty, green_reward
            )

            rewards = jnp.zeros((2, 1))

            rewards = rewards.at[0, 0].set(red_reward[0])
            rewards = rewards.at[1, 0].set(green_reward[0])

            # # single reward or sum reward

            # rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
            # rewards_sum = jnp.sum(rewards)
            # rewards_sum_all_agents += rewards_sum
            # rewards = rewards_sum_all_agents

            # # new_invs = state.agent_invs + apple_matches

            # # state = state.replace(
            # #     agent_invs=new_invs
            # # )

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

            # update agent locations
            state = state.replace(agent_locs=new_locs)


            if self.shared_rewards:
                rewards = jnp.zeros((2, 1))
                rewards = rewards.at[0, 0].set(red_reward[0])
                rewards = rewards.at[1, 0].set(green_reward[0])
                rewards_sum = jnp.sum(rewards)
                rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
                rewards_sum_all_agents += rewards_sum
                rewards = rewards_sum_all_agents
                info = {
                    "original_rewards": rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.inequity_aversion:
                rewards = jnp.zeros((2, 1))
                rewards = rewards.at[0, 0].set(red_reward[0])
                rewards = rewards.at[1, 0].set(green_reward[0])
                original_rewards = rewards * self.num_agents
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
                else:
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.svo:
                rewards = jnp.zeros((2, 1))
                rewards = rewards.at[0, 0].set(red_reward[0])
                rewards = rewards.at[1, 0].set(green_reward[0])
                rewards = rewards * self.num_agents
                rewards, theta = self.get_svo_rewards(rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            else:
                rewards = jnp.zeros((2, 1))
                rewards = rewards.at[0, 0].set(red_reward[0])
                rewards = rewards.at[1, 0].set(green_reward[0])
                rewards = rewards * self.num_agents
                info = {}
            
            eat_own_coins = jnp.zeros((2, 1))
            red_reward, green_reward = 0, 0
            red_reward = jnp.where(
                red_red_matches, red_reward + red_red_reward, red_reward
            )

            green_reward = jnp.where(
                green_green_matches, green_reward + green_green_reward, green_reward
            )

            eat_own_coins = eat_own_coins.at[0, 0].set(red_reward[0])
            eat_own_coins = eat_own_coins.at[1, 0].set(green_reward[0])
            info["eat_own_coins"] = eat_own_coins.squeeze() * self.num_agents

            # if self.shared_rewards:
            #     rewards = jnp.zeros((2, 1))
            #     rewards = rewards.at[0, 0].set(red_reward[0])
            #     rewards = rewards.at[1, 0].set(green_reward[0])
            #     rewards_sum = jnp.sum(rewards)
            #     rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
            #     rewards_sum_all_agents += rewards_sum
            #     rewards = rewards_sum_all_agents
            # else:
            #     rewards = jnp.zeros((2, 1))
            #     rewards = rewards.at[0, 0].set(red_reward[0])
            #     rewards = rewards.at[1, 0].set(green_reward[0])
            #     rewards = rewards * self.num_agents


            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                smooth_rewards=state.smooth_rewards
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


            agent_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYERS)

            apple_pos = self.SPAWNS_APPLE

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
                apples=apple_pos,

                freeze=freeze,
                reborn_locs = agent_locs,
                smooth_rewards=jnp.zeros((self.num_agents, 1))
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

        img = onp.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=(70, 55, 40),
                dtype=onp.uint8,
            )

    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.red_apple:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        elif obj == Items.green_apple:
            # Draw the blue coin as DEFECT/ RED COIN
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (39.0, 214.0, 40.0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

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

        # time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, *agent_inv, time), axis=0)
        # img = onp.concatenate((img, time), axis=0)
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