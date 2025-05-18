import colorsys
from enum import IntEnum
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass

from socialjax.environments import spaces
from socialjax.environments.coop_mining.rendering import (
    render_tile, render_time, render_jax, render_time_jax, )
from socialjax.environments.multi_agent_env import MultiAgentEnv

# ------------------------------------------------------------------------
# ASCII map, CHAR_TO_INT, Items & Actions
# ------------------------------------------------------------------------
ASCII_MAP_COOP_MINING = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WOPOOOOOOOOOPOOOOOPOOOOOPOW
WOOOOOOOOWOOOOOOOOOOOOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWWWWWWWOOOOWOOOPOW
WOPOWWOOOOWOOOOOOOOOWOOOOOW
WOOOOOOOOOWOOPOOOOOOOOOOOOW
WOOOOOOOOOWOOOOOWWWOOOOOOOW
WOOOOOOOOOWOOOOOOOOOOOOOOOW
WOOOOOOOOOOOOOOOOOOOOOOOPOW
WOPOOOWWWOOOOOOWWWWWWWWOOOW
WOOWWWWOOOOOOOOOOOOOOOOOOOW
WOOOOOWOOOOWOOOOOPOOOOOOOOW
WOOOOOWOOOOWOOOOOOOOOOOOPOW
WOOOOOWOOOOOWOOOOOOOOWOOOOW
WOOOOOOWOOOOOWWWWOOOOWOOOOW
WOPOOOOOWOOOOOOOOOOOOWOOOOW
WOOOOOOOOWOOOPOOOOOOOOOOPOW
WOOOOOOOOOWOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOWWWWWWWWOOOW
WOOOOWOOOOOOOOOOOOWOOOOOOOW
WOPOOOOOOPOOOOOOOPOOOOOOPOW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WWWWWWWWWWWWWWWWWWWWWWWWWWW
""".strip('\n')

CHAR_TO_INT = {
    'W': 1,  # wall
    'O': 2,  # ore_wait
    'P': 3,  # spawn point
    ' ': 0,  # floor
}


class Items(IntEnum):
    empty = 0
    wall = 1
    ore_wait = 2
    spawn_point = 3
    iron_ore = 4
    gold_ore = 5
    gold_partial = 6


class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    step_left = 2
    step_right = 3
    forward = 4
    backward = 5
    stay = 6
    mine = 7


ROTATIONS = jnp.array(
    [
        [0, 0, -1],  # turn left
        [0, 0, 1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # mine
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [-1, 0, 0],  # up
        [0, 1, 0],  # right
        [1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=jnp.int8,
)


# ------------------------------------------------------------------------
# For partial-mining logic, we keep a big array listing which agents
# have contributed to gold at (row, col).
# We'll also define a countdown for multi-step partial windows.
# ------------------------------------------------------------------------
@dataclass
class State:
    agent_locs: jnp.ndarray  # shape (num_agents, 3) => row, col, orientation
    grid: jnp.ndarray  # shape (R, C)
    occupant_grid: jnp.ndarray  # shape=(R,C) for occupancy codes
    ore_miners: jnp.ndarray  # shape (R, C, max_miners), e.g. storing agent IDs or -1
    partial_ore_countdown: jnp.ndarray  # shape (R, C) for multi-step partial.
    inner_t: int
    outer_t: int
    last_mined_positions: jnp.ndarray  # shape (num_agents, 2), positions mined in last step
    actions_last_step: jnp.ndarray  # shape (num_agents,) last actions taken
    smooth_rewards: jnp.ndarray


@dataclass
class ViewConfig:
    forward: int
    backward: int
    left: int
    right: int


def ascii_map_to_grid(ascii_map: str, mapping: Dict[str, int]) -> jnp.ndarray:
    lines = ascii_map.strip().split('\n')
    # Turn the entire string into a list of mapped ints
    # Using python list comprehension is fine for a one-time init
    array_of_ints = [[mapping.get(char, 0) for char in line] for line in lines]
    return jnp.array(array_of_ints, dtype=jnp.int32)


# ------------------- RENDERING HELPERS ----------------------------------
# We'll define a color mapping for each item:
ITEM_COLORS = {
    Items.empty: (220, 220, 220),  # light gray floor
    Items.wall: (127, 127, 127),  # darker gray
    Items.ore_wait: (200, 200, 170),  # tan
    Items.spawn_point: (180, 180, 250),  # lightish purple
    Items.iron_ore: (139, 69, 19),  # dark gray-brownish lumps
    Items.gold_ore: (180, 180, 40),  # golden lumps
    Items.gold_partial: (190, 190, 80),  # slightly brighter from gold_ore
}


def check_relative_orientation(
        env,
        agent_id: int,  # current agent's code, e.g. len(Items)+my_idx
        agent_locs: jnp.ndarray,  # shape (num_agents, 3), row/col/orient
        local_grid: jnp.ndarray,  # shape (H, W), integer codes for items/agents
) -> jnp.ndarray:
    """
    Returns a (H, W) array of relative orientations:
      - 0..3 if cell has another agent (orientation minus the current agent's orientation mod 4)
      - -1 otherwise.
    """
    # Convert agent_id to index in [0..num_agents-1]
    # Example: if len(Items)=7 and agent_id=10 => idx=3
    my_idx = agent_id - len(Items)
    my_dir = agent_locs[my_idx, 2]

    # In your environment, valid agent codes are in:
    #   [len(Items), len(Items)+env.num_agents - 1].
    # We'll define a helper function to find relative orientation.
    def relative_dir(cell_code):
        # Convert code -> agent index
        idx = cell_code - len(Items)  # 0..num_agents-1
        their_dir = agent_locs[idx, 2]
        return (their_dir - my_dir) % 4

    # We'll map each cell to 0..3 if it holds a different agent, else -1.
    # For each cell, we want:
    #   if cell_code >= len(Items) and cell_code < len(Items)+env.num_agents
    #      and cell_code != agent_id:
    #        angle = relative_dir(cell_code)
    #   else:
    #        angle = -1
    #
    # We'll implement that with a single jnp.where.
    def per_cell(cell_code):
        # Check if 'cell_code' is in agent-range (i.e. is an agent) and != me:
        is_other_agent = (
                (cell_code >= len(Items)) &
                (cell_code < len(Items) + env.num_agents) &
                (cell_code != agent_id)
        )
        # If is_other_agent, compute orientation; else -1
        angle = relative_dir(cell_code)
        return jnp.where(is_other_agent, angle, -1)

    # vmap over the 2D local_grid
    return jax.vmap(jax.vmap(per_cell))(local_grid)


def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """
    Rotates the local grid according to the agent’s orientation.
    orientation=0 => no rotation, orientation=1 => rotate 90°, etc.
    This code uses jnp.rot90 with axes=(0, 1).
    """
    orientation = agent_loc[2]

    # We’ll do an if-else chain via jnp.where. If orientation=1 => rotate once, etc.
    # One approach: define a small function that does “rot90 grid k times”.
    def rotk(g, k):
        return jnp.rot90(g, k=k, axes=(0, 1))

    g1 = jnp.where(orientation == 1, rotk(grid, 1), grid)
    g2 = jnp.where(orientation == 2, rotk(grid, 2), g1)
    g3 = jnp.where(orientation == 3, rotk(grid, 3), g2)

    return g3


def combine_channels_simple(
        local_grid_1hot: jnp.ndarray,  # shape (OBS_SIZE, OBS_SIZE, channels)
        angles: jnp.ndarray,  # shape (OBS_SIZE, OBS_SIZE)
        my_idx: int
) -> jnp.ndarray:
    """
    Minimal channel-combining:
      - The first (len(Items)-1) channels represent item type one-hots
        (assuming 0=empty => index 0 => all zeros => etc.)
      - Then we add 1 channel for “is the current agent here?”
      - Then 1 channel for “is any other agent here?”
      - Then 4 channels for orientation (one-hot of angles).

    local_grid_1hot has shape (OBS_SIZE, OBS_SIZE, X). We want to produce
    final shape (OBS_SIZE, OBS_SIZE, final_depth).
    """
    # Suppose len(Items)=7 => (Items.empty..Items.gold_partial).
    # The code “(grid - 1)” made channel 0 => “Items.empty”.
    # So the first (len(Items)-1) channels are your item type one-hots.
    num_item_channels = len(Items) - 1  # e.g. 6
    item_oh = local_grid_1hot[..., :num_item_channels]  # shape (OBS_SIZE, OBS_SIZE, 6)

    # The rest up to local_grid_1hot.shape[-1] could be agent occupant bits
    occupant_bits = local_grid_1hot[..., num_item_channels:]  # shape (OBS_SIZE, OBS_SIZE, num_agents)

    # occupant_bits[..., 0] => “this agent?” occupant_bits[..., 1..] => “others?”
    # but you might track which agent is which, so let’s do a simpler approach:
    # we’ll define “me” as occupant_bits[..., 0], “other” as sum(occupant_bits[..., 1..]).
    me = occupant_bits[..., my_idx]
    # sum all occupant bits except the first:
    other = jnp.clip(jnp.sum(occupant_bits, axis=-1) - me, 0, 1)

    # angles => shape (OBS_SIZE, OBS_SIZE). We one-hot => shape (...,4).
    angle_oh = jax.nn.one_hot(angles, 4, axis=-1, dtype=jnp.int8)
    # if angles<0 => mask out => all zeros:
    angle_oh = jnp.where(angles[..., None] < 0, 0, angle_oh)

    me = me[..., None]  # shape (OBS_SIZE, OBS_SIZE, 1)
    other = other[..., None]

    # Now combine: [ item channels, me, other, angle_oh ]
    final_obs = jnp.concatenate([item_oh, me, other, angle_oh], axis=-1)
    return final_obs


def generate_agent_colors(num_agents):
    """
    Generate distinct agent colors using HSV interpolation.
    """
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        # saturation=0.8, value=0.8 => produce bright distinct colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def check_collision(new_agent_locs: jnp.ndarray) -> jnp.ndarray:
    """
    Checks pairwise collisions among agents based on row and column positions.
    Returns a (num_agents, num_agents) boolean matrix where True indicates a collision.
    """
    pos = new_agent_locs[:, :2]  # (N, 2)
    pos1 = pos[:, None, :]  # (N, 1, 2)
    pos2 = pos[None, :, :]  # (1, N, 2)
    collisions = jnp.all(pos1 == pos2, axis=-1)  # (N, N)
    mask = ~jnp.eye(new_agent_locs.shape[0], dtype=bool)
    return collisions & mask


def local_to_global(ar, ac, direction, lr, lc, view_cfg):
    """
    Map a local (lr, lc) in [-view_cfg.backward..view_cfg.forward, -view_cfg.left..view_cfg.right]
    into global grid coords, given the agent’s orientation.
    """
    # First shift local coords so (0,0) is the agent’s position
    # by offsetting them around the agent.
    row_offset = lr - view_cfg.backward
    col_offset = lc - view_cfg.left

    # Then rotate based on orientation
    def up_fn(_):
        # Orientation 0 = up
        # Move row up by row_offset, col right by col_offset
        return jnp.array([ar - row_offset, ac + col_offset], dtype=jnp.int32)

    def right_fn(_):
        # Orientation 1 = right
        return jnp.array([ar + col_offset, ac + row_offset], dtype=jnp.int32)

    def down_fn(_):
        # Orientation 2 = down
        return jnp.array([ar + row_offset, ac - col_offset], dtype=jnp.int32)

    def left_fn(_):
        # Orientation 3 = left
        return jnp.array([ar - col_offset, ac - row_offset], dtype=jnp.int32)

    return jax.lax.switch(
        direction,  # integer 0..3
        [up_fn, right_fn, down_fn, left_fn],
        operand=None
    )


def find_first_ore(agent_loc, grid, mining_range):
    """
    Returns the first ore in front of the agent or -1 if none,
    stopping if we hit a wall or already found ore.
    """
    r, c, orient = agent_loc
    dr, dc = STEP[orient][:2]

    # We'll store (pos, reward, stopped) in the loop carry.
    def body_fn(i, carry):
        (pos, reward, stopped) = carry

        # If we've already stopped in a previous iteration, do nothing further.
        # Keep the same pos/reward.
        def no_op(_):
            return (pos, reward, True)

        # Otherwise check the next cell.
        def do_step(_):
            new_r = r + dr * (i + 1)
            new_c = c + dc * (i + 1)
            in_bounds = (0 <= new_r) & (new_r < grid.shape[0]) & (0 <= new_c) & (new_c < grid.shape[1])
            item = jnp.where(in_bounds, grid[new_r, new_c], Items.wall)

            # If it's a wall => we stop further searching
            is_wall = (item == Items.wall)

            # Check if it's an ore
            ore_items = jnp.array([Items.iron_ore, Items.gold_ore, Items.gold_partial], dtype=jnp.int32)
            is_ore = jnp.isin(item, ore_items)

            # If found ore => update pos & reward=1.0
            found = is_ore
            updated_pos = jnp.where(found, jnp.array([new_r, new_c], dtype=jnp.int32), pos)
            updated_reward = jnp.where(found, 1.0, reward)

            # Stopping if we see either a wall or found ore
            new_stopped = is_wall | found
            return (updated_pos, updated_reward, new_stopped)

        # If 'stopped' is already True, skip do_step
        return jax.lax.cond(stopped, no_op, do_step, None)

    # Initialize carry with pos=(-1,-1), reward=0.0, stopped=False
    init_carry = (jnp.array([-1, -1], dtype=jnp.int32), 0.0, False)

    final_pos, final_reward, _ = jax.lax.fori_loop(0, mining_range, body_fn, init_carry)
    return final_pos, final_reward


def find_mine_target(agent_loc, mine_flag, grid, mining_range):
    """
    Determines the mining target and reward for a single agent.

    Args:
      agent_loc: jnp.ndarray of shape (3,) representing (row, col, orientation).
      mine_flag: bool, whether the agent is performing the mine action.
      grid: jnp.ndarray representing the environment grid.
      mining_range: int, maximum number of tiles to mine ahead.

    Returns:
      pos: jnp.ndarray of shape (2,) representing (row, col) of the mined ore, or (-1, -1).
      reward: jnp.float32, 1.0 if an ore was mined, else 0.0.
    """
    pos, reward = jax.lax.cond(
        mine_flag,
        lambda: find_first_ore(agent_loc, grid, mining_range),
        lambda: (jnp.array([-1, -1], dtype=jnp.int32), 0.0)
    )
    return pos, reward


# ------------------- MAIN ENV -------------------------------------------
class CoopMining(MultiAgentEnv):
    def __init__(
            self,
            num_inner_steps=1000,
            num_outer_steps=1,
            num_agents=4,
            shared_rewards=True,
            inequity_aversion=False,
            inequity_aversion_target_agents=None,
            inequity_aversion_alpha=5,
            inequity_aversion_beta=0.05,
            enable_smooth_rewards=False,
            svo=False,
            svo_target_agents=None,
            svo_w=0.5,
            svo_ideal_angle_degrees=45,
            max_miners=4,  # how many miners can we store per gold cell
            min_gold_miners=2,  # how many distinct miners needed to finalize gold
            mining_range=3,
            reward_iron=1.0,
            reward_gold=8.0,
            gold_mining_window=3,
            regrowth_prob_iron=0.0002,
            regrowth_prob_gold=0.00008,
            cnn=True,
            jit=True,
            view_config=ViewConfig(forward=9, backward=1, left=5, right=5),
    ):
        super().__init__(num_agents=num_agents)
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

        self.view_config = view_config
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.shared_rewards = shared_rewards
        self.max_miners = max_miners
        self.min_gold_miners = min_gold_miners
        self.mining_range = mining_range
        self.iron_reward = reward_iron
        self.gold_reward = reward_gold
        self.partial_window = gold_mining_window
        self.regrowth_prob_iron = regrowth_prob_iron
        self.regrowth_prob_gold = regrowth_prob_gold
        self.cnn = cnn
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        self._agents = jnp.array(self.agents, dtype=jnp.int32) + len(Items)
        self._agent_colors = jnp.array(generate_agent_colors(num_agents), dtype=jnp.uint8)

        self.OBS_SIZE = view_config.forward + view_config.backward + 1
        self.PADDING = self.OBS_SIZE - 1
        self.interact_threshold = 0

        self._grid_base = ascii_map_to_grid(ASCII_MAP_COOP_MINING, CHAR_TO_INT)
        self.grid_shape = self._grid_base.shape
        self.GRID_SIZE_ROW = self.grid_shape[0]
        self.GRID_SIZE_COL = self.grid_shape[1]
        self.tile_size = 32
        self.item_tile_cache = {}
        self.base_tile_cache_jax = {}
        self.final_tile_cache_jax = {}

        # Precompute spawn points
        self._spawn_pts = jnp.argwhere(self._grid_base == Items.spawn_point)

        # Action spaces
        self.action_spaces = {
            i: spaces.Discrete(len(Actions)) for i in range(num_agents)
        }
        # Partial grid integer observation space
        height = view_config.forward + view_config.backward + 1
        width = view_config.left + view_config.right + 1
        local_view_shape = (height, width, 1)
        self.observation_spaces = {
            i: spaces.Box(low=0, high=1E9, shape=local_view_shape, dtype=jnp.uint8)
            for i in range(num_agents)
        }

        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(self.step_env)
            self.reset = jax.jit(self.reset)
            self._get_obs = jax.jit(self._get_obs)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = self.step_env
            self.reset = self.reset
            self._get_obs = self._get_obs
        ################################################################################

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def close(self):
        pass

    @property
    def name(self):
        return "CoopMining"

    def state_space(self) -> spaces.Box:
        # optional
        return spaces.Box(low=0, high=1, shape=(1,), dtype=jnp.uint8)

    def action_space(self, agent_id: Union[int, None] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """
        Returns the observation space for the environment.
        """
        obs_depth = len(Items) + 5
        height = self.view_config.forward + self.view_config.backward + 1
        width = self.view_config.left + self.view_config.right + 1
        obs_size = (height, width, obs_depth) if self.cnn else (height * width * obs_depth)
        return spaces.Box(low=0, high=1E9, shape=obs_size, dtype=jnp.uint8), obs_size

    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
        state = self._reset_state(key)
        obs = self._get_obs(state)
        return obs, state

    def _reset_state(self, key: jnp.ndarray) -> State:
        grid = jnp.copy(self._grid_base)
        occupant_grid = jnp.zeros_like(grid)
        spawn_pts = self._spawn_pts
        key, subkey = jax.random.split(key)
        chosen = jax.random.choice(subkey, spawn_pts.shape[0],
                                   shape=(self.num_agents,), replace=False)
        chosen_spawns = spawn_pts[chosen]

        # random orientation
        key, subkey = jax.random.split(key)
        directions = jax.random.randint(subkey, shape=(self.num_agents,),
                                        minval=0, maxval=4, dtype=jnp.int32)

        agent_locs = jnp.stack([
            chosen_spawns[:, 0],
            chosen_spawns[:, 1],
            directions
        ], axis=-1)

        # fill occupant_grid with each agent code
        occupant_grid = occupant_grid.at[agent_locs[:, 0], agent_locs[:, 1]].set(
            self._agents  # e.g. [7,8,9,10] for 4 agents if Items=7
        )

        # We define ore_miners array for partial gold: shape (R, C, MAX_MINERS)
        # Fill with -1 to indicate "no miner" in each slot
        ore_miners = -1 * jnp.ones(
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, self.max_miners),
            dtype=jnp.int32
        )
        # partial_ore_countdown for multi-step windows
        partial_ore_countdown = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), dtype=jnp.int32)

        # Initialize last actions and mined positions to -1
        last_mined_positions = -1 * jnp.ones((self.num_agents, 2), dtype=jnp.int32)
        last_actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)

        return State(
            agent_locs=agent_locs,
            grid=grid,
            occupant_grid=occupant_grid,
            ore_miners=ore_miners,
            partial_ore_countdown=partial_ore_countdown,
            inner_t=0,
            outer_t=0,
            last_mined_positions=last_mined_positions,
            actions_last_step=last_actions,
            smooth_rewards=jnp.zeros((self.num_agents, 1)),
        )

    def step_env(self, key: jnp.ndarray, state: State, actions: jnp.ndarray):
        """
        step_env for a CoopMining environment.

        Args:
          key: PRNGKey for any random draws (e.g. regrow ore).
          state: current State dataclass (agent positions, grid, partial ore states, etc.).
          actions: a tuple/list of ints (one per agent) from the ACTION_SET.

        Returns:
          obs: observations dict-of-dicts or array-of-arrays.
          next_state: updated State
          rewards: jnp.ndarray of shape (num_agents,) containing immediate reward.
          done: dict with done["__all__"] = bool, and possibly per-agent done flags
          info: extra info (dict)
        """
        actions = jnp.array(actions, dtype=jnp.int32).squeeze()

        # 1) Update orientation from turn actions
        new_orients = (state.agent_locs[:, 2] + ROTATIONS[actions][:, 2]) % 4

        # 2) Calculate new positions
        old_rc = state.agent_locs[:, :2]  # (N, 2)
        offsets = STEP_MOVE[actions][:, :2]  # (N, 2), ignoring orientation in last dimension if zero
        new_rc = old_rc + offsets  # (N, 2)

        # 3) Clip to grid bounds
        row_max, col_max = self.GRID_SIZE_ROW, self.GRID_SIZE_COL
        new_rc = jnp.clip(new_rc, a_min=jnp.array([0, 0]), a_max=jnp.array([row_max - 1, col_max - 1]))

        # 4. Handle walls
        wall_mask = (state.grid[new_rc[:, 0], new_rc[:, 1]] == Items.wall)  # (N,)
        final_rc = jnp.where(wall_mask[:, None], old_rc, new_rc)  # Revert positions where walls are

        # 5. Handle collisions
        collision_matrix = check_collision(final_rc)  # (N, N)
        collided = jnp.any(collision_matrix, axis=1)  # (N,)
        final_rc = jnp.where(collided[:, None], old_rc, final_rc)  # Revert positions where collisions occur

        # 6. Update agent locations
        new_locs = jnp.concatenate([final_rc, new_orients[:, None]], axis=-1)  # (N, 3)

        # 7. Identify mine actions
        mine_flags = (actions == Actions.mine)  # shape (num_agents,)

        # 8) Use the vectorized mining routine
        positions, rewards_iron, rewards_gold, new_grid, new_ore_miners, new_partial_cd = self.vectorized_mining(
            new_locs, mine_flags, state
        )

        # Check occupancy for all cells
        occupant_cleared = jnp.zeros_like(state.occupant_grid)
        occupant_with_agents = occupant_cleared.at[new_locs[:, 0], new_locs[:, 1]].set(self._agents)

        # agent_rows = new_locs[:, 0]  # (N,)
        # agent_cols = new_locs[:, 1]  # (N,)
        # row_max, col_max = self.GRID_SIZE_ROW, self.GRID_SIZE_COL
        # grid_with_agents = new_grid.at[agent_rows, agent_cols].set(self._agents)
        # occupied = jnp.zeros((row_max, col_max), dtype=bool).at[agent_rows, agent_cols].set(True)

        # Generate random numbers for regrowth
        rng_split = jax.random.split(key, num=2)
        rng_iron, rng_gold = rng_split
        # Apply regrowth
        new_grid = self.regrow_ore_vectorized(new_grid, rng_iron, rng_gold)

        # Aggregate Rewards
        if self.shared_rewards:
            total_rewards = jnp.sum(rewards_iron + rewards_gold)  # Scalar
            final_rewards = jnp.full((self.num_agents,), total_rewards)
            info = {
                "original_rewards": final_rewards.squeeze(),
                "shaped_rewards": final_rewards.squeeze(),
            }
        elif self.inequity_aversion:
            final_rewards = (rewards_iron + rewards_gold) * self.num_agents # (N,)
            if self.smooth_rewards:
                should_smooth = (state.inner_t % 1) == 0
                new_smooth_rewards = 0.99 * 0.99 * state.smooth_rewards + final_rewards
                rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                state = state.replace(smooth_rewards=new_smooth_rewards)
                info = {
                "original_rewards": final_rewards.squeeze(),
                "smooth_rewards": state.smooth_rewards.squeeze(),
                "shaped_rewards": rewards.squeeze(),
            }
            else:
                rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(final_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                info = {
                "original_rewards": final_rewards.squeeze(),
                "shaped_rewards": rewards.squeeze(),
            }
        elif self.svo:
            final_rewards = (rewards_iron + rewards_gold) * self.num_agents # (N,)
            rewards, theta = self.get_svo_rewards(final_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
            info = {
                "original_rewards": final_rewards.squeeze(),
                "svo_theta": theta.squeeze(),
                "shaped_rewards": rewards.squeeze(),
            }
        else:
            final_rewards = (rewards_iron + rewards_gold) * self.num_agents # (N,)
            info = {}

        info["mining_gold"] = rewards_gold * self.num_agents

        # if self.shared_rewards:
        #     total_rewards = jnp.sum(rewards_iron + rewards_gold)  # Scalar
        #     final_rewards = jnp.full((self.num_agents,), total_rewards)
        # else:
        #     final_rewards = (rewards_iron + rewards_gold) * self.num_agents # (N,)

        # 9) Build next_state
        new_state = State(
            agent_locs=new_locs,
            grid=new_grid,
            occupant_grid=occupant_with_agents,
            partial_ore_countdown=new_partial_cd,  # or updated if you track time windows
            ore_miners=new_ore_miners,
            inner_t=state.inner_t + 1,
            outer_t=state.outer_t,
            last_mined_positions=positions,  # Record mined positions
            actions_last_step=actions,
            smooth_rewards=state.smooth_rewards,
        )

        # 10) Check if done
        reset_inner = (new_state.inner_t >= self.num_inner_steps)
        new_outer = jnp.where(reset_inner, new_state.outer_t + 1, new_state.outer_t)
        reset_outer = (new_outer >= self.num_outer_steps)
        done_dict = {f"{i}": reset_outer for i in range(self.num_agents)}
        done_dict["__all__"] = reset_outer

        # 11) Observations
        obs = self._get_obs(new_state)

        # 12) Info
        # info = {}

        return obs, new_state, final_rewards, done_dict, info

    def regrow_ore_vectorized(self, items: jnp.ndarray, rng_iron, rng_gold) -> jnp.ndarray:
        # def regrow_ore_vectorized(self, items: jnp.ndarray, rng_iron, rng_gold, occupied: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized regrow function.
        """
        is_wait = (items == Items.ore_wait)

        # Iron regrowth
        draw_iron = jax.random.uniform(rng_iron, shape=items.shape)
        spawn_iron = is_wait & (draw_iron < self.regrowth_prob_iron)
        new_item = jnp.where(spawn_iron, Items.iron_ore, items)

        # Gold regrowth
        draw_gold = jax.random.uniform(rng_gold, shape=items.shape)
        spawn_gold = (new_item == Items.ore_wait) & (draw_gold < self.regrowth_prob_gold)
        new_item = jnp.where(spawn_gold, Items.gold_ore, new_item)

        return new_item

    def vectorized_mining(self, new_locs, mine_flags, state):
        """Fix concurrency by updating the environment after each agent's mining."""

        def mine_one_agent(agent_idx, carry):
            (grid, ore_miners, partial_cd,
             positions, iron_rewards, gold_rewards) = carry

            loc = new_locs[agent_idx]  # (row, col, orientation)
            do_mine = mine_flags[agent_idx]

            # Find the first ore if mining
            pos, _ = jax.lax.cond(
                do_mine,
                lambda _: find_first_ore(loc, grid, self.mining_range),
                lambda _: (jnp.array([-1, -1], dtype=jnp.int32), 0.0),
                operand=None
            )

            # Extract item
            in_bounds = (pos[0] >= 0) & (pos[1] >= 0)
            item = jnp.where(in_bounds, grid[pos[0], pos[1]], Items.empty)

            # Iron logic
            is_iron = (item == Items.iron_ore)
            r_iron = jnp.where(is_iron & do_mine, self.iron_reward, 0.0)
            new_item_if_iron = jnp.where(is_iron, Items.ore_wait, item)

            # Gold/partial logic
            is_gold_full = (new_item_if_iron == Items.gold_ore)
            is_gold_partial = (new_item_if_iron == Items.gold_partial)
            is_gold_like = is_gold_full | is_gold_partial
            mining_gold = is_gold_like & do_mine

            # Insert agent into ore_miners if mining gold
            def insert_miner(miners, agent_id):
                already_in = jnp.any(miners == agent_id)
                free_slot = jnp.argmax(miners < 0)
                new_miners = jnp.where(
                    already_in | (free_slot >= self.max_miners),
                    miners,
                    miners.at[free_slot].set(agent_id)
                )
                return new_miners

            old_miners = jnp.where(in_bounds,
                                   ore_miners[pos[0], pos[1]],
                                   -1 * jnp.ones(self.max_miners, dtype=jnp.int32))
            new_miners = jax.lax.cond(
                mining_gold,
                lambda: insert_miner(old_miners, agent_idx),
                lambda: old_miners
            )

            # Count distinct miners
            distinct_count = jnp.sum(new_miners >= 0)
            finalize = (distinct_count == self.min_gold_miners) & mining_gold
            revert = (distinct_count > self.min_gold_miners) & mining_gold
            partial = (distinct_count > 0) & (distinct_count < self.min_gold_miners) & mining_gold

            # Update item
            gold_item = jnp.where(partial, Items.gold_partial, new_item_if_iron)
            gold_item = jnp.where(finalize, Items.ore_wait, gold_item)
            gold_item = jnp.where(revert, Items.gold_ore, gold_item)

            # Gold reward only if we just finalized it
            r_gold = jnp.where(finalize, self.gold_reward, 0.0)

            # Update partial countdown
            old_cd = jnp.where(in_bounds, partial_cd[pos[0], pos[1]], 0)
            new_cd = old_cd
            new_cd = jnp.where(finalize | revert, 0, new_cd)
            new_cd = jnp.where((item == Items.gold_ore) & do_mine, self.partial_window, new_cd)

            # Write back updates if in bounds
            new_grid = jnp.where(in_bounds,
                                 grid.at[pos[0], pos[1]].set(gold_item),
                                 grid)
            new_ore_miners = jnp.where(in_bounds,
                                       ore_miners.at[pos[0], pos[1]].set(new_miners),
                                       ore_miners
                                       )
            new_partial_cd = jnp.where(in_bounds,
                                       partial_cd.at[pos[0], pos[1]].set(new_cd),
                                       partial_cd
                                       )

            # Store final results for this agent
            new_positions = positions.at[agent_idx].set(pos)
            new_iron_rewards = iron_rewards.at[agent_idx].set(r_iron)
            new_gold_rewards = gold_rewards.at[agent_idx].set(r_gold)

            return (new_grid, new_ore_miners, new_partial_cd,
                    new_positions, new_iron_rewards, new_gold_rewards), None

        # Initialize carry
        init_positions = jnp.full((self.num_agents, 2), -1, dtype=jnp.int32)
        init_iron_rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        init_gold_rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        carry_init = (state.grid,
                      state.ore_miners,
                      state.partial_ore_countdown,
                      init_positions,
                      init_iron_rewards,
                      init_gold_rewards)

        def scan_fn(carry, agent_idx):
            new_carry, _ = mine_one_agent(agent_idx, carry)
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            scan_fn,
            carry_init,
            jnp.arange(self.num_agents)
        )
        (final_grid, final_ore_miners, final_partial_cd,
         final_positions, final_iron_rewards, final_gold_rewards) = final_carry

        # Decrement partial gold timer
        final_partial_cd = jnp.maximum(final_partial_cd - 1, 0)

        return (final_positions,
                final_iron_rewards,
                final_gold_rewards,
                final_grid,
                final_ore_miners,
                final_partial_cd)

    def check_relative_orientation(
            self,
            agent_idx: int,  # Index of the current agent (0 to N-1)
            agent_locs: jnp.ndarray,  # (N, 3)
            grid: jnp.ndarray  # (H, W)
    ) -> jnp.ndarray:
        """
        Computes relative orientations of other agents from the perspective of the current agent.
        Returns a (H, W) array where each cell contains:
        - Relative orientation (0-3) if another agent is present
        - -1 if no agent is present
        """
        current_dir = agent_locs[agent_idx, 2]  # Current agent's orientation

        # Create a mask for other agents
        # Assuming self._agents are unique and correspond to agent indices
        agent_ids = self._agents  # e.g., [4, 5, 6, 7]
        offset = len(Items)
        # Compute agent indices based on grid values
        agent_indices = grid - offset  # Adjust based on your agent ID mapping
        # Valid agent indices are within [0, num_agents)
        valid = (agent_indices >= 0) & (agent_indices < agent_locs.shape[0])

        # Fetch orientations of other agents
        orientations = jnp.where(
            valid,
            agent_locs[agent_indices, 2],
            -1
        )  # (H, W)

        # Compute relative orientations
        relative_orient = (orientations - current_dir) % 4

        # Mask cells without other agents
        relative_orient = jnp.where(valid, relative_orient, -1)

        return relative_orient

    def combine_channels(self,
                         grid_1hot: jnp.ndarray,  # shape = (H, W, channels)
                         angles: jnp.ndarray  # shape = (H, W)
                         ) -> jnp.ndarray:
        num_item_channels = (len(Items) - 1)
        item_oh = grid_1hot[..., :num_item_channels]  # (A, H, W, num_item_channels)
        occupant_bits = grid_1hot[..., num_item_channels:]  # (A, H, W, num_agents)

        me = occupant_bits[..., 0]  # (A, H, W)
        other = jnp.clip(jnp.sum(occupant_bits, axis=-1) - me, 0, 1)  # (A, H, W)

        angle_oh = jax.nn.one_hot(angles, 4, dtype=jnp.int8)  # (A, H, W, 4)
        angle_oh = jnp.where(angles[..., None] < 0,
                             0,
                             angle_oh)  # Mask invalid angles

        me = me[..., None]  # (A, H, W, 1)
        other = other[..., None]  # (A, H, W, 1)

        final_obs = jnp.concatenate([item_oh, me, other, angle_oh], axis=-1)  # (A, H, W, final_depth)
        return final_obs

    def _get_obs(self, state: State) -> jnp.ndarray:
        """
        A new observation function that:
          1) Pads the grid with walls.
          2) For each agent, slices out an (OBS_SIZE x OBS_SIZE) window
             from the padded grid, centered on (or near) the agent’s location
             according to orientation.
          3) Rotates the local patch based on agent orientation.
          4) Checks relative orientations of other agents => angles array
          5) One-hots the items
          6) combine_channels_simple => merges occupant bits + angle channels.

        Returns shape: (num_agents, OBS_SIZE, OBS_SIZE, final_depth)
        """
        # 1) Pad the grid with walls
        padded_items = jnp.pad(
            state.grid,
            pad_width=((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
            constant_values=Items.wall
        )
        padded_occupants = jnp.pad(
            state.occupant_grid,
            pad_width=((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
            constant_values=Items.wall
        )

        # 2) For each agent, compute the top-left corner of the obs slice
        def get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            # agent_loc is (row, col, orientation)
            # We want to shift by PADDING to accommodate the pad, then
            # move to the top-left corner s.t. the agent’s center is inside the slice
            # A simple approach: center the slice on the agent’s row/col:
            row, col, _ = agent_loc
            row = row + self.PADDING - (self.OBS_SIZE // 2)
            col = col + self.PADDING - (self.OBS_SIZE // 2)
            return jnp.array([row, col], dtype=jnp.int32)

        agent_positions = jax.vmap(get_obs_point)(state.agent_locs)  # (num_agents, 2)

        # 3) dynamic_slice the patch
        def slice_patch_items(pos2d):
            return jax.lax.dynamic_slice(
                padded_items,
                start_indices=(pos2d[0], pos2d[1]),
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

        def slice_patch_occ(pos2d):
            return jax.lax.dynamic_slice(
                padded_occupants,
                start_indices=(pos2d[0], pos2d[1]),
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

        local_items = jax.vmap(slice_patch_items)(agent_positions)  # (N, H, W)
        local_occs = jax.vmap(slice_patch_occ)(agent_positions)  # (N, H, W)

        # 4) rotate each patch according to orientation
        def rotate_for_agent(agent_loc, patch):
            return rotate_grid(agent_loc, patch)

        rotated_items = jax.vmap(rotate_for_agent)(state.agent_locs, local_items)
        rotated_occs = jax.vmap(rotate_for_agent)(state.agent_locs, local_occs)

        # Merge occupant code => occupant overrides item if occupant != 0
        # If occupant is 0 => keep item code, else occupant code
        merged_patches = jnp.where(rotated_occs > 0, rotated_occs, rotated_items)

        # shape=(N, H, W), containing either item code or occupant code

        # 5) compute angles => shape (num_agents, OBS_SIZE, OBS_SIZE)
        def angles_for_agent(agent_id, agent_locs, occ_grid):
            return check_relative_orientation(self, agent_id, agent_locs, occ_grid)

        angles = jax.vmap(
            angles_for_agent,
            in_axes=(0, None, 0)
        )(
            self._agents,
            state.agent_locs,
            rotated_occs
        )
        # angles has shape (num_agents, OBS_SIZE, OBS_SIZE)

        # 6) one-hot the items => shape (num_agents, OBS_SIZE, OBS_SIZE, (num_agents + len(Items)-1))
        # We do "rotated_patches - 1" so that 1 => index 0, i.e. Items.wall => channel 0
        # But be careful with empty=0 => -1 => out-of-range => we can just rely on "where < 0 => no item"
        # Usually we do: x = jnp.clip(x-1, 0, <some_upper>) if we want to handle negative
        # For simplicity, just do one_hot(rotated_patches - 1, depth=...).
        num_agents = self.num_agents
        # “(self.num_agents + (len(Items)-1))” => occupant bits + item bits
        patches_1hot = jax.nn.one_hot(
            merged_patches - 1,
            self.num_agents + (len(Items) - 1),
            dtype=jnp.int8
        )

        # 7) combine channels => shape: (num_agents, OBS_SIZE, OBS_SIZE, final_depth)
        def combine_for_agent(agent_id, grid_1hot, angle_map):
            return combine_channels_simple(grid_1hot, angle_map, agent_id - len(Items))

        final_obs = jax.vmap(
            combine_for_agent
        )(
            self._agents,
            patches_1hot,
            angles
        )

        return final_obs

    # ------------------ RENDERING ----------------------------------------
    def render(self, state: State):
        """
        Master function that returns a jnp array of shape (H, W, 3) in uint8,
        entirely computed in JAX on GPU/CPU.
        """
        # 1) Build highlight_mask, beam_mask in JAX
        highlight_mask_jax, beam_mask_jax = build_masks_jax(state, self.view_config, beam_range=3)

        # 2) Render the main grid
        main_img = render_jax(state, self.tile_size, ITEM_COLORS, self._agent_colors, self.base_tile_cache_jax,
                              self.final_tile_cache_jax, highlight_mask_jax, beam_mask_jax)

        # 3) Render the time bar (2*32 in height)
        bar = render_time_jax(
            inner_t=state.inner_t,
            outer_t=state.outer_t,
            max_inner=self.num_inner_steps,
            max_outer=self.num_outer_steps,
            width_px=main_img.shape[1]
        )

        # 4) Stack vertically
        final_img = jnp.concatenate([main_img, bar], axis=0)
        return final_img

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


def build_masks_jax(state: State,
                    view_cfg: ViewConfig,
                    beam_range: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns two jnp.bool_ arrays (highlight_mask, beam_mask), both shape=(R, C).
    - highlight_mask: True in each agent's partial-view area
    - beam_mask: True for cells in front of mining agents (up to beam_range or blocking)
    """
    rows, cols = state.grid.shape
    num_agents = state.agent_locs.shape[0]

    # Initialize empty masks
    highlight_mask = jnp.zeros((rows, cols), dtype=bool)
    beam_mask = jnp.zeros((rows, cols), dtype=bool)

    # -------------------------------------------------------------
    # 1) PARTIAL-VIEW HIGHLIGHT: each agent sees a (forward+backward+1) x (left+right+1) rectangle
    #    We gather local coords => transform them => scatter True into highlight_mask.
    # -------------------------------------------------------------
    local_rs = jnp.arange(view_cfg.forward + view_cfg.backward + 1)
    local_cs = jnp.arange(view_cfg.left + view_cfg.right + 1)

    # Use meshgrid to correctly create coordinate pairs
    local_rs, local_cs = jnp.meshgrid(
        jnp.arange(view_cfg.forward + view_cfg.backward + 1),
        jnp.arange(view_cfg.left + view_cfg.right + 1),
        indexing="ij"
    )
    # Stack to get (LH, 2) array of local row/col pairs
    local_coords_2d = jnp.stack([local_rs.ravel(), local_cs.ravel()], axis=-1)  # shape (LH, 2)

    # Next, map each agent + local_coord => global_coord
    def agent_local_to_global(agent_loc):
        """
        Return shape (LH,2) global coords for the entire local patch
        around one agent. We can do a vmap over local_coords_2d.
        """

        def per_localcoord(lc):
            return local_to_global(agent_loc[0], agent_loc[1], agent_loc[2],
                                   lc[0], lc[1], view_cfg)

        # Ensure the function returns a stacked JAX array, not a tuple
        return jnp.stack(jax.vmap(per_localcoord)(local_coords_2d), axis=0)  # shape (LH,2)

    # shape (N, LH, 2): each agent => the local patch => global coords
    all_patches = jax.vmap(agent_local_to_global)(state.agent_locs)

    # We'll flatten them => shape (N*LH, 2), then scatter True in highlight_mask
    all_patches_flat = all_patches.reshape(-1, 2)

    # Next we remove out-of-bounds points
    def in_bounds(rc):
        return ((rc[0] >= 0) & (rc[0] < rows) &
                (rc[1] >= 0) & (rc[1] < cols))

    # We'll define an array of booleans (N*LH,) telling which are in-bounds
    bounds_mask = jax.vmap(in_bounds)(all_patches_flat)
    coords_in = jnp.where(bounds_mask[:, None], all_patches_flat, -1)  # shape (N*LH,2)

    # We do a scatter update. For each (r,c) in coords_in, set highlight=True
    # 'index' shape must match. We'll do a loop or we can do a custom scatter.
    # One approach: "jax.lax.scatter" doesn't directly handle "set True" for multiple points
    # that might overlap, so we can do a "segment sum" approach.
    # A simpler universal approach: roll our own function that sets highlight_mask[rc] = True.

    def scatter_highlight(carry, rc):
        hmask = carry
        r, c = rc

        # if r<0 => skip
        # We can do jnp.where, or a conditional:
        def set_true(m):
            return m.at[r, c].set(True)

        def no_op(m):
            return m

        # 'r<0 => invalid'
        hmask = jax.lax.cond(r >= 0, set_true, no_op, hmask)
        return hmask, None

    highlight_mask, _ = jax.lax.scan(scatter_highlight, highlight_mask, coords_in)

    # -------------------------------------------------------------
    # 2) BEAM MASK: for each agent that Mined last step, trace up to beam_range
    # -------------------------------------------------------------
    # shape (N,) => did agent do 'mine' last step?  We'll just check state.actions_last_step if needed
    # But let's assume we want to highlight for all agents that actually used 'mine' last step:
    do_beam = (state.actions_last_step == Actions.mine)

    # We'll define a function that returns up to beam_range coords in front of agent i
    def compute_beam_cells(agent_i):
        """
        Return shape (beam_range, 2) of (row,col). Possibly -1 if out-of-bounds or we need to stop.
        """
        loc = state.agent_locs[agent_i]
        # orientation => STEP
        dr, dc, _ = STEP[loc[2]]
        r0, c0 = loc[0], loc[1]

        # We'll do a small loop that moves up to beam_range times
        # or stops if we see wall / iron_ore. We'll accumulate coords.

        def step_fn(carry, i):
            (rr, cc, stop) = carry
            # next cell
            rr_new = rr + dr
            cc_new = cc + dc

            # Check bounds
            inbound = (rr_new >= 0) & (rr_new < rows) & (cc_new >= 0) & (cc_new < cols)

            # If inbound, check item
            item_ = jnp.where(inbound, state.grid[rr_new, cc_new], Items.wall)
            # If item is wall or iron_ore => we must stop after this

            # When do we stop? If we hit a wall or ore
            blocked = jnp.any(jnp.array([
                item_ == Items.wall,
                item_ == Items.iron_ore,
                item_ == Items.gold_ore,
                item_ == Items.gold_partial
            ]))
            new_stop = stop | blocked

            # If we've already stopped, we keep the same coords => mark them as -1
            rr_final = jnp.where(stop, -1, rr_new)
            cc_final = jnp.where(stop, -1, cc_new)
            return (rr_final, cc_final, new_stop), jnp.array([rr_final, cc_final])

        carry_init = (r0, c0, False)
        # run up to beam_range steps
        (rf, cf, stopped), coords_emitted = jax.lax.scan(step_fn, carry_init, jnp.arange(beam_range))
        # coords_emitted => shape (beam_range, 2)
        return coords_emitted

    # We'll vmap that over all agents => shape (N, beam_range, 2)
    all_beams = jax.vmap(compute_beam_cells)(jnp.arange(num_agents))  # (N, beam_range, 2)

    # Keep only agents that do the beam => set coords to -1 if do_beam[i]==False
    # We'll broadcast do_beam => shape (N, beam_range, 1)
    do_beam_b = do_beam[:, None]
    # zero out or -1 the coords where do_beam is False
    all_beams = jnp.where(do_beam_b[..., None], all_beams, -1)

    # Flatten => shape (N*beam_range, 2)
    beam_flat = all_beams.reshape(-1, 2)

    def scatter_beam(carry, rc):
        bmask = carry
        r, c = rc

        def set_true(m):
            return m.at[r, c].set(True)

        def no_op(m):
            return m

        bmask = jax.lax.cond(r >= 0, set_true, no_op, bmask)
        return bmask, None

    beam_mask, _ = jax.lax.scan(scatter_beam, beam_mask, beam_flat)

    # Done! Return the two masks
    return highlight_mask, beam_mask