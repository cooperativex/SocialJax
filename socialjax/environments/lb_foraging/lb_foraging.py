import colorsys
from enum import IntEnum
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass

from socialjax.environments import spaces
from socialjax.environments.multi_agent_env import MultiAgentEnv

# ------------------------------------------------------------------------
# Level-Based Foraging (LBF) Environment
# Based on: https://github.com/semitable/lb-foraging
# Adapted for SocialJax with JAX acceleration
# ------------------------------------------------------------------------

# Simple 8x8 grid map with walls and spawn points
ASCII_MAP_LBF = """
WWWWWWWW
W      W
W  PP  W
W      W
W  PP  W
W      W
WWWWWWWW
""".strip('\n')

CHAR_TO_INT = {
    'W': 1,  # wall
    ' ': 0,  # floor
    'P': 2,  # spawn point
}


class Items(IntEnum):
    empty = 0
    wall = 1
    spawn_point = 2
    # Food levels are represented as values 10 + level (e.g., 10, 11, 12, 13 for levels 0-3)
    # This allows us to easily extract level via: (value - 10)


class Actions(IntEnum):
    """LBF uses 6 actions: NONE, 4 directions, and LOAD"""
    NONE = 0
    NORTH = 1  # Move up
    SOUTH = 2  # Move down
    WEST = 3   # Move left
    EAST = 4   # Move right
    LOAD = 5   # Attempt to collect adjacent food


# Movement deltas for each action (row, col)
MOVE_DELTAS = jnp.array(
    [
        [0, 0],   # NONE
        [-1, 0],  # NORTH (up)
        [1, 0],   # SOUTH (down)
        [0, -1],  # WEST (left)
        [0, 1],   # EAST (right)
        [0, 0],   # LOAD (no movement)
    ],
    dtype=jnp.int8,
)

# Adjacency offsets (N, S, W, E) for checking neighbors
ADJACENCY_OFFSETS = jnp.array([
    [-1, 0],  # North
    [1, 0],   # South
    [0, -1],  # West
    [0, 1],   # East
], dtype=jnp.int8)


@dataclass
class State:
    """LBF State - simplified from coop_mining (no orientation!)"""
    agent_positions: jnp.ndarray  # shape (num_agents, 2) => row, col only
    agent_levels: jnp.ndarray     # shape (num_agents,) => fixed level per agent
    food_grid: jnp.ndarray        # shape (rows, cols) => 0 = no food, 10+level = food
    inner_t: int
    outer_t: int


@dataclass
class ViewConfig:
    """Observation window configuration"""
    sight: int  # Sight radius (e.g., 8 for 8x8 view)


def ascii_map_to_grid(ascii_map: str, mapping: Dict[str, int]) -> jnp.ndarray:
    lines = ascii_map.strip().split('\n')
    array_of_ints = [[mapping.get(char, 0) for char in line] for line in lines]
    return jnp.array(array_of_ints, dtype=jnp.int32)


def generate_agent_colors(num_agents):
    """Generate distinct agent colors using HSV interpolation."""
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def check_collision_lbf(new_positions: jnp.ndarray) -> jnp.ndarray:
    """
    Check for agent-agent collisions. In LBF, if multiple agents try to move
    to the same cell, ALL of them fail to move.

    Returns: (num_agents,) bool array where True = agent has collision
    """
    # For each agent, check if any OTHER agent has the same position
    pos1 = new_positions[:, None, :]  # (N, 1, 2)
    pos2 = new_positions[None, :, :]  # (1, N, 2)
    same_pos = jnp.all(pos1 == pos2, axis=-1)  # (N, N)

    # Exclude self-comparison
    mask = ~jnp.eye(new_positions.shape[0], dtype=bool)
    collisions = same_pos & mask  # (N, N)

    # Agent has collision if ANY other agent is at same position
    has_collision = jnp.any(collisions, axis=1)  # (N,)
    return has_collision


# ------------------- MAIN ENV -------------------------------------------
class LevelBasedForaging(MultiAgentEnv):
    """
    Level-Based Foraging (LBF) Environment

    Key differences from Coop Mining:
    - No agent orientation (simple 4-directional movement)
    - Agents have levels, food has levels
    - Cooperation required when sum(agent_levels) >= food_level
    - Contribution-based rewards: reward_i = agent_i.level × food.level
    - Food doesn't respawn (finite episode)
    - Must be adjacent + execute LOAD action to collect
    """

    def __init__(
            self,
            num_inner_steps=100,
            num_outer_steps=1,
            num_agents=4,
            grid_size=(8, 8),
            num_food=4,
            max_food_level=3,
            max_player_level=2,
            sight=8,
            force_coop=False,
            normalize_reward=True,
            load_penalty=0.1,
            shared_rewards=False,
            # Keep these for compatibility with SocialJax algorithms
            inequity_aversion=False,
            inequity_aversion_target_agents=None,
            inequity_aversion_alpha=5,
            inequity_aversion_beta=0.05,
            svo=False,
            svo_target_agents=None,
            svo_w=0.5,
            svo_ideal_angle_degrees=45,
            interest=False,
            s_interest=0.5,
            s_interest_schedule=None,
            s_interest_change_every=30000000,
            cnn=True,
            jit=True,
    ):
        super().__init__(num_agents=num_agents)

        # LBF-specific parameters
        self.grid_size = grid_size
        self.num_food = num_food
        self.max_food_level = max_food_level
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self.normalize_reward = normalize_reward
        self.load_penalty = load_penalty

        # SocialJax compatibility parameters
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.shared_rewards = shared_rewards
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.interest = interest
        self.s_interest = s_interest
        if s_interest_schedule is not None:
            self.s_interest_schedule = jnp.array(s_interest_schedule)
        else:
            self.s_interest_schedule = None
        self.s_interest_change_every = s_interest_change_every

        self.cnn = cnn
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        self._agent_colors = jnp.array(generate_agent_colors(num_agents), dtype=jnp.uint8)

        self.view_config = ViewConfig(sight=sight)

        # Grid setup
        self._grid_base = ascii_map_to_grid(ASCII_MAP_LBF, CHAR_TO_INT)
        self.grid_shape = self._grid_base.shape
        self.GRID_SIZE_ROW = self.grid_shape[0]
        self.GRID_SIZE_COL = self.grid_shape[1]

        # Precompute spawn points
        self._spawn_pts = jnp.argwhere(self._grid_base == Items.spawn_point)

        # Action and observation spaces
        self.action_spaces = {
            i: spaces.Discrete(len(Actions)) for i in range(num_agents)
        }

        # Grid observation: (sight, sight, 3) - agents layer, food layer, accessibility layer
        obs_shape = (sight, sight, 3) if cnn else (sight * sight * 3,)
        self.observation_spaces = {
            i: spaces.Box(low=0, high=255, shape=obs_shape, dtype=jnp.uint8)
            for i in range(num_agents)
        }

        # JIT compilation
        if jit:
            self.step_env = jax.jit(self.step_env)
            self.reset = jax.jit(self.reset)
            self._get_obs = jax.jit(self._get_obs)

    @property
    def num_actions(self) -> int:
        return len(Actions)

    def close(self):
        pass

    @property
    def name(self):
        return "LevelBasedForaging"

    def state_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=jnp.uint8)

    def action_space(self, agent_id: Union[int, None] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """
        Returns grid observation: (sight, sight, 3) where:
        - Layer 0: Agent levels (0 if empty)
        - Layer 1: Food levels (0 if empty)
        - Layer 2: Accessibility (1 = walkable, 0 = wall)
        """
        obs_size = (self.sight, self.sight, 3) if self.cnn else (self.sight * self.sight * 3,)
        return spaces.Box(low=0, high=255, shape=obs_size, dtype=jnp.uint8), obs_size

    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
        state = self._reset_state(key)
        obs = self._get_obs(state)
        return obs, state

    def _reset_state(self, key: jnp.ndarray) -> State:
        """Initialize LBF environment state"""
        # 1. Spawn agents at random spawn points
        spawn_pts = self._spawn_pts
        key, subkey = jax.random.split(key)
        chosen = jax.random.choice(subkey, spawn_pts.shape[0],
                                   shape=(self.num_agents,), replace=False)
        agent_positions = spawn_pts[chosen]  # (num_agents, 2)

        # 2. Assign agent levels (uniform random from 1 to max_player_level)
        key, subkey = jax.random.split(key)
        agent_levels = jax.random.randint(
            subkey,
            shape=(self.num_agents,),
            minval=1,
            maxval=self.max_player_level + 1
        )

        # 3. Spawn food items
        food_grid = jnp.zeros(self.grid_shape, dtype=jnp.int32)

        # Pre-compute all valid positions at init time (stored in self._valid_food_positions)
        # For JIT compatibility, we need fixed-size arrays
        # We'll use a different approach: scatter food randomly on empty cells

        # Create a flat index array for all cells
        total_cells = self.GRID_SIZE_ROW * self.GRID_SIZE_COL

        # For each food item, we'll try random positions until we find valid ones
        # Using a scan to avoid dynamic shapes
        def place_one_food(carry, food_idx):
            food_grid, key = carry

            # Generate random position
            key, subkey = jax.random.split(key)
            row = jax.random.randint(subkey, (), 0, self.GRID_SIZE_ROW)
            key, subkey = jax.random.split(key)
            col = jax.random.randint(subkey, (), 0, self.GRID_SIZE_COL)

            # Check if valid (not wall, not spawn point, not already occupied)
            is_valid = (
                (self._grid_base[row, col] == Items.empty) &
                (food_grid[row, col] == 0)
            )

            # Assign random level
            key, subkey = jax.random.split(key)
            if self.force_coop:
                level = self.max_food_level
            else:
                level = jax.random.randint(subkey, (), 1, self.max_food_level + 1)

            # Place food only if valid
            new_food_grid = jnp.where(
                is_valid,
                food_grid.at[row, col].set(level + 10),
                food_grid
            )

            return (new_food_grid, key), None

        # Place all food items
        (food_grid, key), _ = jax.lax.scan(
            place_one_food,
            (food_grid, key),
            jnp.arange(self.num_food)
        )

        return State(
            agent_positions=agent_positions,
            agent_levels=agent_levels,
            food_grid=food_grid,
            inner_t=0,
            outer_t=0,
        )

    def get_current_s_interest(self, timestep):
        """Calculate current s_interest based on timestep and schedule."""
        if self.s_interest_schedule is None:
            return self.s_interest
        phase = timestep // self.s_interest_change_every
        phase_idx = phase % self.s_interest_schedule.shape[0]
        return self.s_interest_schedule[phase_idx]

    def step_env(self, key: jnp.ndarray, state: State, actions: jnp.ndarray, timestep: int = 0):
        """
        LBF step function

        1. Process movement (4-directional, no orientation)
        2. Handle collisions (agent-agent, walls, food blocking)
        3. Process LOAD actions (adjacency + level-sum check)
        4. Distribute rewards (contribution-based)
        5. Check episode termination (all food collected or max steps)
        """
        actions = jnp.array(actions, dtype=jnp.int32).squeeze()

        # 1. Calculate intended new positions
        old_positions = state.agent_positions  # (N, 2)
        move_offsets = MOVE_DELTAS[actions]     # (N, 2)
        new_positions = old_positions + move_offsets

        # 2. Clip to grid bounds
        new_positions = jnp.clip(
            new_positions,
            a_min=jnp.array([0, 0]),
            a_max=jnp.array([self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1])
        )

        # 3. Check wall collisions
        wall_mask = (self._grid_base[new_positions[:, 0], new_positions[:, 1]] == Items.wall)
        new_positions = jnp.where(wall_mask[:, None], old_positions, new_positions)

        # 4. Check food blocking (can't move onto food)
        food_mask = (state.food_grid[new_positions[:, 0], new_positions[:, 1]] > 0)
        new_positions = jnp.where(food_mask[:, None], old_positions, new_positions)

        # 5. Check agent-agent collisions
        collision_mask = check_collision_lbf(new_positions)
        final_positions = jnp.where(collision_mask[:, None], old_positions, new_positions)

        # 6. Process LOAD actions (food collection)
        rewards, new_food_grid = self._process_load_actions(
            state, final_positions, actions, key
        )

        # 7. Apply reward shaping if configured
        if self.shared_rewards:
            total_reward = jnp.sum(rewards)
            final_rewards = jnp.full((self.num_agents,), total_reward)
            shaped_rewards = final_rewards  # Set shaped_rewards
            info = {
                "original_rewards": final_rewards.squeeze(),
                "shaped_rewards": final_rewards.squeeze(),
            }
        elif self.inequity_aversion:
            final_rewards = rewards * self.num_agents
            shaped_rewards, _, _ = self.get_inequity_aversion_rewards(
                final_rewards, self.inequity_aversion_target_agents,
                self.inequity_aversion_alpha, self.inequity_aversion_beta
            )
            info = {
                "original_rewards": final_rewards.squeeze(),
                "shaped_rewards": shaped_rewards.squeeze(),
            }
        elif self.svo:
            final_rewards = rewards * self.num_agents
            shaped_rewards, theta = self.get_svo_rewards(
                final_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents
            )
            info = {
                "original_rewards": final_rewards.squeeze(),
                "svo_theta": theta.squeeze(),
                "shaped_rewards": shaped_rewards.squeeze(),
            }
        elif self.interest:
            final_rewards = rewards * self.num_agents
            current_s_interest = self.get_current_s_interest(timestep)
            total_reward = jnp.sum(final_rewards)
            others_reward = total_reward - final_rewards
            shaped_rewards = (current_s_interest * final_rewards +
                            (1 - current_s_interest) / (self.num_agents - 1) * others_reward)
            info = {
                "original_rewards": final_rewards.squeeze(),
                "shaped_rewards": shaped_rewards.squeeze(),
                "s_interest": current_s_interest,
            }
        else:
            final_rewards = rewards
            shaped_rewards = rewards
            info = {}

        # 8. Build next state
        new_state = State(
            agent_positions=final_positions,
            agent_levels=state.agent_levels,
            food_grid=new_food_grid,
            inner_t=state.inner_t + 1,
            outer_t=state.outer_t,
        )

        # 9. Check episode termination
        # Episode ends when: (1) all food collected OR (2) max steps reached
        all_food_collected = (jnp.sum(new_food_grid > 0) == 0)
        max_steps_reached = (new_state.inner_t >= self.num_inner_steps)
        reset_inner = all_food_collected | max_steps_reached

        new_outer = jnp.where(reset_inner, new_state.outer_t + 1, new_state.outer_t)
        reset_outer = (new_outer >= self.num_outer_steps)

        done_dict = {f"{i}": reset_outer for i in range(self.num_agents)}
        done_dict["__all__"] = reset_outer

        # 10. Get observations
        obs = self._get_obs(new_state)

        return obs, new_state, shaped_rewards, done_dict, info

    def _process_load_actions(
        self,
        state: State,
        positions: jnp.ndarray,
        actions: jnp.ndarray,
        key: jnp.ndarray  # kept for interface compatibility, not used in LBF
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process LOAD actions for all agents.

        For each food item:
        1. Find all adjacent agents executing LOAD
        2. Check if sum(agent_levels) >= food_level
        3. If success: distribute rewards and remove food
        4. If fail: apply penalty

        Returns:
            rewards: (num_agents,) array of rewards
            new_food_grid: updated food grid
        """
        # Identify agents executing LOAD action
        is_loading = (actions == Actions.LOAD)  # (num_agents,)

        # Initialize rewards and food grid
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        new_food_grid = state.food_grid.copy()

        # Instead of using argwhere (which needs static size),
        # we iterate over ALL grid cells and check if there's food
        def process_one_cell(carry, cell_idx):
            food_grid, agent_rewards = carry

            # Convert flat index to (row, col)
            row = cell_idx // self.GRID_SIZE_COL
            col = cell_idx % self.GRID_SIZE_COL

            # Check if this cell has food
            has_food = food_grid[row, col] > 0

            # Extract food level (or 0 if no food)
            food_level = jnp.where(has_food, food_grid[row, col] - 10, 0)

            # Find adjacent agents
            food_pos = jnp.array([row, col])
            agent_to_food = positions - food_pos[None, :]  # (num_agents, 2)
            distances = jnp.abs(agent_to_food[:, 0]) + jnp.abs(agent_to_food[:, 1])
            is_adjacent = (distances == 1)  # (num_agents,)

            # Filter for agents that are BOTH adjacent AND loading
            participating = is_adjacent & is_loading & has_food  # (num_agents,)

            # Sum levels of participating agents
            participating_levels = jnp.where(
                participating,
                state.agent_levels,
                0
            )
            total_level = jnp.sum(participating_levels)

            # Check if collection succeeds
            num_participants = jnp.sum(participating)
            success = (total_level >= food_level) & (num_participants > 0) & has_food

            # Calculate rewards for participating agents
            # reward_i = agent_i.level × food_level (if success)
            individual_rewards = jnp.where(
                participating,
                state.agent_levels.astype(jnp.float32) * food_level.astype(jnp.float32),
                0.0
            )

            # Apply normalization if configured
            if self.normalize_reward:
                # Normalize by total participating levels
                normalization = jnp.where(
                    success,
                    total_level.astype(jnp.float32),
                    1.0
                )
                individual_rewards = individual_rewards / normalization

            # Apply rewards only if success
            individual_rewards = jnp.where(success, individual_rewards, 0.0)

            # Apply penalty if failed
            penalty_rewards = jnp.where(
                participating & (~success),
                -self.load_penalty,
                0.0
            )

            # Combine success rewards and penalties
            total_individual_rewards = individual_rewards + penalty_rewards

            # Update food grid: remove food if collected
            new_food_val = jnp.where(success, 0, food_grid[row, col])
            new_grid = food_grid.at[row, col].set(new_food_val)

            # Accumulate rewards
            new_rewards = agent_rewards + total_individual_rewards

            return (new_grid, new_rewards), None

        # Process all grid cells
        total_cells = self.GRID_SIZE_ROW * self.GRID_SIZE_COL
        (final_food_grid, final_rewards), _ = jax.lax.scan(
            process_one_cell,
            (new_food_grid, rewards),
            jnp.arange(total_cells)
        )

        return final_rewards, final_food_grid

    def _get_obs(self, state: State) -> jnp.ndarray:
        """
        Generate grid observations for all agents.

        Returns: (num_agents, sight, sight, 3) where:
        - Layer 0: Agent levels at each position (0 if no agent)
        - Layer 1: Food levels at each position (0 if no food)
        - Layer 2: Accessibility (1 = walkable, 0 = wall)
        """
        # Pad grids with walls for boundary handling
        pad_width = self.sight // 2

        # Create base accessibility layer
        accessibility = (self._grid_base != Items.wall).astype(jnp.uint8)
        padded_accessibility = jnp.pad(
            accessibility,
            pad_width=((pad_width, pad_width), (pad_width, pad_width)),
            constant_values=0
        )

        # Create agent level grid
        agent_grid = jnp.zeros(self.grid_shape, dtype=jnp.uint8)
        agent_grid = agent_grid.at[
            state.agent_positions[:, 0],
            state.agent_positions[:, 1]
        ].set(state.agent_levels)
        padded_agents = jnp.pad(
            agent_grid,
            pad_width=((pad_width, pad_width), (pad_width, pad_width)),
            constant_values=0
        )

        # Create food level grid (extract level from encoding: value - 10)
        food_level_grid = jnp.where(
            state.food_grid > 0,
            state.food_grid - 10,
            0
        ).astype(jnp.uint8)
        padded_food = jnp.pad(
            food_level_grid,
            pad_width=((pad_width, pad_width), (pad_width, pad_width)),
            constant_values=0
        )

        # Extract sight×sight window for each agent
        def get_agent_obs(agent_pos):
            # Center window on agent position
            row, col = agent_pos
            row_padded = row + pad_width
            col_padded = col + pad_width

            # Slice sight×sight window
            start_row = row_padded - self.sight // 2
            start_col = col_padded - self.sight // 2

            agent_layer = jax.lax.dynamic_slice(
                padded_agents,
                (start_row, start_col),
                (self.sight, self.sight)
            )
            food_layer = jax.lax.dynamic_slice(
                padded_food,
                (start_row, start_col),
                (self.sight, self.sight)
            )
            access_layer = jax.lax.dynamic_slice(
                padded_accessibility,
                (start_row, start_col),
                (self.sight, self.sight)
            )

            # Stack into (sight, sight, 3)
            obs = jnp.stack([agent_layer, food_layer, access_layer], axis=-1)
            return obs

        # Generate observations for all agents
        observations = jax.vmap(get_agent_obs)(state.agent_positions)

        # If not using CNN, flatten
        if not self.cnn:
            observations = observations.reshape(self.num_agents, -1)

        return observations

    # ------------------- Reward Shaping Methods (from Coop Mining) -------------------

    def get_inequity_aversion_rewards(self, rewards, target_agents=None, alpha=5, beta=0.05):
        """Inequity aversion reward shaping"""
        rewards = rewards.reshape(-1, 1)
        r_i = rewards
        r_j = jnp.transpose(rewards)

        disadvantageous = jnp.maximum(r_j - r_i, 0)
        advantageous = jnp.maximum(r_i - r_j, 0)

        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask

        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        subjective_rewards = rewards - inequity_penalty
        subjective_rewards = jnp.where(jnp.all(rewards == 0), -(alpha + beta) * n_others, subjective_rewards)

        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, subjective_rewards, rewards).squeeze(), \
                   jnp.sum(disadvantageous, axis=1, keepdims=True).squeeze(), \
                   jnp.sum(advantageous, axis=1, keepdims=True).squeeze()
        else:
            return subjective_rewards.squeeze(), \
                   jnp.sum(disadvantageous, axis=1, keepdims=True).squeeze(), \
                   jnp.sum(advantageous, axis=1, keepdims=True).squeeze()

    def get_svo_rewards(self, rewards, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """Social Value Orientation reward shaping"""
        rewards = rewards.reshape(-1, 1)
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0

        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, rewards)
        mean_others = others_rewards / (self.num_agents - 1)

        r_i = rewards
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)

        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, rewards).squeeze(), theta.squeeze()
        else:
            return svo_utility.squeeze(), theta.squeeze()

    # ------------------- RENDERING ----------------------------------------
    def render(self, state: State, cell_size: int = 50, cumulative_rewards: dict = None) -> onp.ndarray:
        """
        Render the LBF environment state as an RGB image.

        Args:
            state: Current environment state
            cell_size: Size of each grid cell in pixels
            cumulative_rewards: Dict of cumulative rewards per agent (for display)

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        from socialjax.environments.lb_foraging.rendering import render_lb_foraging

        # Convert JAX state to format expected by rendering function
        # Create a simple namespace object to hold state attributes
        class RenderState:
            def __init__(self, env_ref, state_obj):
                # Extract food positions and levels from food_grid
                food_positions = []
                food_levels_list = []
                food_active_list = []

                food_grid_np = onp.array(state_obj.food_grid)
                for r in range(food_grid_np.shape[0]):
                    for c in range(food_grid_np.shape[1]):
                        if food_grid_np[r, c] > 0:
                            food_positions.append([r, c])
                            food_levels_list.append(food_grid_np[r, c] - 10)
                            food_active_list.append(True)

                # Pad to num_food size (in case some food was collected)
                while len(food_positions) < env_ref.num_food:
                    food_positions.append([0, 0])
                    food_levels_list.append(1)
                    food_active_list.append(False)

                self.food_pos = onp.array(food_positions[:env_ref.num_food])
                self.food_levels = onp.array(food_levels_list[:env_ref.num_food])
                self.food_active = onp.array(food_active_list[:env_ref.num_food])
                self.agent_pos = onp.array(state_obj.agent_positions)
                self.agent_levels = onp.array(state_obj.agent_levels)
                self.step_count = state_obj.inner_t

        render_state = RenderState(self, state)

        return render_lb_foraging(
            render_state,
            field_size=self.grid_shape,
            cell_size=cell_size,
            cumulative_rewards=cumulative_rewards
        )
