"""
Simplified Level-Based Foraging Environment in JAX
Based on https://github.com/semitable/lb-foraging

This is a multi-agent foraging environment where:
- Agents navigate a grid world to collect food items
- Each agent and food item has a level
- Food can only be collected when sum(adjacent_agent_levels) >= food_level
- This creates cooperation opportunities when food levels exceed individual agent levels
"""

from enum import IntEnum
from typing import Tuple, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces


class Actions(IntEnum):
    """Action space for agents"""
    NONE = 0      # No action
    NORTH = 1     # Move up
    SOUTH = 2     # Move down
    WEST = 3      # Move left
    EAST = 4      # Move right
    LOAD = 5      # Attempt to pick up food


class CellType(IntEnum):
    """Grid cell types"""
    EMPTY = 0
    WALL = 1
    FOOD = 2
    AGENT = 3


@struct.dataclass
class State:
    """Environment state"""
    agent_pos: chex.Array      # (num_agents, 2) - agent positions [x, y]
    agent_levels: chex.Array   # (num_agents,) - agent levels
    food_pos: chex.Array       # (num_food, 2) - food positions [x, y]
    food_levels: chex.Array    # (num_food,) - food levels
    food_active: chex.Array    # (num_food,) - whether food is active (not collected)
    grid: chex.Array           # (grid_size, grid_size) - grid state
    step_count: int            # current step count
    done: bool                 # episode done
    initial_food_total: float  # total sum of food levels at reset (for normalization)


# Movement deltas for each action
MOVEMENT = jnp.array([
    [0, 0],   # NONE
    [-1, 0],  # NORTH (up decreases row)
    [1, 0],   # SOUTH (down increases row)
    [0, -1],  # WEST (left decreases col)
    [0, 1],   # EAST (right increases col)
    [0, 0],   # LOAD
], dtype=jnp.int32)


class LBForaging(MultiAgentEnv):
    """
    Level-Based Foraging Environment

    Simplified version focusing on core cooperation mechanics:
    - Grid world with walls on borders
    - Fixed number of food items with random levels
    - Agents must cooperate to collect high-level food
    - Individual or shared rewards
    """

    def __init__(
        self,
        num_agents: int = 3,
        field_size: Tuple[int, int] = (10, 10),  # (rows, cols)
        num_food: int = 3,
        max_food_level: int = 3,
        min_food_level: int = 1,
        max_agent_level: int = 3,
        min_agent_level: int = 1,
        num_inner_steps: int = 500,
        shared_rewards: bool = False,
        force_coop: bool = False,
        sight: int = 3,
        cnn: bool = True,
        jit: bool = True,
        normalize_reward: bool = True,
        grid_observation: bool = True,
        observe_agent_levels: bool = True,
        penalty: float = 0.0,
    ):
        """
        Args:
            num_agents: Number of agents in environment
            field_size: Size of field (rows, cols)
            num_food: Number of food items
            max_food_level: Maximum food level (or array per food item)
            min_food_level: Minimum food level (or array per food item)
            max_agent_level: Maximum agent level (or array per agent)
            min_agent_level: Minimum agent level (or array per agent)
            num_inner_steps: Episode length
            shared_rewards: If True, all agents share rewards
            force_coop: If True, all food items have max level (requires cooperation)
            sight: Observation radius (agents see sight*2+1 x sight*2+1 grid)
            cnn: Whether to use CNN-compatible observations (alias for grid_observation)
            jit: Whether to JIT compile methods
            normalize_reward: Whether to normalize rewards by total food and agent levels
            grid_observation: If True, use grid observations; if False, use flat observations
            observe_agent_levels: If True, include agent levels in observations
            penalty: Penalty applied for failed LOAD attempts
        """
        super().__init__(num_agents=num_agents)

        # Use string agent IDs to match SocialJax convention
        self.agents = [str(i) for i in range(num_agents)]
        self.field_size = field_size
        self.rows = field_size[0]
        self.cols = field_size[1]
        self.num_food = num_food

        # Convert food levels to arrays
        if isinstance(min_food_level, int):
            self.min_food_level = jnp.array([min_food_level] * num_food)
        else:
            self.min_food_level = jnp.array(min_food_level)

        if isinstance(max_food_level, int):
            self.max_food_level = jnp.array([max_food_level] * num_food)
        else:
            self.max_food_level = jnp.array(max_food_level)

        # Convert agent levels to arrays
        if isinstance(min_agent_level, int):
            self.min_agent_level = jnp.array([min_agent_level] * num_agents)
        else:
            self.min_agent_level = jnp.array(min_agent_level)

        if isinstance(max_agent_level, int):
            self.max_agent_level = jnp.array([max_agent_level] * num_agents)
        else:
            self.max_agent_level = jnp.array(max_agent_level)

        self.num_inner_steps = num_inner_steps
        self.shared_rewards = shared_rewards
        self.force_coop = force_coop
        self.sight = sight
        self.cnn = cnn
        self.grid_observation = grid_observation if grid_observation is not None else cnn
        self.observe_agent_levels = observe_agent_levels
        self.normalize_reward = normalize_reward
        self.penalty = penalty
        self.obs_size = sight * 2 + 1

        # For compatibility
        self.grid_size = max(self.rows, self.cols)

        # Setup observation and action spaces
        max_food_lvl = int(jnp.max(self.max_food_level))
        max_agent_lvl = int(jnp.max(self.max_agent_level))

        for agent in self.agents:
            self.action_spaces[str(agent)] = spaces.Discrete(len(Actions))
            if self.grid_observation:
                # Grid observation: (obs_size, obs_size, channels)
                # Channels: [agents, foods, access]
                self.observation_spaces[str(agent)] = spaces.Box(
                    low=0, high=max(max_food_lvl, max_agent_lvl if observe_agent_levels else 1),
                    shape=(self.obs_size, self.obs_size, 3),
                    dtype=jnp.float32
                )
            else:
                # Flat observation: [food_positions_and_levels, agent_positions_and_levels]
                # Format: [(x, y, level) for each food] + [(x, y, level?) for each agent]
                player_obs_len = 3 if observe_agent_levels else 2
                obs_dim = num_food * 3 + num_agents * player_obs_len
                self.observation_spaces[str(agent)] = spaces.Box(
                    low=-1,
                    high=max(self.rows, self.cols, max_food_lvl, max_agent_lvl),
                    shape=(obs_dim,),
                    dtype=jnp.float32
                )

        # Define internal methods
        def _reset(key: chex.PRNGKey) -> State:
            """Reset environment to initial state"""
            key, key_agents, key_food = jax.random.split(key, 3)

            # Initialize empty grid
            grid = jnp.zeros(field_size, dtype=jnp.int32)

            # Spawn agents with random levels within min/max ranges
            def spawn_agent(i, carry):
                """Spawn agent i at random valid position"""
                pos_array, level_array, rng = carry
                rng, rng_pos, rng_level = jax.random.split(rng, 3)

                # Sample random position (avoiding borders)
                pos = jax.random.randint(
                    rng_pos,
                    shape=(2,),
                    minval=jnp.array([0, 0]),
                    maxval=jnp.array([self.rows, self.cols]),
                )

                # Sample agent level from min/max range
                level = jax.random.randint(
                    rng_level,
                    shape=(),
                    minval=self.min_agent_level[i],
                    maxval=self.max_agent_level[i] + 1,
                )

                pos_array = pos_array.at[i].set(pos)
                level_array = level_array.at[i].set(level)
                return (pos_array, level_array, rng)

            agent_pos = jnp.zeros((num_agents, 2), dtype=jnp.int32)
            agent_levels = jnp.zeros(num_agents, dtype=jnp.int32)
            agent_pos, agent_levels, key_agents = jax.lax.fori_loop(
                0, num_agents,
                spawn_agent,
                (agent_pos, agent_levels, key_agents)
            )

            # Spawn food with random levels, avoiding neighbors
            def spawn_food(i, carry):
                """Spawn food i at random valid position"""
                pos_array, level_array, rng = carry
                rng, rng_pos, rng_level = jax.random.split(rng, 3)

                # Sample random position (avoiding borders)
                pos = jax.random.randint(
                    rng_pos,
                    shape=(2,),
                    minval=jnp.array([1, 1]),
                    maxval=jnp.array([self.rows - 1, self.cols - 1]),
                )

                # Sample food level from min/max range
                # If force_coop, use max level; otherwise sample from range
                level = jax.lax.cond(
                    force_coop,
                    lambda _: self.max_food_level[i],
                    lambda _: jax.lax.cond(
                        self.min_food_level[i] == self.max_food_level[i],
                        lambda __: self.min_food_level[i],
                        lambda __: jax.random.randint(
                            rng_level,
                            shape=(),
                            minval=self.min_food_level[i],
                            maxval=self.max_food_level[i] + 1,
                        ),
                        None
                    ),
                    None
                )

                pos_array = pos_array.at[i].set(pos)
                level_array = level_array.at[i].set(level)
                return (pos_array, level_array, rng)

            food_pos = jnp.zeros((num_food, 2), dtype=jnp.int32)
            food_levels = jnp.zeros(num_food, dtype=jnp.int32)
            food_pos, food_levels, key_food = jax.lax.fori_loop(
                0, num_food,
                spawn_food,
                (food_pos, food_levels, key_food)
            )

            # All food starts active
            food_active = jnp.ones(num_food, dtype=jnp.bool_)

            # Calculate initial total food level for normalization
            initial_food_total = jnp.sum(food_levels)

            return State(
                agent_pos=agent_pos,
                agent_levels=agent_levels,
                food_pos=food_pos,
                food_levels=food_levels,
                food_active=food_active,
                grid=grid,
                step_count=0,
                done=False,
                initial_food_total=initial_food_total,
            )

        def _get_obs_single(state: State, agent_idx: int) -> chex.Array:
            """Get observation for a single agent"""
            agent_pos = state.agent_pos[agent_idx]

            if self.grid_observation:
                # Grid observation with 3 channels: agents, foods, access
                # Create observation grid padded with sight on all sides
                grid_shape_x = self.rows + 2 * sight
                grid_shape_y = self.cols + 2 * sight

                # Initialize layers
                agents_layer = jnp.zeros((grid_shape_x, grid_shape_y), dtype=jnp.float32)
                foods_layer = jnp.zeros((grid_shape_x, grid_shape_y), dtype=jnp.float32)
                access_layer = jnp.ones((grid_shape_x, grid_shape_y), dtype=jnp.float32)

                # Fill agents layer
                def add_agent_to_layer(i, layer):
                    pos = state.agent_pos[i]
                    level_val = state.agent_levels[i] if observe_agent_levels else 1.0
                    return layer.at[pos[0] + sight, pos[1] + sight].set(level_val)

                agents_layer = jax.lax.fori_loop(0, num_agents, add_agent_to_layer, agents_layer)

                # Fill foods layer
                def add_food_to_layer(i, layer):
                    is_active = state.food_active[i]
                    pos = state.food_pos[i]
                    return jax.lax.cond(
                        is_active,
                        lambda l: l.at[pos[0] + sight, pos[1] + sight].set(state.food_levels[i]),
                        lambda l: l,
                        layer
                    )

                foods_layer = jax.lax.fori_loop(0, num_food, add_food_to_layer, foods_layer)

                # Fill access layer
                # Out of bounds not accessible
                access_layer = access_layer.at[:sight, :].set(0.0)
                access_layer = access_layer.at[-sight:, :].set(0.0)
                access_layer = access_layer.at[:, :sight].set(0.0)
                access_layer = access_layer.at[:, -sight:].set(0.0)

                # Agent locations are not accessible
                def mark_agent_inaccessible(i, layer):
                    pos = state.agent_pos[i]
                    return layer.at[pos[0] + sight, pos[1] + sight].set(0.0)

                access_layer = jax.lax.fori_loop(0, num_agents, mark_agent_inaccessible, access_layer)

                # Food locations are not accessible
                def mark_food_inaccessible(i, layer):
                    is_active = state.food_active[i]
                    pos = state.food_pos[i]
                    return jax.lax.cond(
                        is_active,
                        lambda l: l.at[pos[0] + sight, pos[1] + sight].set(0.0),
                        lambda l: l,
                        layer
                    )

                access_layer = jax.lax.fori_loop(0, num_food, mark_food_inaccessible, access_layer)

                # Extract agent's view using dynamic_slice
                # lax.dynamic_slice requires static sizes
                view_size = 2 * sight + 1

                agents_view = jax.lax.dynamic_slice(
                    agents_layer,
                    (agent_pos[0], agent_pos[1]),
                    (view_size, view_size)
                )
                foods_view = jax.lax.dynamic_slice(
                    foods_layer,
                    (agent_pos[0], agent_pos[1]),
                    (view_size, view_size)
                )
                access_view = jax.lax.dynamic_slice(
                    access_layer,
                    (agent_pos[0], agent_pos[1]),
                    (view_size, view_size)
                )

                obs = jnp.stack([agents_view, foods_view, access_view], axis=-1)

            else:
                # Flat observation: positions and levels of foods and agents
                # Format: [(x, y, level) for each food] + [(x, y, level?) for each agent]
                player_obs_len = 3 if observe_agent_levels else 2
                obs = jnp.zeros(num_food * 3 + num_agents * player_obs_len, dtype=jnp.float32)

                # Add food observations
                def add_food_obs(i, o):
                    is_active = state.food_active[i]
                    # Transform to neighborhood coordinates
                    rel_y = state.food_pos[i, 0] - agent_pos[0] + sight
                    rel_x = state.food_pos[i, 1] - agent_pos[1] + sight

                    # Check if in sight
                    in_sight = is_active & (rel_y >= 0) & (rel_y <= 2 * sight) & (rel_x >= 0) & (rel_x <= 2 * sight)

                    o = o.at[3 * i].set(jnp.where(in_sight, rel_y, -1.0))
                    o = o.at[3 * i + 1].set(jnp.where(in_sight, rel_x, -1.0))
                    o = o.at[3 * i + 2].set(jnp.where(in_sight, state.food_levels[i], 0.0))
                    return o

                obs = jax.lax.fori_loop(0, num_food, add_food_obs, obs)

                # Add agent observations (self first, then others)
                def add_agent_obs(i, o):
                    # Reorder: self first, then others
                    actual_idx = jax.lax.cond(i == 0, lambda: agent_idx, lambda: jnp.where(i - 1 < agent_idx, i - 1, i))

                    # Transform to neighborhood coordinates
                    rel_y = state.agent_pos[actual_idx, 0] - agent_pos[0] + sight
                    rel_x = state.agent_pos[actual_idx, 1] - agent_pos[1] + sight

                    # Check if in sight
                    in_sight = (rel_y >= 0) & (rel_y <= 2 * sight) & (rel_x >= 0) & (rel_x <= 2 * sight)

                    base_idx = num_food * 3 + i * player_obs_len
                    o = o.at[base_idx].set(jnp.where(in_sight, rel_y, -1.0))
                    o = o.at[base_idx + 1].set(jnp.where(in_sight, rel_x, -1.0))

                    if observe_agent_levels:
                        o = o.at[base_idx + 2].set(jnp.where(in_sight, state.agent_levels[actual_idx], 0.0))

                    return o

                obs = jax.lax.fori_loop(0, num_agents, add_agent_obs, obs)

            return obs

        def _get_obs(state: State) -> Dict[str, chex.Array]:
            """Get observations for all agents"""
            obs_list = jax.vmap(lambda i: _get_obs_single(state, i))(
                jnp.arange(num_agents)
            )
            return {str(i): obs_list[i] for i in range(num_agents)}

        def _step_env(
            key: chex.PRNGKey,
            state: State,
            actions,  # Can be dict or list/array
            timestep: int = 0,
        ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
            """Execute environment step"""

            # Convert actions to array (handle both dict and list/array input)
            if isinstance(actions, dict):
                # Try string keys first (SocialJax convention), then integer keys
                try:
                    actions_array = jnp.array([actions[str(i)] for i in range(num_agents)])
                except (KeyError, TypeError):
                    actions_array = jnp.array([actions[i] for i in range(num_agents)])
            else:
                # actions is already a list or array
                actions_array = jnp.array(actions)

            # Movement phase
            # First, identify which agents want to move vs load
            load_mask = (actions_array == Actions.LOAD)
            move_mask = ~load_mask

            # Calculate new positions for movement actions
            movements = MOVEMENT[actions_array]
            new_positions = state.agent_pos + movements

            # Clip to valid grid positions (no walls at borders in original)
            new_positions = jnp.clip(new_positions, 0, jnp.array([self.rows - 1, self.cols - 1]))

            # Check collisions with food (movement blocked)
            def check_food_collision(pos):
                """Check if position collides with any food"""
                def check_single_food(i, blocked):
                    food_at_pos = state.food_active[i] & \
                                 (state.food_pos[i, 0] == pos[0]) & \
                                 (state.food_pos[i, 1] == pos[1])
                    return blocked | food_at_pos
                return jax.lax.fori_loop(0, num_food, check_single_food, False)

            food_collision_mask = jax.vmap(check_food_collision)(new_positions)

            # Check collisions with other agents
            # If multiple agents try to move to same position, all stay in place
            def check_collision(i):
                """Check if agent i collides with any other agent's new position"""
                pos = new_positions[i]
                # Check against all other agents
                collisions = jax.vmap(lambda j: jnp.where(
                    i != j,
                    (new_positions[j, 0] == pos[0]) & (new_positions[j, 1] == pos[1]),
                    False
                ))(jnp.arange(num_agents))
                return jnp.any(collisions)

            agent_collision_mask = jax.vmap(check_collision)(jnp.arange(num_agents))

            # Combine collision masks
            collision_mask = food_collision_mask | agent_collision_mask

            # If collision or not moving, stay at old position
            final_positions = jnp.where(
                (collision_mask | load_mask)[:, None],
                state.agent_pos,
                new_positions
            )

            # Food collection phase
            # For each food item, check if it can be collected
            def try_collect_food(i, carry):
                """Try to collect food item i"""
                food_active_here, rewards_carry = carry

                # Skip if already collected
                def collect_logic(_):
                    food_pos_i = state.food_pos[i]
                    food_level_i = state.food_levels[i]

                    # Find agents adjacent to this food (Manhattan distance = 1)
                    # Adjacent means exactly 1 step away in one direction (not diagonal)
                    distances = jnp.abs(final_positions - food_pos_i)
                    manhattan_dist = distances[:, 0] + distances[:, 1]
                    adjacent = (manhattan_dist == 1) & load_mask

                    # Sum levels of adjacent agents attempting to load
                    total_level = jnp.sum(jnp.where(adjacent, state.agent_levels, 0))

                    # Can collect if total level >= food level
                    can_collect = total_level >= food_level_i

                    # Distribute rewards to agents who participated
                    # Each agent gets: agent_level * food_level
                    # Then normalize if enabled
                    def compute_rewards(_):
                        # Raw reward: agent_level * food_level
                        raw_rewards = jnp.where(
                            adjacent,
                            state.agent_levels * food_level_i,
                            0.0
                        )
                        # Normalize rewards if enabled
                        if normalize_reward:
                            # Sum of adjacent agent levels
                            adj_level_sum = jnp.sum(jnp.where(adjacent, state.agent_levels, 0))
                            # Use initial food total (set at reset) for consistent normalization
                            normalization_factor = adj_level_sum * state.initial_food_total
                            normalized_rewards = raw_rewards / jnp.maximum(normalization_factor, 1.0)
                            return normalized_rewards
                        else:
                            return raw_rewards

                    def compute_penalties(_):
                        # Failed to collect: apply penalty to loading agents
                        return jnp.where(adjacent, -penalty, 0.0)

                    reward_per_agent = jax.lax.cond(
                        can_collect,
                        compute_rewards,
                        compute_penalties,
                        None
                    )

                    # Update food active status
                    new_food_active = jnp.where(can_collect, False, food_active_here[i])

                    return (
                        food_active_here.at[i].set(new_food_active),
                        rewards_carry + reward_per_agent
                    )

                return jax.lax.cond(
                    food_active_here[i],
                    collect_logic,
                    lambda _: carry,
                    None
                )

            # Process all food items
            initial_carry = (state.food_active, jnp.zeros(num_agents, dtype=jnp.float32))
            new_food_active, rewards_array = jax.lax.fori_loop(
                0, num_food,
                try_collect_food,
                initial_carry
            )

            # Convert rewards to dict (use string keys to match SocialJax convention)
            if shared_rewards:
                total_reward = jnp.sum(rewards_array)
                rewards = {str(i): total_reward for i in range(num_agents)}
            else:
                rewards = {str(i): float(rewards_array[i]) for i in range(num_agents)}

            # Update state
            new_step_count = state.step_count + 1
            # Episode ends when time limit reached OR all food collected
            all_food_collected = jnp.all(~new_food_active)
            done = (new_step_count >= num_inner_steps) | all_food_collected

            new_state = State(
                agent_pos=final_positions,
                agent_levels=state.agent_levels,
                food_pos=state.food_pos,
                food_levels=state.food_levels,
                food_active=new_food_active,
                grid=state.grid,
                step_count=new_step_count,
                done=done,
                initial_food_total=state.initial_food_total,
            )

            # Get new observations
            obs = _get_obs(new_state)

            # Create dones dict (use string keys to match SocialJax convention)
            dones = {str(i): done for i in range(num_agents)}
            dones["__all__"] = done

            # Info
            info = {
                "food_collected": jnp.sum(~new_food_active),
                "total_reward": jnp.sum(rewards_array),
            }

            return obs, new_state, rewards, dones, info

        def _reset_full(key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
            """Full reset returning obs and state"""
            state = _reset(key)
            obs = _get_obs(state)
            return obs, state

        def _get_avail_actions(state: State) -> Dict[str, chex.Array]:
            """Get available actions for each agent (all actions always available)"""
            avail = jnp.ones(len(Actions), dtype=jnp.int32)
            return {str(i): avail for i in range(num_agents)}

        # Assign methods
        if jit:
            self.reset = jax.jit(_reset_full)
            self.step_env = jax.jit(_step_env)
            self.get_obs = jax.jit(_get_obs)
            self.get_avail_actions = jax.jit(_get_avail_actions)
        else:
            self.reset = _reset_full
            self.step_env = _step_env
            self.get_obs = _get_obs
            self.get_avail_actions = _get_avail_actions

    @property
    def name(self) -> str:
        """Environment name"""
        return "LBForaging"

    def observation_space(self, agent_id=None):
        """Observation space"""
        max_food_lvl = int(jnp.max(self.max_food_level))
        max_agent_lvl = int(jnp.max(self.max_agent_level))

        if self.grid_observation:
            # Grid observation: 3 channels (agents, foods, access)
            shape = (self.obs_size, self.obs_size, 3)
            high = max(max_food_lvl, max_agent_lvl if self.observe_agent_levels else 1)
            if agent_id is not None:
                return spaces.Box(low=0, high=high, shape=shape, dtype=jnp.float32)
            return [
                spaces.Box(low=0, high=high, shape=shape, dtype=jnp.float32)
                for _ in range(self.num_agents)
            ]
        else:
            # Flat observation
            player_obs_len = 3 if self.observe_agent_levels else 2
            shape = (self.num_food * 3 + self.num_agents * player_obs_len,)
            high = max(self.rows, self.cols, max_food_lvl, max_agent_lvl)
            if agent_id is not None:
                return spaces.Box(low=-1, high=high, shape=shape, dtype=jnp.float32)
            return [
                spaces.Box(low=-1, high=high, shape=shape, dtype=jnp.float32)
                for _ in range(self.num_agents)
            ]

    def action_space(self, agent_id=None):
        """Action space"""
        if agent_id is not None:
            return spaces.Discrete(len(Actions))
        return [spaces.Discrete(len(Actions)) for _ in range(self.num_agents)]

    def render(self, state, cell_size=40, cumulative_rewards=None):
        """
        Render the environment state as an RGB image

        Args:
            state: Environment state
            cell_size: Size of each grid cell in pixels
            cumulative_rewards: Optional dict or list of cumulative rewards for each agent

        Returns:
            RGB numpy array
        """
        from socialjax.environments.lb_foraging.rendering import render_lb_foraging
        return render_lb_foraging(state, self.field_size, cell_size, cumulative_rewards)
