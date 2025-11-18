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
        grid_size: int = 10,
        num_food: int = 3,
        max_food_level: int = 3,
        max_agent_level: int = 3,  # NEW: Maximum agent level
        num_inner_steps: int = 500,
        shared_rewards: bool = False,
        force_coop: bool = False,  # NEW: Force cooperation mode
        sight: int = 3,  # observation radius
        cnn: bool = True,
        jit: bool = True,
    ):
        """
        Args:
            num_agents: Number of agents in environment
            grid_size: Size of square grid
            num_food: Number of food items
            max_food_level: Maximum food level
            max_agent_level: Maximum agent level (agents will have levels 1 to max_agent_level)
            num_inner_steps: Episode length
            shared_rewards: If True, all agents share rewards
            force_coop: If True, all food items have max level (requires cooperation)
            sight: Observation radius (agents see sight*2+1 x sight*2+1 grid)
            cnn: Whether to use CNN-compatible observations
            jit: Whether to JIT compile methods
        """
        super().__init__(num_agents=num_agents)

        # Use string agent IDs to match SocialJax convention
        self.agents = [str(i) for i in range(num_agents)]
        self.grid_size = grid_size
        self.num_food = num_food
        self.max_food_level = max_food_level
        self.max_agent_level = max_agent_level
        self.num_inner_steps = num_inner_steps
        self.shared_rewards = shared_rewards
        self.force_coop = force_coop
        self.sight = sight
        self.cnn = cnn
        self.obs_size = sight * 2 + 1

        # Setup observation and action spaces
        for agent in self.agents:
            self.action_spaces[str(agent)] = spaces.Discrete(len(Actions))
            if cnn:
                # CNN observation: (obs_size, obs_size, channels)
                # Channels: [empty, wall, food_level_1, ..., food_level_max,
                #            agent_level_1, ..., agent_level_max, self]
                num_channels = 2 + max_food_level + max_agent_level + 1
                self.observation_spaces[str(agent)] = spaces.Box(
                    low=0, high=1, shape=(self.obs_size, self.obs_size, num_channels),
                    dtype=jnp.float32
                )
            else:
                # Flat observation
                obs_dim = self.obs_size * self.obs_size * (2 + max_food_level + max_agent_level + 1)
                self.observation_spaces[str(agent)] = spaces.Box(
                    low=0, high=1, shape=(obs_dim,), dtype=jnp.float32
                )

        # Define internal methods
        def _reset(key: chex.PRNGKey) -> State:
            """Reset environment to initial state"""
            key, key_agents, key_agent_levels, key_food, key_food_levels = jax.random.split(key, 5)

            # Initialize empty grid (walls will be added on borders)
            grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)

            # Sample random agent positions (avoiding borders)
            agent_pos = jax.random.randint(
                key_agents,
                shape=(num_agents, 2),
                minval=1,
                maxval=grid_size - 1,
            )

            # Sample random agent levels (1 to max_agent_level)
            agent_levels = jax.random.randint(
                key_agent_levels,
                shape=(num_agents,),
                minval=1,
                maxval=max_agent_level + 1,
            )

            # Sample random food positions (avoiding borders and agent positions)
            food_pos = jax.random.randint(
                key_food,
                shape=(num_food, 2),
                minval=1,
                maxval=grid_size - 1,
            )

            # Sample random food levels (1 to max_food_level)
            # If force_coop, all food has max level
            food_levels = jax.lax.cond(
                force_coop,
                lambda _: jnp.full(num_food, max_food_level, dtype=jnp.int32),
                lambda _: jax.random.randint(
                    key_food_levels,
                    shape=(num_food,),
                    minval=1,
                    maxval=max_food_level + 1,
                ),
                None
            )

            # All food starts active
            food_active = jnp.ones(num_food, dtype=jnp.bool_)

            return State(
                agent_pos=agent_pos,
                agent_levels=agent_levels,  # Now using random levels
                food_pos=food_pos,
                food_levels=food_levels,
                food_active=food_active,
                grid=grid,
                step_count=0,
                done=False,
            )

        def _get_obs_single(state: State, agent_idx: int) -> chex.Array:
            """Get observation for a single agent"""
            agent_pos = state.agent_pos[agent_idx]

            # Create local observation grid centered on agent
            # Channels: empty, wall, food_levels (max_food_level), agent_levels (max_agent_level), self
            num_obs_channels = 2 + max_food_level + max_agent_level + 1
            obs = jnp.zeros((self.obs_size, self.obs_size, num_obs_channels), dtype=jnp.float32)

            # Get bounds of observation window
            x_min = agent_pos[0] - sight
            x_max = agent_pos[0] + sight + 1
            y_min = agent_pos[1] - sight
            y_max = agent_pos[1] + sight + 1

            # Fill observation grid
            for dx in range(self.obs_size):
                for dy in range(self.obs_size):
                    world_x = x_min + dx
                    world_y = y_min + dy

                    # Check if out of bounds or wall
                    is_wall = (world_x < 0) | (world_x >= grid_size) | \
                             (world_y < 0) | (world_y >= grid_size)

                    # Channel 0: empty, Channel 1: wall
                    obs = obs.at[dx, dy, 0].set(jnp.where(is_wall, 0.0, 1.0))
                    obs = obs.at[dx, dy, 1].set(jnp.where(is_wall, 1.0, 0.0))

                    # Check for food at this position
                    def check_food(i, obs_food):
                        """Check if food i is at this position"""
                        food_here = state.food_active[i] & \
                                   (state.food_pos[i, 0] == world_x) & \
                                   (state.food_pos[i, 1] == world_y)
                        food_level = state.food_levels[i]
                        # Set food level channel (channels 2 to 2+max_food_level-1)
                        obs_food = jax.lax.cond(
                            food_here,
                            lambda o: o.at[dx, dy, 1 + food_level].set(1.0),
                            lambda o: o,
                            obs_food
                        )
                        return obs_food

                    obs = jax.lax.fori_loop(0, num_food,
                                           lambda i, o: check_food(i, o), obs)

                    # Check for other agents (use different channels for different levels)
                    def check_agent(i, obs_agent):
                        """Check if agent i is at this position"""
                        agent_here = (i != agent_idx) & \
                                    (state.agent_pos[i, 0] == world_x) & \
                                    (state.agent_pos[i, 1] == world_y)
                        agent_level = state.agent_levels[i]
                        # Set agent level channel (channels after food channels)
                        # Channel index: 2 + max_food_level + (agent_level - 1)
                        obs_agent = jax.lax.cond(
                            agent_here,
                            lambda o: o.at[dx, dy, 1 + max_food_level + agent_level].set(1.0),
                            lambda o: o,
                            obs_agent
                        )
                        return obs_agent

                    obs = jax.lax.fori_loop(0, num_agents,
                                           lambda i, o: check_agent(i, o), obs)

                    # Check if self at this position (last channel)
                    self_here = (agent_pos[0] == world_x) & (agent_pos[1] == world_y)
                    obs = obs.at[dx, dy, -1].set(jnp.where(self_here, 1.0, 0.0))

            if not cnn:
                obs = obs.reshape(-1)

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

            # Apply movement actions
            movements = MOVEMENT[actions_array]
            new_positions = state.agent_pos + movements

            # Clip to valid grid positions (accounting for walls at borders)
            new_positions = jnp.clip(new_positions, 1, grid_size - 2)

            # Handle collisions: if multiple agents try to move to same position, all stay in place
            def check_collision(i):
                """Check if agent i collides with any other agent"""
                pos = new_positions[i]
                # Check against all other agents
                collisions = jax.vmap(lambda j: jnp.where(
                    i != j,
                    (new_positions[j, 0] == pos[0]) & (new_positions[j, 1] == pos[1]),
                    False
                ))(jnp.arange(num_agents))
                return jnp.any(collisions)

            collision_mask = jax.vmap(check_collision)(jnp.arange(num_agents))

            # If collision, stay at old position
            final_positions = jnp.where(
                collision_mask[:, None],
                state.agent_pos,
                new_positions
            )

            # Handle LOAD actions
            load_mask = (actions_array == Actions.LOAD)

            # For each food item, check if it can be collected
            def try_collect_food(i, carry):
                """Try to collect food item i"""
                food_pos_here, food_active_here, rewards_carry = carry

                # Skip if already collected
                def collect_logic(_):
                    # Find all agents adjacent to this food
                    distances = jnp.abs(final_positions - state.food_pos[i])
                    adjacent = (distances[:, 0] <= 1) & (distances[:, 1] <= 1) & load_mask

                    # Sum levels of adjacent agents attempting to load
                    total_level = jnp.sum(jnp.where(adjacent, state.agent_levels, 0))

                    # Can collect if total level >= food level
                    can_collect = total_level >= state.food_levels[i]

                    # Distribute rewards to agents who participated
                    reward_per_agent = jnp.where(
                        can_collect & adjacent,
                        state.food_levels[i] * state.agent_levels / jnp.maximum(total_level, 1),
                        0.0
                    )

                    # Update food active status
                    new_food_active = jnp.where(can_collect, False, food_active_here[i])

                    return (
                        food_pos_here,
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
            initial_carry = (state.food_pos, state.food_active, jnp.zeros(num_agents))
            _, new_food_active, rewards_array = jax.lax.fori_loop(
                0, num_food,
                try_collect_food,
                initial_carry
            )

            # Convert rewards to dict (use string keys to match SocialJax convention)
            if shared_rewards:
                total_reward = jnp.sum(rewards_array)
                rewards = {str(i): total_reward for i in range(num_agents)}
            else:
                rewards = {str(i): rewards_array[i] for i in range(num_agents)}

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
        if self.cnn:
            # Channels: empty, wall, food_levels (max_food_level), agent_levels (max_agent_level), self
            num_channels = 2 + self.max_food_level + self.max_agent_level + 1
            shape = (self.obs_size, self.obs_size, num_channels)
        else:
            shape = (self.obs_size * self.obs_size * (2 + self.max_food_level + self.max_agent_level + 1),)

        if agent_id is not None:
            return spaces.Box(low=0, high=1, shape=shape, dtype=jnp.float32)
        return [
            spaces.Box(low=0, high=1, shape=shape, dtype=jnp.float32)
            for _ in range(self.num_agents)
        ]

    def action_space(self, agent_id=None):
        """Action space"""
        if agent_id is not None:
            return spaces.Discrete(len(Actions))
        return [spaces.Discrete(len(Actions)) for _ in range(self.num_agents)]

    def render(self, state, cell_size=40):
        """
        Render the environment state as an RGB image

        Args:
            state: Environment state
            cell_size: Size of each grid cell in pixels

        Returns:
            RGB numpy array
        """
        from socialjax.environments.lb_foraging.rendering import render_lb_foraging
        return render_lb_foraging(state, self.grid_size, cell_size)
