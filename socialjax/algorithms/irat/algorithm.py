"""IRAT Algorithm — Individual Reward Assisted Multi-Agent RL.

Paper: "Individual Reward Assisted Multi-Agent Reinforcement Learning"
       (ICML 2022) https://arxiv.org/abs/2202.03612

IRAT maintains two separate policies:
- Individual policy (ind_actor + ind_critic): optimized for individual rewards,
  acts in the environment (drives exploration with dense rewards)
- Team policy (team_actor + team_critic): optimized for mean team reward,
  learns from individual's trajectories via importance sampling

Per the paper, the INDIVIDUAL actor acts in the environment. The team actor
evaluates the same actions via importance sampling. The two policies are
coupled through:
1. IRAT cooperation clip (surr3) on individual actor loss
2. KL(pi_team || pi_ind) regularizer on individual actor (alpha, increasing)
3. KL(pi_ind || pi_team) regularizer on team actor (beta, decreasing)

Note: Full training is done via scripts/train_irat.py which uses JAX scan
for the ~1B step training runs. This module registers IRAT in the algorithm
registry for config management and future evaluation use.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from socialjax.core.base_algorithm import BaseAlgorithm, AlgorithmState
from socialjax.algorithms.registry import register_algorithm
from socialjax.algorithms.irat.config import IRAT_DEFAULT_CONFIG, get_irat_config
from socialjax.algorithms.irat.network import IRATActorCNN, IRATCriticCNN


class IRATAlgorithmState(struct.PyTreeNode):
    """State container for IRAT algorithm with 4 networks.

    Attributes:
        ind_actor_state: Individual actor TrainState
        ind_critic_state: Individual critic TrainState
        team_actor_state: Team actor TrainState (acts in environment)
        team_critic_state: Team critic TrainState (uses world state)
        rng: Random key
        timestep: Current training timestep
        update_step: Number of parameter updates performed
    """
    ind_actor_state: TrainState
    ind_critic_state: TrainState
    team_actor_state: TrainState
    team_critic_state: TrainState
    rng: jax.random.PRNGKey
    timestep: int = 0
    update_step: int = 0


@register_algorithm("irat")
class IRATAlgorithm(BaseAlgorithm):
    """IRAT: Individual Reward Assisted Multi-Agent Reinforcement Learning.

    Registers IRAT in the algorithm registry. Full training is performed
    by scripts/train_irat.py using a JAX-scan-based training loop for
    performance at 1B steps.

    For inference/evaluation, use compute_action() which uses the team actor.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict[str, Any]] = None,
        num_agents: int = 1,
    ):
        merged_config = get_irat_config(config)
        self.num_agents = num_agents
        super().__init__(observation_space, action_space, merged_config)

    def _build_network(self) -> Tuple[Any, Any, Any, Any]:
        """Build all 4 IRAT networks.

        Returns:
            Tuple of (ind_actor, ind_critic, team_actor, team_critic)
        """
        action_dim = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space
        activation = self.config.get("ACTIVATION", "relu")

        ind_actor = IRATActorCNN(action_dim=action_dim, activation=activation)
        ind_critic = IRATCriticCNN(activation=activation)
        team_actor = IRATActorCNN(action_dim=action_dim, activation=activation)
        team_critic = IRATCriticCNN(activation=activation)

        return ind_actor, ind_critic, team_actor, team_critic

    def _build_optimizer(self) -> Tuple[Any, Any, Any, Any]:
        """Build 4 independent optimizers (one per network).

        Returns:
            Tuple of (ind_actor_opt, ind_critic_opt, team_actor_opt, team_critic_opt)
        """
        lr = self.config.get("LR", 5e-4)
        max_grad_norm = self.config.get("MAX_GRAD_NORM", 0.5)
        anneal_lr = self.config.get("ANNEAL_LR", True)

        def make_optimizer(lr_or_schedule):
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr_or_schedule, eps=1e-5),
            )

        # Static LR (schedule handled by train_irat.py for 1B step runs)
        return (
            make_optimizer(lr),
            make_optimizer(lr),
            make_optimizer(lr),
            make_optimizer(lr),
        )

    def init_state(self, rng: jax.random.PRNGKey) -> IRATAlgorithmState:
        """Initialize all 4 network states.

        Args:
            rng: JAX random key

        Returns:
            IRATAlgorithmState with initialized parameters
        """
        obs_shape = (
            self.observation_space.shape
            if hasattr(self.observation_space, 'shape')
            else self.observation_space
        )

        # Local obs for actors and individual critic
        actor_init_x = jnp.zeros((1, *obs_shape))

        # World state for team critic: stack all agents' channels
        world_state_shape = (*obs_shape[:-1], obs_shape[-1] * self.num_agents)
        critic_init_x = jnp.zeros((1, *world_state_shape))

        ind_actor, ind_critic, team_actor, team_critic = self.network

        rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

        ind_actor_state = TrainState.create(
            apply_fn=ind_actor.apply,
            params=ind_actor.init(r1, actor_init_x),
            tx=self.optimizer[0],
        )
        ind_critic_state = TrainState.create(
            apply_fn=ind_critic.apply,
            params=ind_critic.init(r2, actor_init_x),
            tx=self.optimizer[1],
        )
        team_actor_state = TrainState.create(
            apply_fn=team_actor.apply,
            params=team_actor.init(r3, actor_init_x),
            tx=self.optimizer[2],
        )
        team_critic_state = TrainState.create(
            apply_fn=team_critic.apply,
            params=team_critic.init(r4, critic_init_x),
            tx=self.optimizer[3],
        )

        return IRATAlgorithmState(
            ind_actor_state=ind_actor_state,
            ind_critic_state=ind_critic_state,
            team_actor_state=team_actor_state,
            team_critic_state=team_critic_state,
            rng=rng,
            timestep=0,
            update_step=0,
        )

    def compute_action(
        self,
        state: IRATAlgorithmState,
        observation: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute action using the TEAM actor (the one that executes in env).

        Args:
            state: Current IRAT algorithm state
            observation: Local observation array
            rng: Random key
            deterministic: If True, return greedy action

        Returns:
            Tuple of (action, info) where info contains log_prob
        """
        obs_shape = (
            self.observation_space.shape
            if hasattr(self.observation_space, 'shape')
            else self.observation_space
        )
        if observation.ndim == len(obs_shape):
            observation = observation[jnp.newaxis, ...]

        pi = state.team_actor_state.apply_fn(state.team_actor_state.params, observation)

        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            action = pi.sample(seed=rng)

        log_prob = pi.log_prob(action)

        return action.squeeze(0), {"log_prob": log_prob.squeeze(0)}

    def update(
        self,
        state: IRATAlgorithmState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[IRATAlgorithmState, Dict[str, float]]:
        """Update all 4 networks using a minibatch.

        This implements the core IRAT update:
        - Individual actor/critic: PPO update on individual rewards
        - Team actor/critic: PPO update on mean team rewards

        Args:
            state: Current IRAT algorithm state
            batch: Minibatch containing obs, world_state, actions,
                   ind_advantages, ind_targets, ind_values,
                   team_advantages, team_targets, team_values,
                   old_ind_log_probs, old_team_log_probs

        Returns:
            Tuple of (new_state, metrics)
        """
        clip_eps = self.config.get("CLIP_EPS", 0.2)
        vf_coef = self.config.get("VF_COEF", 0.5)
        ent_coef = self.config.get("ENT_COEF", 0.01)

        obs = batch["obs"]
        world_state = batch["world_state"]
        actions = batch["actions"]
        ind_advantages = batch["ind_advantages"]
        ind_targets = batch["ind_targets"]
        ind_values = batch["ind_values"]
        team_advantages = batch["team_advantages"]
        team_targets = batch["team_targets"]
        team_values = batch["team_values"]
        old_ind_log_probs = batch["old_ind_log_probs"]
        old_team_log_probs = batch["old_team_log_probs"]

        ind_actor_ts = state.ind_actor_state
        ind_critic_ts = state.ind_critic_state
        team_actor_ts = state.team_actor_state
        team_critic_ts = state.team_critic_state

        # Individual actor loss (PPO on individual rewards)
        def _ind_actor_loss(params):
            pi = ind_actor_ts.apply_fn(params, obs)
            log_prob = pi.log_prob(actions)
            entropy = pi.entropy().mean()
            ratio = jnp.exp(log_prob - old_ind_log_probs)
            adv_norm = (ind_advantages - ind_advantages.mean()) / (ind_advantages.std() + 1e-8)
            loss = -jnp.minimum(
                ratio * adv_norm,
                jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
            ).mean()
            return loss - ent_coef * entropy, (loss, entropy)

        # Individual critic loss (MSE on individual value targets)
        def _ind_critic_loss(params):
            value = ind_critic_ts.apply_fn(params, obs)
            value_clipped = ind_values + (value - ind_values).clip(-clip_eps, clip_eps)
            vf_loss = 0.5 * jnp.maximum(
                jnp.square(value - ind_targets),
                jnp.square(value_clipped - ind_targets)
            ).mean()
            return vf_coef * vf_loss, vf_loss

        # Team actor loss (PPO on team/mean rewards)
        def _team_actor_loss(params):
            pi = team_actor_ts.apply_fn(params, obs)
            log_prob = pi.log_prob(actions)
            entropy = pi.entropy().mean()
            ratio = jnp.exp(log_prob - old_team_log_probs)
            adv_norm = (team_advantages - team_advantages.mean()) / (team_advantages.std() + 1e-8)
            loss = -jnp.minimum(
                ratio * adv_norm,
                jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
            ).mean()
            return loss - ent_coef * entropy, (loss, entropy)

        # Team critic loss (MSE on team value targets, uses world state)
        def _team_critic_loss(params):
            value = team_critic_ts.apply_fn(params, world_state)
            value_clipped = team_values + (value - team_values).clip(-clip_eps, clip_eps)
            vf_loss = 0.5 * jnp.maximum(
                jnp.square(value - team_targets),
                jnp.square(value_clipped - team_targets)
            ).mean()
            return vf_coef * vf_loss, vf_loss

        # Compute gradients and update all 4 networks
        (ia_loss, (ia_loss_val, ia_entropy)), ia_grads = jax.value_and_grad(
            _ind_actor_loss, has_aux=True)(ind_actor_ts.params)
        (ic_loss, ic_vf_loss), ic_grads = jax.value_and_grad(
            _ind_critic_loss, has_aux=True)(ind_critic_ts.params)
        (ta_loss, (ta_loss_val, ta_entropy)), ta_grads = jax.value_and_grad(
            _team_actor_loss, has_aux=True)(team_actor_ts.params)
        (tc_loss, tc_vf_loss), tc_grads = jax.value_and_grad(
            _team_critic_loss, has_aux=True)(team_critic_ts.params)

        new_ind_actor_ts = ind_actor_ts.apply_gradients(grads=ia_grads)
        new_ind_critic_ts = ind_critic_ts.apply_gradients(grads=ic_grads)
        new_team_actor_ts = team_actor_ts.apply_gradients(grads=ta_grads)
        new_team_critic_ts = team_critic_ts.apply_gradients(grads=tc_grads)

        new_state = IRATAlgorithmState(
            ind_actor_state=new_ind_actor_ts,
            ind_critic_state=new_ind_critic_ts,
            team_actor_state=new_team_actor_ts,
            team_critic_state=new_team_critic_ts,
            rng=state.rng,
            timestep=state.timestep,
            update_step=state.update_step + 1,
        )

        metrics = {
            "total_loss": ia_loss + ic_loss + ta_loss + tc_loss,
            "ind_actor_loss": ia_loss_val,
            "ind_value_loss": ic_vf_loss,
            "team_actor_loss": ta_loss_val,
            "team_value_loss": tc_vf_loss,
            "ind_entropy": ia_entropy,
            "team_entropy": ta_entropy,
        }

        return new_state, metrics

    def save(self, path: str) -> None:
        """Save algorithm state."""
        import pickle
        from pathlib import Path
        if not hasattr(self, "_state") or self._state is None:
            return
        state = self._state
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "ind_actor_params": state.ind_actor_state.params,
            "ind_critic_params": state.ind_critic_state.params,
            "team_actor_params": state.team_actor_state.params,
            "team_critic_params": state.team_critic_state.params,
            "timestep": state.timestep,
            "update_step": state.update_step,
            "config": self.config,
            "num_agents": self.num_agents,
        }
        with open(path / "checkpoint.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, path: str) -> IRATAlgorithmState:
        """Load algorithm state."""
        import pickle
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        # Reconstruct TrainStates from saved params
        # (requires calling init_state first to get optimizer states)
        rng = jax.random.PRNGKey(0)
        init_state = self.init_state(rng)
        return IRATAlgorithmState(
            ind_actor_state=init_state.ind_actor_state.replace(
                params=save_dict["ind_actor_params"]),
            ind_critic_state=init_state.ind_critic_state.replace(
                params=save_dict["ind_critic_params"]),
            team_actor_state=init_state.team_actor_state.replace(
                params=save_dict["team_actor_params"]),
            team_critic_state=init_state.team_critic_state.replace(
                params=save_dict["team_critic_params"]),
            rng=rng,
            timestep=save_dict.get("timestep", 0),
            update_step=save_dict.get("update_step", 0),
        )
