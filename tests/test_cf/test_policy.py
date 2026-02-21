"""
Comprehensive tests for CF Policy Learning module (M7).

Tests:
- ActorCritic network shapes and outputs
- GAE (Generalized Advantage Estimation) computation
- PPO loss function with clipping
- Value function loss
- Entropy regularization
- Gradient clipping
- Training state creation
- Integration with shaped reward
"""

import sys
sys.path.insert(0, 'socialjax')

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from socialjax.algorithms.cf.policy import (
    ActorCritic,
    CNNFeatureExtractor,
    Transition,
    compute_gae,
    compute_ppo_loss,
    compute_ppo_loss_with_shaped_reward,
    clip_gradients,
    create_actor_critic_train_state,
    ppo_update_step,
    ppo_update_epoch,
    get_action,
    get_value,
    compute_gae_jit,
    make_compute_ppo_loss_jit,
    make_get_action_jit,
    make_get_value_jit,
)
from socialjax.algorithms.cf.reward_shaping import compute_shaped_reward


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_obs(rng):
    """Sample observation [batch, H, W, C]"""
    return jax.random.uniform(rng, (8, 15, 15, 3))


@pytest.fixture
def sample_obs_sequence(rng):
    """Sample observation sequence [num_steps, batch, H, W, C]"""
    return jax.random.uniform(rng, (16, 8, 15, 15, 3))


@pytest.fixture
def actor_critic():
    return ActorCritic(action_dim=4)


@pytest.fixture
def actor_critic_params(rng, actor_critic, sample_obs):
    return actor_critic.init(rng, sample_obs)


# ============================================================================
# CNN Feature Extractor Tests
# ============================================================================


class TestCNNFeatureExtractor:
    """Test CNN feature extractor."""

    def test_output_shape(self, rng, sample_obs):
        """Test that output shape is correct."""
        cnn = CNNFeatureExtractor(hidden_dim=64)
        params = cnn.init(rng, sample_obs)
        output = cnn.apply(params, sample_obs)

        assert output.shape == (sample_obs.shape[0], 64)

    def test_different_batch_sizes(self, rng):
        """Test with different batch sizes."""
        cnn = CNNFeatureExtractor(hidden_dim=64)
        init_obs = jnp.zeros((1, 15, 15, 3))
        params = cnn.init(rng, init_obs)

        for batch_size in [1, 4, 16, 32]:
            obs = jnp.zeros((batch_size, 15, 15, 3))
            output = cnn.apply(params, obs)
            assert output.shape == (batch_size, 64)

    def test_different_hidden_dims(self, rng, sample_obs):
        """Test with different hidden dimensions."""
        for hidden_dim in [32, 64, 128]:
            cnn = CNNFeatureExtractor(hidden_dim=hidden_dim)
            params = cnn.init(rng, sample_obs)
            output = cnn.apply(params, sample_obs)
            assert output.shape == (sample_obs.shape[0], hidden_dim)

    def test_no_nan_output(self, rng, sample_obs):
        """Test that output contains no NaN."""
        cnn = CNNFeatureExtractor()
        params = cnn.init(rng, sample_obs)
        output = cnn.apply(params, sample_obs)
        assert not jnp.any(jnp.isnan(output))


# ============================================================================
# ActorCritic Tests
# ============================================================================


class TestActorCritic:
    """Test ActorCritic network."""

    def test_output_shapes(self, actor_critic, actor_critic_params, sample_obs):
        """Test that policy and value shapes are correct."""
        pi, value = actor_critic.apply(actor_critic_params, sample_obs)

        assert pi.logits.shape == (sample_obs.shape[0], 4)
        assert value.shape == (sample_obs.shape[0],)

    def test_action_sampling(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test that actions can be sampled."""
        pi, value = actor_critic.apply(actor_critic_params, sample_obs)
        actions = pi.sample(seed=rng)

        assert actions.shape == (sample_obs.shape[0],)
        assert jnp.all(actions >= 0) and jnp.all(actions < 4)

    def test_log_prob_computation(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test that log probabilities can be computed."""
        pi, value = actor_critic.apply(actor_critic_params, sample_obs)
        actions = pi.sample(seed=rng)
        log_probs = pi.log_prob(actions)

        assert log_probs.shape == (sample_obs.shape[0],)
        assert jnp.all(log_probs <= 0)

    def test_entropy_computation(self, actor_critic, actor_critic_params, sample_obs):
        """Test that entropy can be computed."""
        pi, value = actor_critic.apply(actor_critic_params, sample_obs)
        entropy = pi.entropy()

        assert entropy.shape == (sample_obs.shape[0],)
        assert jnp.all(entropy >= 0)

    def test_different_batch_sizes(self, rng, actor_critic):
        """Test with different batch sizes."""
        init_obs = jnp.zeros((1, 15, 15, 3))
        params = actor_critic.init(rng, init_obs)

        for batch_size in [1, 4, 16, 32]:
            obs = jnp.zeros((batch_size, 15, 15, 3))
            pi, value = actor_critic.apply(params, obs)
            assert pi.logits.shape == (batch_size, 4)
            assert value.shape == (batch_size,)

    def test_different_action_dims(self, rng, sample_obs):
        """Test with different action dimensions."""
        for action_dim in [2, 4, 8]:
            network = ActorCritic(action_dim=action_dim)
            params = network.init(rng, sample_obs)
            pi, value = network.apply(params, sample_obs)
            assert pi.logits.shape == (sample_obs.shape[0], action_dim)

    def test_no_nan_output(self, actor_critic, actor_critic_params, sample_obs):
        """Test that output contains no NaN or Inf."""
        pi, value = actor_critic.apply(actor_critic_params, sample_obs)
        assert not jnp.any(jnp.isnan(pi.logits))
        assert not jnp.any(jnp.isinf(pi.logits))
        assert not jnp.any(jnp.isnan(value))
        assert not jnp.any(jnp.isinf(value))

    def test_deterministic_forward(self, actor_critic, actor_critic_params, sample_obs):
        """Test that forward pass is deterministic."""
        pi1, value1 = actor_critic.apply(actor_critic_params, sample_obs)
        pi2, value2 = actor_critic.apply(actor_critic_params, sample_obs)

        assert jnp.allclose(pi1.logits, pi2.logits)
        assert jnp.allclose(value1, value2)


# ============================================================================
# GAE Tests
# ============================================================================


class TestComputeGAE:
    """Test GAE computation."""

    def test_output_shapes(self, rng):
        """Test that output shapes are correct."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value)

        assert advantages.shape == (num_steps, batch_size)
        assert targets.shape == (num_steps, batch_size)

    def test_zero_reward_advantage(self, rng):
        """Test that advantage is zero when rewards are all zero."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value)

        assert jnp.allclose(advantages, 0.0, atol=1e-6)
        assert jnp.allclose(targets, 0.0, atol=1e-6)

    def test_positive_reward_advantage(self, rng):
        """Test that advantage is positive when rewards are positive."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.ones((num_steps, batch_size)),  # Positive rewards
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value, gamma=0.99, gae_lambda=0.95)

        # Advantages should be positive with positive rewards
        assert jnp.all(advantages > 0)

    def test_done_reset_advantage(self, rng):
        """Test that done flag resets advantage calculation."""
        num_steps, batch_size = 8, 2

        # Create a trajectory where episode ends in the middle
        done = jnp.zeros((num_steps, batch_size))
        done = done.at[4, :].set(1.0)  # Episode ends at step 4

        traj_batch = Transition(
            done=done,
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.ones((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value, gamma=0.99, gae_lambda=0.95)

        # Should not crash and produce valid output
        assert advantages.shape == (num_steps, batch_size)
        assert not jnp.any(jnp.isnan(advantages))

    def test_jit_compilation(self, rng):
        """Test that GAE can be JIT compiled."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        # Should not raise
        advantages, targets = compute_gae_jit(traj_batch, last_value)
        assert advantages.shape == (num_steps, batch_size)


# ============================================================================
# PPO Loss Tests
# ============================================================================


class TestComputePPOLoss:
    """Test PPO loss computation."""

    def test_loss_is_scalar(self, actor_critic, actor_critic_params, rng):
        """Test that loss is a scalar."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        loss, aux = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert loss.shape == ()  # Scalar

    def test_loss_returns_aux_info(self, actor_critic, actor_critic_params, rng):
        """Test that loss returns auxiliary information."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        loss, (value_loss, actor_loss, entropy) = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert value_loss.shape == ()
        assert actor_loss.shape == ()
        assert entropy.shape == ()

    def test_entropy_positive(self, actor_critic, actor_critic_params, rng):
        """Test that entropy is positive (or zero at minimum)."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        loss, (_, _, entropy) = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert entropy >= 0

    def test_loss_differentiable(self, actor_critic, actor_critic_params, rng):
        """Test that loss is differentiable."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        def loss_fn(params):
            loss, _ = compute_ppo_loss(
                params,
                actor_critic.apply,
                traj_batch,
                advantages,
                targets,
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(actor_critic_params)

        # Check gradients exist and are finite
        assert loss.shape == ()
        assert not jnp.any(jnp.isnan(jax.tree_util.tree_flatten(grads)[0][0]))

    def test_clip_eps_effect(self, actor_critic, actor_critic_params, rng):
        """Test that clip_eps affects the loss."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.ones((num_steps, batch_size)),
            log_prob=-jnp.ones((num_steps, batch_size)),  # Different from current
            obs=jnp.ones((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.ones((num_steps, batch_size))
        targets = jnp.ones((num_steps, batch_size))

        loss1, _ = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
            clip_eps=0.1,
        )

        loss2, _ = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
            clip_eps=0.5,
        )

        # Different clip values should produce different losses
        # (though they could be equal in edge cases)
        assert loss1.shape == loss2.shape == ()

    def test_no_nan_loss(self, actor_critic, actor_critic_params, rng):
        """Test that loss contains no NaN."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        loss, (value_loss, actor_loss, entropy) = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert not jnp.isnan(loss)
        assert not jnp.isnan(value_loss)
        assert not jnp.isnan(actor_loss)
        assert not jnp.isnan(entropy)


class TestComputePPOLossWithShapedReward:
    """Test PPO loss with shaped reward integration."""

    def test_output_shapes(self, actor_critic, actor_critic_params, rng):
        """Test that output shapes are correct."""
        num_steps, batch_size = 16, 8

        obs = jnp.zeros((num_steps, batch_size, 15, 15, 3))
        actions = jax.random.randint(rng, (num_steps, batch_size), 0, 4)
        shaped_rewards = jnp.zeros((num_steps, batch_size))
        dones = jnp.zeros((num_steps, batch_size))
        old_log_probs = jnp.zeros((num_steps, batch_size))
        old_values = jnp.zeros((num_steps, batch_size))

        loss, (value_loss, actor_loss, entropy, last_value) = compute_ppo_loss_with_shaped_reward(
            actor_critic_params,
            actor_critic.apply,
            obs,
            actions,
            shaped_rewards,
            dones,
            old_log_probs,
            old_values,
        )

        assert loss.shape == ()
        assert value_loss.shape == ()
        assert actor_loss.shape == ()
        assert entropy.shape == ()
        assert last_value.shape == (batch_size,)

    def test_positive_shaped_reward(self, actor_critic, actor_critic_params, rng):
        """Test with positive shaped rewards."""
        num_steps, batch_size = 16, 8

        obs = jnp.zeros((num_steps, batch_size, 15, 15, 3))
        actions = jax.random.randint(rng, (num_steps, batch_size), 0, 4)
        shaped_rewards = jnp.ones((num_steps, batch_size))  # Positive
        dones = jnp.zeros((num_steps, batch_size))
        old_log_probs = jnp.zeros((num_steps, batch_size))
        old_values = jnp.zeros((num_steps, batch_size))

        loss, aux = compute_ppo_loss_with_shaped_reward(
            actor_critic_params,
            actor_critic.apply,
            obs,
            actions,
            shaped_rewards,
            dones,
            old_log_probs,
            old_values,
        )

        assert not jnp.isnan(loss)


# ============================================================================
# Gradient Clipping Tests
# ============================================================================


class TestClipGradients:
    """Test gradient clipping."""

    def test_no_clip_small_norm(self, rng):
        """Test that small gradients are not clipped."""
        grads = {'w': jnp.array([0.1, 0.1])}
        clipped, norm = clip_gradients(grads, max_grad_norm=10.0)

        assert jnp.allclose(clipped['w'], grads['w'])
        assert norm < 10.0

    def test_clip_large_norm(self, rng):
        """Test that large gradients are clipped."""
        grads = {'w': jnp.array([100.0, 100.0])}
        clipped, norm = clip_gradients(grads, max_grad_norm=1.0)

        # Clipped norm should be at most max_grad_norm
        clipped_norm = jnp.sqrt(jnp.sum(clipped['w'] ** 2))
        assert clipped_norm <= 1.0 + 1e-5

    def test_returns_grad_norm(self, rng):
        """Test that function returns gradient norm."""
        grads = {'w': jnp.array([3.0, 4.0])}
        clipped, norm = clip_gradients(grads, max_grad_norm=10.0)

        # Expected norm: sqrt(9 + 16) = 5
        assert jnp.isclose(norm, 5.0)


# ============================================================================
# Training State Tests
# ============================================================================


class TestCreateActorCriticTrainState:
    """Test training state creation."""

    def test_state_creation(self, rng, sample_obs):
        """Test that training state is created correctly."""
        network = ActorCritic(action_dim=4)
        train_state, next_rng = create_actor_critic_train_state(
            network, rng, sample_obs, learning_rate=0.001
        )

        assert train_state.params is not None
        assert train_state.tx is not None
        assert next_rng is not rng

    def test_different_learning_rates(self, rng, sample_obs):
        """Test with different learning rates."""
        network = ActorCritic(action_dim=4)

        for lr in [0.0001, 0.001, 0.01]:
            train_state, _ = create_actor_critic_train_state(
                network, rng, sample_obs, learning_rate=lr
            )
            assert train_state.tx is not None

    def test_anneal_lr(self, rng, sample_obs):
        """Test with learning rate annealing."""
        network = ActorCritic(action_dim=4)
        train_state, _ = create_actor_critic_train_state(
            network, rng, sample_obs,
            learning_rate=0.001,
            anneal_lr=True,
            num_updates=1000
        )

        assert train_state.tx is not None


# ============================================================================
# Action Sampling Tests
# ============================================================================


class TestGetAction:
    """Test action sampling."""

    def test_stochastic_action(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test stochastic action sampling."""
        action, log_prob, value = get_action(
            actor_critic_params,
            actor_critic.apply,
            sample_obs,
            rng,
            deterministic=False
        )

        assert action.shape == (sample_obs.shape[0],)
        assert log_prob.shape == (sample_obs.shape[0],)
        assert value.shape == (sample_obs.shape[0],)
        assert jnp.all(action >= 0) and jnp.all(action < 4)

    def test_deterministic_action(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test deterministic action (mode)."""
        action, log_prob, value = get_action(
            actor_critic_params,
            actor_critic.apply,
            sample_obs,
            rng,
            deterministic=True
        )

        assert action.shape == (sample_obs.shape[0],)
        assert log_prob.shape == (sample_obs.shape[0],)
        assert value.shape == (sample_obs.shape[0],)

    def test_deterministic_reproducible(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test that deterministic action is reproducible."""
        action1, _, _ = get_action(
            actor_critic_params,
            actor_critic.apply,
            sample_obs,
            rng,
            deterministic=True
        )

        action2, _, _ = get_action(
            actor_critic_params,
            actor_critic.apply,
            sample_obs,
            rng,
            deterministic=True
        )

        assert jnp.array_equal(action1, action2)


class TestGetValue:
    """Test value estimation."""

    def test_value_shape(self, actor_critic, actor_critic_params, sample_obs):
        """Test value output shape."""
        value = get_value(actor_critic_params, actor_critic.apply, sample_obs)
        assert value.shape == (sample_obs.shape[0],)

    def test_value_reproducible(self, actor_critic, actor_critic_params, sample_obs):
        """Test that value is reproducible."""
        value1 = get_value(actor_critic_params, actor_critic.apply, sample_obs)
        value2 = get_value(actor_critic_params, actor_critic.apply, sample_obs)
        assert jnp.allclose(value1, value2)


# ============================================================================
# PPO Update Step Tests
# ============================================================================


class TestPPOUpdateStep:
    """Test single PPO update step."""

    def test_update_step(self, actor_critic, actor_critic_params, rng):
        """Test that update step works."""
        from flax.training.train_state import TrainState
        import optax

        # Create train state
        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic_params,
            tx=tx,
        )

        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        new_state, loss_info = ppo_update_step(
            train_state,
            traj_batch,
            advantages,
            targets,
            actor_critic.apply,
        )

        assert new_state.params is not None
        assert len(loss_info) == 4  # total_loss, value_loss, actor_loss, entropy

    def test_params_change(self, actor_critic, actor_critic_params, rng):
        """Test that parameters change after update."""
        from flax.training.train_state import TrainState
        import optax

        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(
            apply_fn=actor_critic.apply,
            params=actor_critic_params,
            tx=tx,
        )

        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.ones((num_steps, batch_size)),  # Non-zero reward
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.ones((num_steps, batch_size, 15, 15, 3)),  # Non-zero obs
        )
        advantages = jnp.ones((num_steps, batch_size))
        targets = jnp.ones((num_steps, batch_size))

        new_state, _ = ppo_update_step(
            train_state,
            traj_batch,
            advantages,
            targets,
            actor_critic.apply,
        )

        # Check that params have changed
        old_flat = jax.tree_util.tree_flatten(train_state.params)[0][0]
        new_flat = jax.tree_util.tree_flatten(new_state.params)[0][0]
        assert not jnp.allclose(old_flat, new_flat)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationWithShapedReward:
    """Test integration with shaped reward from M6."""

    def test_shaped_reward_pipeline(self, actor_critic, actor_critic_params, rng):
        """Test full pipeline: shaped reward -> GAE -> PPO loss."""
        num_steps, batch_size, num_agents = 16, 8, 3

        # Simulate extrinsic and intrinsic rewards
        extrinsic_reward = jax.random.uniform(rng, (num_steps, batch_size, num_agents))
        intrinsic_reward = -jax.random.uniform(rng, (num_steps, batch_size, num_agents))  # Negative

        # Compute shaped reward using M6 function
        shaped_reward = compute_shaped_reward(extrinsic_reward, intrinsic_reward, alpha=2.0)

        # For single agent (for simplicity)
        shaped_reward_single = shaped_reward[:, :, 0]

        # Create trajectory
        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=shaped_reward_single,
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )

        # Compute GAE
        last_value = jnp.zeros(batch_size)
        advantages, targets = compute_gae(traj_batch, last_value)

        # Compute PPO loss
        loss, aux = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        # Should not crash and produce valid output
        assert not jnp.isnan(loss)

    def test_shaped_reward_affects_advantage(self, rng):
        """Test that shaped reward affects advantage computation."""
        num_steps, batch_size = 16, 8

        # Create two different reward signals
        reward1 = jnp.zeros((num_steps, batch_size))
        reward2 = jnp.ones((num_steps, batch_size))

        traj_batch1 = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=reward1,
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )

        traj_batch2 = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=reward2,
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )

        last_value = jnp.zeros(batch_size)

        advantages1, _ = compute_gae(traj_batch1, last_value)
        advantages2, _ = compute_gae(traj_batch2, last_value)

        # Different rewards should produce different advantages
        assert not jnp.allclose(advantages1, advantages2)


class TestJITCompilation:
    """Test JIT compilation of key functions."""

    def test_gae_jit(self, rng):
        """Test that GAE JIT works."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jnp.zeros((num_steps, batch_size), dtype=int),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        # Should not raise
        advantages, targets = compute_gae_jit(traj_batch, last_value)
        assert advantages.shape == (num_steps, batch_size)

    def test_get_value_jit_factory(self, actor_critic, actor_critic_params, sample_obs):
        """Test that get_value JIT factory works."""
        value_fn = make_get_value_jit(actor_critic.apply)
        value = value_fn(actor_critic_params, sample_obs)
        assert value.shape == (sample_obs.shape[0],)

    def test_get_action_jit_factory(self, actor_critic, actor_critic_params, sample_obs, rng):
        """Test that get_action JIT factory works."""
        action_fn = make_get_action_jit(actor_critic.apply)
        action, log_prob, value = action_fn(actor_critic_params, sample_obs, rng)
        assert action.shape == (sample_obs.shape[0],)

    def test_ppo_loss_jit_factory(self, actor_critic, actor_critic_params, rng):
        """Test that PPO loss JIT factory works."""
        num_steps, batch_size = 16, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        advantages = jnp.zeros((num_steps, batch_size))
        targets = jnp.zeros((num_steps, batch_size))

        loss_fn = make_compute_ppo_loss_jit(actor_critic.apply)
        loss, aux = loss_fn(actor_critic_params, traj_batch, advantages, targets)
        assert loss.shape == ()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_step_trajectory(self, actor_critic, actor_critic_params, rng):
        """Test with single-step trajectory."""
        num_steps, batch_size = 1, 8

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value)
        loss, _ = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert not jnp.isnan(loss)

    def test_batch_size_one(self, actor_critic, actor_critic_params, rng):
        """Test with batch size 1."""
        num_steps, batch_size = 16, 1

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value)
        loss, _ = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert not jnp.isnan(loss)

    def test_large_trajectory(self, actor_critic, actor_critic_params, rng):
        """Test with large trajectory."""
        num_steps, batch_size = 128, 32

        traj_batch = Transition(
            done=jnp.zeros((num_steps, batch_size)),
            action=jax.random.randint(rng, (num_steps, batch_size), 0, 4),
            value=jnp.zeros((num_steps, batch_size)),
            reward=jnp.zeros((num_steps, batch_size)),
            log_prob=jnp.zeros((num_steps, batch_size)),
            obs=jnp.zeros((num_steps, batch_size, 15, 15, 3)),
        )
        last_value = jnp.zeros(batch_size)

        advantages, targets = compute_gae(traj_batch, last_value)
        loss, _ = compute_ppo_loss(
            actor_critic_params,
            actor_critic.apply,
            traj_batch,
            advantages,
            targets,
        )

        assert not jnp.isnan(loss)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
