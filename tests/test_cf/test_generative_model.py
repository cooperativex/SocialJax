"""
Unit tests for CF-IMPL-001: Generative Model (RewardModel)

Test criteria:
- Output shape correct: [batch, num_agents]
- MSE loss computable and differentiable
- Support different batch_size
- Support different num_agents
- No NaN/Inf
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np

from socialjax.algorithms.cf.generative_model import (
    RewardModel,
    CNNFeatureExtractor,
    generative_model_loss,
    compute_generative_model_loss,
    create_reward_model_train_state,
)


class TestCNNFeatureExtractor:
    """Test CNN feature extractor"""

    def test_output_shape(self, key):
        """CNN output should have shape [batch, hidden_dim]"""
        cnn = CNNFeatureExtractor(hidden_dim=64)
        obs = jnp.zeros((8, 15, 15, 4))  # [batch, H, W, C]

        params = cnn.init(key, obs)
        output = cnn.apply(params, obs)

        assert output.shape == (8, 64), f"Expected (8, 64), got {output.shape}"

    def test_different_batch_sizes(self, key):
        """CNN should handle different batch sizes"""
        cnn = CNNFeatureExtractor(hidden_dim=32)

        for batch_size in [1, 8, 32, 64]:
            obs = jnp.zeros((batch_size, 15, 15, 4))
            params = cnn.init(key, obs)
            output = cnn.apply(params, obs)
            assert output.shape == (batch_size, 32)

    def test_no_nan_inf(self, key):
        """CNN output should not contain NaN or Inf"""
        cnn = CNNFeatureExtractor(hidden_dim=64)
        obs = jnp.ones((4, 15, 15, 4))

        params = cnn.init(key, obs)
        output = cnn.apply(params, obs)

        assert jnp.all(jnp.isfinite(output)), "CNN output contains NaN or Inf"


class TestRewardModel:
    """Test RewardModel class"""

    def test_output_shape_batch4_agents3(self, key):
        """Test output shape with batch=4, num_agents=3"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))  # [batch, num_agents, H, W, C]
        actions = jnp.zeros((4, 3), dtype=jnp.int32)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_different_batch_sizes(self, key, batch_size):
        """Test with different batch sizes"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((batch_size, 3, 15, 15, 4))
        actions = jnp.zeros((batch_size, 3), dtype=jnp.int32)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (batch_size, 3)

    @pytest.mark.parametrize("num_agents", [2, 3, 4, 5, 7])
    def test_different_num_agents(self, key, num_agents):
        """Test with different number of agents"""
        model = RewardModel(num_agents=num_agents, action_dim=4)
        obs = jnp.zeros((4, num_agents, 15, 15, 4))
        actions = jnp.zeros((4, num_agents), dtype=jnp.int32)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (4, num_agents)

    @pytest.mark.parametrize("action_dim", [2, 4, 8])
    def test_different_action_dims(self, key, action_dim):
        """Test with different action dimensions"""
        model = RewardModel(num_agents=3, action_dim=action_dim)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (4, 3)

    def test_no_nan_inf(self, key):
        """Output should not contain NaN or Inf"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.ones((4, 3, 15, 15, 4))
        actions = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]])

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert jnp.all(jnp.isfinite(output)), "RewardModel output contains NaN or Inf"

    def test_random_input(self, key):
        """Test with random input"""
        model = RewardModel(num_agents=3, action_dim=4)

        key, subkey = jax.random.split(key)
        obs = jax.random.normal(subkey, (8, 3, 15, 15, 4))

        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (8, 3), 0, 4)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (8, 3)
        assert jnp.all(jnp.isfinite(output))


class TestGenerativeModelLoss:
    """Test generative_model_loss function"""

    def test_loss_shape_mean(self, key):
        """Mean reduction should return scalar"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([[1.5, 2.5], [2.5, 3.5]])

        loss = generative_model_loss(predicted, actual, reduction="mean")
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_shape_sum(self, key):
        """Sum reduction should return scalar"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([[1.5, 2.5], [2.5, 3.5]])

        loss = generative_model_loss(predicted, actual, reduction="sum")
        assert loss.shape == ()

    def test_loss_shape_none(self, key):
        """None reduction should return [batch, num_agents]"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([[1.5, 2.5], [2.5, 3.5]])

        loss = generative_model_loss(predicted, actual, reduction="none")
        assert loss.shape == (2, 2), f"Expected (2, 2), got {loss.shape}"

    def test_loss_zero_when_equal(self, key):
        """Loss should be zero when predictions equal actual"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = predicted

        loss = generative_model_loss(predicted, actual, reduction="mean")
        assert jnp.allclose(loss, 0.0, atol=1e-6)

    def test_loss_positive_when_different(self, key):
        """Loss should be positive when predictions differ from actual"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([[1.5, 2.5], [2.5, 3.5]])

        loss = generative_model_loss(predicted, actual, reduction="mean")
        assert loss > 0

    def test_loss_with_mask(self, key):
        """Loss with mask should ignore masked entries"""
        predicted = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([[1.5, 2.5], [2.5, 3.5]])
        mask = jnp.array([[1.0, 0.0], [1.0, 1.0]])  # Ignore [0,1]

        loss = generative_model_loss(predicted, actual, mask=mask, reduction="mean")
        # Only consider [0,0], [1,0], [1,1]: (0.25 + 0.25 + 0.25) / 3 = 0.25
        expected = ((1.0-1.5)**2 + (3.0-2.5)**2 + (4.0-3.5)**2) / 3
        assert jnp.allclose(loss, expected, atol=1e-6)


class TestComputeGenerativeModelLoss:
    """Test compute_generative_model_loss function"""

    def test_loss_is_scalar(self, key):
        """Loss should be a scalar"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jnp.zeros((4, 3))

        params = model.init(key, obs, actions)
        loss, predicted = compute_generative_model_loss(params, model, obs, actions, rewards)

        assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"

    def test_predicted_rewards_shape(self, key):
        """Predicted rewards should have shape [batch, num_agents]"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jnp.zeros((4, 3))

        params = model.init(key, obs, actions)
        loss, predicted = compute_generative_model_loss(params, model, obs, actions, rewards)

        assert predicted.shape == (4, 3), f"Expected (4, 3), got {predicted.shape}"

    def test_loss_is_differentiable(self, key):
        """Loss should be differentiable (gradient exists)"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.ones((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jnp.ones((4, 3))

        params = model.init(key, obs, actions)

        # Define loss function for gradient computation
        def loss_fn(params):
            loss, _ = compute_generative_model_loss(params, model, obs, actions, rewards)
            return loss

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # Check that gradients are not None and have finite values
        assert grads is not None
        for v in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(v)), "Gradient contains NaN/Inf"


class TestCreateRewardModelTrainState:
    """Test create_reward_model_train_state function"""

    def test_creates_train_state(self, key):
        """Should create a valid TrainState"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)

        train_state, next_rng = create_reward_model_train_state(
            model, key, obs, actions, learning_rate=0.001
        )

        assert train_state is not None
        assert train_state.params is not None
        assert next_rng is not None

    def test_train_state_has_optimizer(self, key):
        """TrainState should have optimizer"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)

        train_state, _ = create_reward_model_train_state(
            model, key, obs, actions, learning_rate=0.001
        )

        # Check that tx (optimizer) is present
        assert hasattr(train_state, 'tx')

    def test_different_learning_rates(self, key):
        """Should work with different learning rates"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.zeros((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)

        for lr in [0.0001, 0.001, 0.01]:
            train_state, _ = create_reward_model_train_state(
                model, key, obs, actions, learning_rate=lr
            )
            assert train_state is not None


class TestIntegration:
    """Integration tests with actual environment"""

    @pytest.fixture
    def coin_game_env(self):
        """Create coin_game environment"""
        import socialjax
        return socialjax.make('coin_game', num_agents=3)

    def test_with_real_environment(self, key, coin_game_env):
        """Test with real coin_game environment observations"""
        model = RewardModel(num_agents=3, action_dim=4)

        # Get actual observation from environment
        key, reset_key = jax.random.split(key)
        obs, state = coin_game_env.reset(reset_key)

        # obs shape might be different, let's check
        if len(obs.shape) == 4:  # [num_agents, H, W, C]
            obs = obs[jnp.newaxis, ...]  # Add batch dim -> [1, num_agents, H, W, C]

        actions = jnp.zeros((1, 3), dtype=jnp.int32)

        params = model.init(key, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape[1] == 3  # num_agents
        assert jnp.all(jnp.isfinite(output))


class TestGradientFlow:
    """Test gradient flow through the model"""

    def test_gradient_exists_for_all_parameters(self, key):
        """Gradient should exist for all model parameters"""
        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.ones((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jnp.ones((4, 3))

        params = model.init(key, obs, actions)

        def loss_fn(params):
            pred = model.apply(params, obs, actions)
            return jnp.mean((pred - rewards) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Check all parameters have gradients
        param_count = len(jax.tree_util.tree_leaves(params))
        grad_count = len(jax.tree_util.tree_leaves(grads))
        assert param_count == grad_count, f"Param count {param_count} != grad count {grad_count}"

    def test_gradient_updates_change_parameters(self, key):
        """Gradient update should change parameters"""
        import optax

        model = RewardModel(num_agents=3, action_dim=4)
        obs = jnp.ones((4, 3, 15, 15, 4))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jnp.ones((4, 3))

        params = model.init(key, obs, actions)
        original_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

        def loss_fn(params):
            pred = model.apply(params, obs, actions)
            return jnp.mean((pred - rewards) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Apply gradient update
        tx = optax.adam(learning_rate=0.01)
        opt_state = tx.init(params)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        # Check that at least some parameters changed
        original_leaves = jax.tree_util.tree_leaves(original_params)
        new_leaves = jax.tree_util.tree_leaves(new_params)

        params_changed = False
        for v1, v2 in zip(original_leaves, new_leaves):
            if v1.size > 0 and not jnp.allclose(v1, v2):
                params_changed = True
                break
        assert params_changed, "Parameters didn't change after gradient update"


class TestCFDebug004TrainingLoss:
    """CF-DEBUG-004: Debug training loss - verify loss decreases, no NaN/Inf, learning rate appropriate

    Test criteria:
    - Loss decreases during training
    - No NaN/Inf values
    - Learning rate is appropriate
    - Loss calculation matches Eq.6
    """

    def test_loss_decreases_over_training(self, key):
        """Loss should decrease over multiple training steps"""
        import optax
        from flax.training.train_state import TrainState

        # Setup model and data
        model = RewardModel(num_agents=3, action_dim=4)
        batch_size = 32

        # Create random but consistent data for training
        key, obs_key, action_key, reward_key = jax.random.split(key, 4)
        obs = jax.random.normal(obs_key, (batch_size, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (batch_size, 3), 0, 4)
        # Create target rewards with some pattern for the model to learn
        rewards = jax.random.normal(reward_key, (batch_size, 3))

        # Initialize model and optimizer
        params = model.init(key, obs, actions)
        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @jax.jit
        def train_step(train_state, obs, actions, rewards):
            def loss_fn(params):
                pred = model.apply(params, obs, actions)
                loss = jnp.mean((pred - rewards) ** 2)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        # Track losses over training
        losses = []
        for _ in range(50):
            train_state, loss = train_step(train_state, obs, actions, rewards)
            losses.append(float(loss))

        # Verify loss decreased
        initial_loss = losses[0]
        final_loss = losses[-1]

        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

        # Verify overall downward trend (last 10 should be lower than first 10)
        early_avg = jnp.mean(jnp.array(losses[:10]))
        late_avg = jnp.mean(jnp.array(losses[-10:]))
        assert late_avg < early_avg, \
            f"Loss trend should decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"

    def test_no_nan_inf_during_training(self, key):
        """No NaN or Inf values should appear during training"""
        import optax
        from flax.training.train_state import TrainState

        model = RewardModel(num_agents=3, action_dim=4)
        batch_size = 32

        key, obs_key, action_key, reward_key = jax.random.split(key, 4)
        obs = jax.random.normal(obs_key, (batch_size, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (batch_size, 3), 0, 4)
        rewards = jax.random.normal(reward_key, (batch_size, 3))

        params = model.init(key, obs, actions)
        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @jax.jit
        def train_step(train_state, obs, actions, rewards):
            def loss_fn(params):
                pred = model.apply(params, obs, actions)
                loss = jnp.mean((pred - rewards) ** 2)
                return loss, pred

            (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss, pred

        # Run training and check for NaN/Inf
        for step in range(100):
            train_state, loss, pred = train_step(train_state, obs, actions, rewards)

            # Check loss
            assert jnp.isfinite(loss), f"Loss became NaN/Inf at step {step}: {loss}"

            # Check predictions
            assert jnp.all(jnp.isfinite(pred)), f"Predictions contain NaN/Inf at step {step}"

            # Check parameters
            for name, param in jax.tree_util.tree_leaves_with_path(train_state.params):
                assert jnp.all(jnp.isfinite(param)), \
                    f"Parameter became NaN/Inf at step {step}"

    def test_learning_rate_appropriate(self, key):
        """Test that standard learning rates work (not too high/low)"""
        import optax
        from flax.training.train_state import TrainState

        model = RewardModel(num_agents=3, action_dim=4)
        batch_size = 32

        key, obs_key, action_key, reward_key = jax.random.split(key, 4)
        obs = jax.random.normal(obs_key, (batch_size, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (batch_size, 3), 0, 4)
        rewards = jax.random.normal(reward_key, (batch_size, 3))

        # Test different learning rates
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        results = {}

        for lr in learning_rates:
            params = model.init(key, obs, actions)
            tx = optax.adam(learning_rate=lr)
            train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            @jax.jit
            def train_step(train_state, obs, actions, rewards):
                def loss_fn(params):
                    pred = model.apply(params, obs, actions)
                    return jnp.mean((pred - rewards) ** 2)

                loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            losses = []
            for _ in range(50):
                train_state, loss = train_step(train_state, obs, actions, rewards)
                losses.append(float(loss))

            results[lr] = {
                'initial': losses[0],
                'final': losses[-1],
                'min': min(losses),
                'has_nan': any(not jnp.isfinite(l) for l in losses)
            }

        # Check that at least some learning rates work well
        working_lrs = [lr for lr, r in results.items()
                       if not r['has_nan'] and r['final'] < r['initial']]

        assert len(working_lrs) > 0, "No learning rates worked successfully"

        # Check that standard learning rate 0.001 works
        assert 0.001 in working_lrs, f"Standard LR 0.001 failed: {results[0.001]}"

    def test_loss_matches_eq6_mse(self, key):
        """Verify loss calculation matches Eq.6 (MSE)"""
        model = RewardModel(num_agents=3, action_dim=4)

        key, obs_key, action_key = jax.random.split(key, 3)
        obs = jax.random.normal(obs_key, (4, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (4, 3), 0, 4)

        params = model.init(key, obs, actions)
        predicted = model.apply(params, obs, actions)

        # Set actual rewards with known difference
        actual = predicted + 1.0  # Every prediction is off by 1.0

        # Expected MSE: mean of 1^2 = 1.0
        expected_mse = 1.0

        # Test generative_model_loss function
        loss = generative_model_loss(predicted, actual, reduction="mean")

        assert jnp.allclose(loss, expected_mse, atol=1e-6), \
            f"Expected MSE={expected_mse}, got {loss}"

    def test_loss_computation_with_real_data(self, key):
        """Test loss computation with data that should be learnable"""
        import optax
        from flax.training.train_state import TrainState

        model = RewardModel(num_agents=3, action_dim=4)
        batch_size = 64

        # Create data where reward is a simple function of action
        key, obs_key = jax.random.split(key)
        obs = jax.random.normal(obs_key, (batch_size, 3, 15, 15, 4))
        key, action_key = jax.random.split(key)
        actions = jax.random.randint(action_key, (batch_size, 3), 0, 4)

        # Create rewards with a simple pattern: reward = action_value * 0.1
        rewards = actions.astype(jnp.float32) * 0.1

        params = model.init(key, obs, actions)
        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @jax.jit
        def train_step(train_state, obs, actions, rewards):
            def loss_fn(params):
                pred = model.apply(params, obs, actions)
                return jnp.mean((pred - rewards) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        # Train for several steps
        losses = []
        for _ in range(100):
            train_state, loss = train_step(train_state, obs, actions, rewards)
            losses.append(float(loss))

        # Verify training progresses
        assert jnp.all(jnp.isfinite(jnp.array(losses))), "NaN/Inf in losses"

        # Loss should decrease significantly
        improvement_ratio = losses[0] / (losses[-1] + 1e-8)
        assert improvement_ratio > 1.5, \
            f"Insufficient loss improvement: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_training_with_environment_data(self, key):
        """Test training with actual environment data"""
        import socialjax
        import optax
        from flax.training.train_state import TrainState

        env = socialjax.make('coin_game', num_agents=3)

        # Collect some real transitions
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        # Get model and adapter info
        num_agents = env.num_agents
        action_dim = 7  # coin_game has 7 actions

        model = RewardModel(num_agents=num_agents, action_dim=action_dim)

        # Add batch dimension if needed
        if len(obs.shape) == 4:  # [num_agents, H, W, C]
            obs = obs[jnp.newaxis, ...]  # [1, num_agents, H, W, C]

        # Sample random actions
        key, action_key = jax.random.split(key)
        actions = jax.random.randint(action_key, (1, num_agents), 0, action_dim)

        # Step environment - returns (obs2, state2, rewards, dones, infos)
        key, step_key = jax.random.split(key)
        obs2, state2, rewards, dones, infos = env.step(step_key, state, actions[0])

        # Initialize model
        params = model.init(key, obs, actions)

        # Compute loss
        loss, pred = compute_generative_model_loss(params, model, obs, actions, rewards[jnp.newaxis, ...])

        # Verify outputs
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        assert jnp.all(jnp.isfinite(pred)), "Predictions contain NaN/Inf"
        assert loss >= 0, f"Loss should be non-negative: {loss}"

    def test_gradient_norm_is_reasonable(self, key):
        """Gradient norms should be reasonable (not too large/small)"""
        model = RewardModel(num_agents=3, action_dim=4)

        key, obs_key, action_key, reward_key = jax.random.split(key, 4)
        obs = jax.random.normal(obs_key, (32, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (32, 3), 0, 4)
        rewards = jax.random.normal(reward_key, (32, 3))

        params = model.init(key, obs, actions)

        def loss_fn(params):
            pred = model.apply(params, obs, actions)
            return jnp.mean((pred - rewards) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Compute global gradient norm
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
        ))

        # Gradient norm should be finite and reasonable
        assert jnp.isfinite(grad_norm), f"Gradient norm is not finite: {grad_norm}"
        assert grad_norm > 1e-8, f"Gradient norm too small (vanishing): {grad_norm}"
        assert grad_norm < 1e6, f"Gradient norm too large (exploding): {grad_norm}"

    def test_training_stability_over_many_steps(self, key):
        """Training should remain stable over many steps"""
        import optax
        from flax.training.train_state import TrainState

        model = RewardModel(num_agents=3, action_dim=4)
        batch_size = 32

        key, obs_key, action_key, reward_key = jax.random.split(key, 4)
        obs = jax.random.normal(obs_key, (batch_size, 3, 15, 15, 4))
        actions = jax.random.randint(action_key, (batch_size, 3), 0, 4)
        rewards = jax.random.normal(reward_key, (batch_size, 3))

        params = model.init(key, obs, actions)
        tx = optax.adam(learning_rate=0.001)
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @jax.jit
        def train_step(train_state, obs, actions, rewards):
            def loss_fn(params):
                pred = model.apply(params, obs, actions)
                return jnp.mean((pred - rewards) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        # Run for 500 steps
        losses = []
        for step in range(500):
            train_state, loss = train_step(train_state, obs, actions, rewards)
            losses.append(float(loss))

            # Periodic checks
            if step % 100 == 0:
                assert jnp.isfinite(loss), f"Loss NaN/Inf at step {step}"

        # Verify stability
        losses_arr = jnp.array(losses)

        # No NaN/Inf
        assert jnp.all(jnp.isfinite(losses_arr)), "Losses contain NaN/Inf"

        # Loss decreased
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

        # No explosion (max should be reasonable, near initial)
        max_loss = float(jnp.max(losses_arr))
        assert max_loss < losses[0] * 10, \
            f"Loss exploded: max={max_loss:.4f}, initial={losses[0]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
