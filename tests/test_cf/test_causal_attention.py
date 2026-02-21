"""
Tests for Causal Attention Module (CF-IMPL-008)

Test criteria:
- Attention weights sum to 1
- Causal mask correctly applied
- Output shape compatible with RewardModel
- No NaN/Inf
"""

import pytest
import sys
sys.path.insert(0, 'socialjax')

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from socialjax.algorithms.cf.causal_attention import (
    create_causal_mask,
    create_attention_mask_for_causal,
    CausalMultiHeadAttention,
    AgentFeatureExtractor,
    TransformerBlock,
    CausalRewardModel,
    compute_causal_reward_model_loss,
    create_causal_reward_model_train_state,
    verify_attention_weights,
    get_attention_statistics,
)


class TestCreateCausalMask:
    """Test causal mask generation"""

    def test_causal_mask_shape(self):
        """Mask should have shape [num_agents, num_agents]"""
        for num_agents in [2, 3, 4, 5, 7]:
            mask = create_causal_mask(num_agents)
            assert mask.shape == (num_agents, num_agents)

    def test_causal_mask_lower_triangular(self):
        """Mask should be lower triangular (can attend to self and earlier)"""
        for num_agents in [3, 5]:
            mask = create_causal_mask(num_agents)
            # Check it's lower triangular
            for i in range(num_agents):
                for j in range(num_agents):
                    if j <= i:
                        assert mask[i, j] == 1, f"mask[{i},{j}] should be 1"
                    else:
                        assert mask[i, j] == 0, f"mask[{i},{j}] should be 0"

    def test_causal_mask_values(self):
        """Mask values should be 0 or 1"""
        mask = create_causal_mask(4)
        assert jnp.all((mask == 0) | (mask == 1))


class TestCreateAttentionMaskForCausal:
    """Test attention mask generation for scaled dot-product attention"""

    def test_attention_mask_shape(self):
        """Mask should have shape [1, 1, num_agents, num_agents]"""
        for num_agents in [3, 5]:
            mask = create_attention_mask_for_causal(num_agents, causal=True)
            assert mask.shape == (1, 1, num_agents, num_agents)

    def test_attention_mask_causal_values(self):
        """Causal mask should have 0 for valid, -inf for masked"""
        mask = create_attention_mask_for_causal(3, causal=True)
        # Lower triangular should be 0
        assert mask[0, 0, 0, 0] == 0  # Can attend to self
        assert mask[0, 0, 1, 0] == 0  # Can attend to earlier
        assert mask[0, 0, 1, 1] == 0  # Can attend to self
        # Upper triangular should be -inf (or large negative)
        assert mask[0, 0, 0, 1] < -1e8  # Cannot attend to future
        assert mask[0, 0, 0, 2] < -1e8  # Cannot attend to future
        assert mask[0, 0, 1, 2] < -1e8  # Cannot attend to future

    def test_attention_mask_non_causal(self):
        """Non-causal mask should allow full attention"""
        mask = create_attention_mask_for_causal(3, causal=False)
        assert jnp.allclose(mask, 0.0)


class TestCausalMultiHeadAttention:
    """Test multi-head attention with causal masking"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def sample_input(self):
        # [batch=2, num_agents=3, embed_dim=32]
        return jax.random.normal(jax.random.PRNGKey(0), (2, 3, 32))

    def test_output_shape(self, rng, sample_input):
        """Output should have same shape as input"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)
        params = model.init(rng, sample_input)
        output, _ = model.apply(params, sample_input)
        assert output.shape == sample_input.shape

    def test_attention_weights_shape(self, rng, sample_input):
        """Attention weights should be [batch, num_heads, num_agents, num_agents]"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        batch, num_agents, _ = sample_input.shape
        assert attn_weights.shape == (batch, model.num_heads, num_agents, num_agents)

    def test_attention_weights_sum_to_one(self, rng, sample_input):
        """Attention weights should sum to 1 along the key dimension"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        # Sum along last axis (keys)
        weight_sums = jnp.sum(attn_weights, axis=-1)
        assert jnp.allclose(weight_sums, 1.0, atol=1e-5)

    def test_attention_weights_causal_masking(self, rng, sample_input):
        """With causal masking, attention to future should be 0"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8, causal=True)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        # For causal attention, agent i should not attend to agent j > i
        batch, num_heads, num_agents, _ = attn_weights.shape
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Attention from agent i to agent j should be ~0
                assert jnp.allclose(attn_weights[:, :, i, j], 0.0, atol=1e-6)

    def test_attention_weights_no_nan_inf(self, rng, sample_input):
        """Attention weights should not contain NaN or Inf"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        assert not jnp.any(jnp.isnan(attn_weights))
        assert not jnp.any(jnp.isinf(attn_weights))

    def test_causal_vs_non_causal(self, rng, sample_input):
        """Causal and non-causal should have different attention patterns"""
        model_causal = CausalMultiHeadAttention(num_heads=4, head_dim=8, causal=True)
        model_non_causal = CausalMultiHeadAttention(num_heads=4, head_dim=8, causal=False)

        params_c = model_causal.init(rng, sample_input)
        params_nc = model_non_causal.init(rng, sample_input)

        _, attn_c = model_causal.apply(params_c, sample_input)
        _, attn_nc = model_non_causal.apply(params_nc, sample_input)

        # Non-causal should have non-zero attention to future
        # Causal should have zero attention to future
        assert jnp.any(attn_nc[:, :, 0, 1] > 0)  # Non-causal can attend to future
        assert jnp.allclose(attn_c[:, :, 0, 1], 0.0, atol=1e-6)  # Causal cannot

    def test_different_batch_sizes(self, rng):
        """Should work with different batch sizes"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)

        for batch_size in [1, 4, 8, 16]:
            x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3, 32))
            params = model.init(rng, x)
            output, attn = model.apply(params, x)
            assert output.shape == x.shape
            assert attn.shape == (batch_size, 4, 3, 3)

    def test_different_num_agents(self, rng):
        """Should work with different numbers of agents"""
        model = CausalMultiHeadAttention(num_heads=4, head_dim=8)

        for num_agents in [2, 3, 5, 7]:
            x = jax.random.normal(jax.random.PRNGKey(0), (2, num_agents, 32))
            params = model.init(rng, x)
            output, attn = model.apply(params, x)
            assert output.shape == x.shape
            assert attn.shape == (2, 4, num_agents, num_agents)


class TestAgentFeatureExtractor:
    """Test agent feature extraction"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def sample_inputs(self):
        # obs: [batch=2, num_agents=3, H=8, W=8, C=3]
        obs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8, 8, 3))
        # actions: [batch=2, num_agents=3]
        actions = jnp.array([[0, 1, 2], [1, 0, 2]])
        return obs, actions

    def test_output_shape(self, rng, sample_inputs):
        """Output should be [batch, num_agents, hidden_dim]"""
        obs, actions = sample_inputs
        model = AgentFeatureExtractor(
            hidden_dim=64,
            action_dim=4,
        )
        params = model.init(rng, obs, actions)
        output = model.apply(params, obs, actions)

        assert output.shape == (2, 3, 64)

    def test_different_hidden_dims(self, rng, sample_inputs):
        """Should work with different hidden dimensions"""
        obs, actions = sample_inputs

        for hidden_dim in [32, 64, 128]:
            model = AgentFeatureExtractor(hidden_dim=hidden_dim, action_dim=4)
            params = model.init(rng, obs, actions)
            output = model.apply(params, obs, actions)
            assert output.shape == (2, 3, hidden_dim)

    def test_no_nan_inf(self, rng, sample_inputs):
        """Output should not contain NaN or Inf"""
        obs, actions = sample_inputs
        model = AgentFeatureExtractor(hidden_dim=64, action_dim=4)
        params = model.init(rng, obs, actions)
        output = model.apply(params, obs, actions)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))


class TestTransformerBlock:
    """Test transformer block with self-attention and feed-forward"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def sample_input(self):
        return jax.random.normal(jax.random.PRNGKey(0), (2, 3, 64))

    def test_output_shape(self, rng, sample_input):
        """Output should have same shape as input"""
        model = TransformerBlock(num_heads=4, head_dim=16, mlp_dim=128)
        params = model.init(rng, sample_input)
        output, _ = model.apply(params, sample_input)
        assert output.shape == sample_input.shape

    def test_residual_connection(self, rng, sample_input):
        """Output should be different from input (residual connection adds to input)"""
        model = TransformerBlock(num_heads=4, head_dim=16, mlp_dim=128)
        params = model.init(rng, sample_input)
        output, _ = model.apply(params, sample_input)

        # Output should be different from input
        assert not jnp.allclose(output, sample_input)

        # But they should be correlated (residual connection)
        correlation = jnp.corrcoef(sample_input.flatten(), output.flatten())[0, 1]
        # Just check they're not identical
        assert correlation < 1.0 or jnp.isnan(correlation)

    def test_attention_weights_shape(self, rng, sample_input):
        """Should return attention weights"""
        model = TransformerBlock(num_heads=4, head_dim=16, mlp_dim=128)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        batch, num_agents, _ = sample_input.shape
        assert attn_weights.shape == (batch, model.num_heads, num_agents, num_agents)

    def test_causal_masking_in_block(self, rng, sample_input):
        """Causal masking should work in transformer block"""
        model = TransformerBlock(num_heads=4, head_dim=16, mlp_dim=128, causal=True)
        params = model.init(rng, sample_input)
        _, attn_weights = model.apply(params, sample_input)

        # Check causal mask is applied
        batch, num_heads, num_agents, _ = attn_weights.shape
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                assert jnp.allclose(attn_weights[:, :, i, j], 0.0, atol=1e-6)


class TestCausalRewardModel:
    """Test full CausalRewardModel"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def sample_inputs(self):
        # obs: [batch=2, num_agents=3, H=8, W=8, C=3]
        obs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8, 8, 3))
        # actions: [batch=2, num_agents=3]
        actions = jnp.array([[0, 1, 2], [1, 0, 2]])
        return obs, actions

    def test_output_shape(self, rng, sample_inputs):
        """Output should be [batch, num_agents] - compatible with RewardModel"""
        obs, actions = sample_inputs
        model = CausalRewardModel(
            num_agents=3,
            action_dim=4,
            hidden_dim=64,
        )
        params = model.init(rng, obs, actions)
        rewards, attn = model.apply(params, obs, actions)

        assert rewards.shape == (2, 3)

    def test_attention_weights_shape(self, rng, sample_inputs):
        """Should return attention weights"""
        obs, actions = sample_inputs
        model = CausalRewardModel(
            num_agents=3,
            action_dim=4,
            hidden_dim=64,
            num_heads=4,
        )
        params = model.init(rng, obs, actions)
        _, attn = model.apply(params, obs, actions)

        assert attn.shape == (2, 4, 3, 3)

    def test_output_no_nan_inf(self, rng, sample_inputs):
        """Output should not contain NaN or Inf"""
        obs, actions = sample_inputs
        model = CausalRewardModel(num_agents=3, action_dim=4)
        params = model.init(rng, obs, actions)
        rewards, _ = model.apply(params, obs, actions)

        assert not jnp.any(jnp.isnan(rewards))
        assert not jnp.any(jnp.isinf(rewards))

    def test_different_batch_sizes(self, rng):
        """Should work with different batch sizes"""
        model = CausalRewardModel(num_agents=3, action_dim=4)

        for batch_size in [1, 4, 8, 16]:
            obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3, 8, 8, 3))
            actions = jnp.zeros((batch_size, 3), dtype=jnp.int32)
            params = model.init(rng, obs, actions)
            rewards, _ = model.apply(params, obs, actions)
            assert rewards.shape == (batch_size, 3)

    def test_different_num_agents(self, rng):
        """Should work with different numbers of agents"""
        for num_agents in [2, 3, 5, 7]:
            model = CausalRewardModel(num_agents=num_agents, action_dim=4)
            obs = jax.random.normal(jax.random.PRNGKey(0), (2, num_agents, 8, 8, 3))
            actions = jnp.zeros((2, num_agents), dtype=jnp.int32)
            params = model.init(rng, obs, actions)
            rewards, _ = model.apply(params, obs, actions)
            assert rewards.shape == (2, num_agents)

    def test_different_action_dims(self, rng):
        """Should work with different action dimensions"""
        for action_dim in [2, 4, 8]:
            model = CausalRewardModel(num_agents=3, action_dim=action_dim)
            obs = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8, 8, 3))
            actions = jnp.zeros((2, 3), dtype=jnp.int32)
            params = model.init(rng, obs, actions)
            rewards, _ = model.apply(params, obs, actions)
            assert rewards.shape == (2, 3)

    def test_causal_vs_non_causal(self, rng, sample_inputs):
        """Causal and non-causal models should produce different outputs"""
        obs, actions = sample_inputs

        model_causal = CausalRewardModel(num_agents=3, action_dim=4, causal=True)
        model_non_causal = CausalRewardModel(num_agents=3, action_dim=4, causal=False)

        params_c = model_causal.init(rng, obs, actions)
        params_nc = model_non_causal.init(rng, obs, actions)

        rewards_c, _ = model_causal.apply(params_c, obs, actions)
        rewards_nc, _ = model_non_causal.apply(params_nc, obs, actions)

        # Outputs should be different
        assert not jnp.allclose(rewards_c, rewards_nc)

    def test_attention_weights_sum_to_one(self, rng, sample_inputs):
        """Attention weights should sum to 1"""
        obs, actions = sample_inputs
        model = CausalRewardModel(num_agents=3, action_dim=4)
        params = model.init(rng, obs, actions)
        _, attn = model.apply(params, obs, actions)

        weight_sums = jnp.sum(attn, axis=-1)
        assert jnp.allclose(weight_sums, 1.0, atol=1e-5)

    def test_attention_causal_masking(self, rng, sample_inputs):
        """Causal attention should not attend to future agents"""
        obs, actions = sample_inputs
        model = CausalRewardModel(num_agents=3, action_dim=4, causal=True)
        params = model.init(rng, obs, actions)
        _, attn = model.apply(params, obs, actions)

        # Check upper triangular is zero
        for i in range(3):
            for j in range(i + 1, 3):
                assert jnp.allclose(attn[:, :, i, j], 0.0, atol=1e-6)


class TestComputeCausalRewardModelLoss:
    """Test loss computation"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def model_and_data(self, rng):
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.array([[0, 1, 2], [1, 0, 2]])
        rewards = jax.random.normal(rng, (2, 3))
        params = model.init(rng, obs, actions)
        return model, params, obs, actions, rewards

    def test_loss_is_scalar(self, model_and_data):
        """Loss should be a scalar"""
        model, params, obs, actions, rewards = model_and_data
        loss, _, _ = compute_causal_reward_model_loss(params, model, obs, actions, rewards)
        assert loss.shape == ()
        assert loss.ndim == 0

    def test_loss_non_negative(self, model_and_data):
        """MSE loss should be non-negative"""
        model, params, obs, actions, rewards = model_and_data
        loss, _, _ = compute_causal_reward_model_loss(params, model, obs, actions, rewards)
        assert loss >= 0

    def test_loss_zero_for_perfect_prediction(self, rng):
        """Loss should be near zero when prediction matches target"""
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.zeros((2, 3), dtype=jnp.int32)
        params = model.init(rng, obs, actions)

        # Get model prediction
        predicted, _ = model.apply(params, obs, actions)

        # Loss with prediction as target should be near zero
        loss, _, _ = compute_causal_reward_model_loss(params, model, obs, actions, predicted)
        assert loss < 1e-6

    def test_loss_differentiable(self, model_and_data):
        """Loss should be differentiable"""
        model, params, obs, actions, rewards = model_and_data

        def loss_fn(p):
            loss, _, _ = compute_causal_reward_model_loss(p, model, obs, actions, rewards)
            return loss

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # Check gradients exist and are not all zero
        grad_norm = sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        assert grad_norm > 0


class TestCreateCausalRewardModelTrainState:
    """Test training state creation"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def test_creates_train_state(self, rng):
        """Should create a valid TrainState"""
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.zeros((2, 3), dtype=jnp.int32)

        train_state, _ = create_causal_reward_model_train_state(model, rng, obs, actions)

        assert isinstance(train_state, TrainState)
        assert train_state.params is not None

    def test_rng_split(self, rng):
        """Should return a new random key"""
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.zeros((2, 3), dtype=jnp.int32)

        _, new_rng = create_causal_reward_model_train_state(model, rng, obs, actions)

        assert new_rng is not rng
        assert new_rng.shape == rng.shape


class TestVerifyAttentionWeights:
    """Test attention weight verification"""

    def test_valid_weights(self):
        """Valid attention weights should pass"""
        attn = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (2, 4, 3, 3)), axis=-1)
        is_valid, message = verify_attention_weights(attn)
        assert is_valid
        assert "passed" in message.lower()

    def test_nan_detection(self):
        """Should detect NaN values"""
        attn = jnp.ones((2, 4, 3, 3)) / 3
        attn = attn.at[0, 0, 0, 0].set(jnp.nan)
        is_valid, message = verify_attention_weights(attn)
        assert not is_valid
        assert "nan" in message.lower()

    def test_inf_detection(self):
        """Should detect Inf values"""
        attn = jnp.ones((2, 4, 3, 3)) / 3
        attn = attn.at[0, 0, 0, 0].set(jnp.inf)
        is_valid, message = verify_attention_weights(attn)
        assert not is_valid
        assert "inf" in message.lower()

    def test_sum_not_one_detection(self):
        """Should detect when weights don't sum to 1"""
        attn = jnp.ones((2, 4, 3, 3)) * 0.5  # Sum = 1.5, not 1.0
        is_valid, message = verify_attention_weights(attn)
        assert not is_valid
        assert "sum" in message.lower()


class TestGetAttentionStatistics:
    """Test attention statistics computation"""

    def test_returns_dict(self):
        """Should return a dictionary of statistics"""
        attn = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (2, 4, 3, 3)), axis=-1)
        stats = get_attention_statistics(attn)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "entropy" in stats

    def test_mean_approximately_correct(self):
        """Mean should be approximately 1/num_agents for uniform-ish attention"""
        num_agents = 3
        attn = jnp.ones((1, 1, num_agents, num_agents)) / num_agents
        stats = get_attention_statistics(attn)
        assert jnp.isclose(stats["mean"], 1.0 / num_agents, atol=0.01)

    def test_min_max_in_valid_range(self):
        """Min and max should be in [0, 1]"""
        attn = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (2, 4, 3, 3)), axis=-1)
        stats = get_attention_statistics(attn)

        assert 0 <= stats["min"] <= 1
        assert 0 <= stats["max"] <= 1
        assert stats["min"] <= stats["max"]


class TestIntegration:
    """Integration tests for CausalRewardModel"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def test_full_forward_pass(self, rng):
        """Test complete forward pass"""
        # Create model
        model = CausalRewardModel(
            num_agents=3,
            action_dim=4,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        # Create sample data
        obs = jax.random.normal(rng, (4, 3, 8, 8, 3))
        actions = jnp.array([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 0, 0]])

        # Initialize and forward pass
        params = model.init(rng, obs, actions)
        rewards, attn = model.apply(params, obs, actions)

        # Verify outputs
        assert rewards.shape == (4, 3)
        assert attn.shape == (4, 4, 3, 3)
        assert not jnp.any(jnp.isnan(rewards))
        assert not jnp.any(jnp.isinf(rewards))

    def test_training_step(self, rng):
        """Test a single training step"""
        # Create model and training state
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (4, 3, 8, 8, 3))
        actions = jnp.zeros((4, 3), dtype=jnp.int32)
        rewards = jax.random.normal(rng, (4, 3))

        train_state, _ = create_causal_reward_model_train_state(model, rng, obs, actions)

        # Compute loss and gradients
        def loss_fn(params):
            loss, _, _ = compute_causal_reward_model_loss(
                params, model, obs, actions, rewards
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)

        # Update parameters
        new_train_state = train_state.apply_gradients(grads=grads)

        # Verify
        assert loss >= 0
        assert new_train_state.params is not None

    def test_jit_compilation(self, rng):
        """Test that forward pass can be JIT compiled"""
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.zeros((2, 3), dtype=jnp.int32)
        params = model.init(rng, obs, actions)

        # JIT compile forward pass
        @jax.jit
        def forward(params, obs, actions):
            return model.apply(params, obs, actions, deterministic=True)

        # Run compiled function
        rewards, attn = forward(params, obs, actions)

        assert rewards.shape == (2, 3)
        assert attn.shape == (2, 4, 3, 3)

    def test_different_observation_sizes(self, rng):
        """Should work with different observation sizes"""
        model = CausalRewardModel(num_agents=3, action_dim=4)

        for h, w in [(8, 8), (16, 16), (10, 10)]:
            obs = jax.random.normal(rng, (2, 3, h, w, 3))
            actions = jnp.zeros((2, 3), dtype=jnp.int32)
            params = model.init(rng, obs, actions)
            rewards, _ = model.apply(params, obs, actions)
            assert rewards.shape == (2, 3)


class TestEdgeCases:
    """Edge case tests"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def test_single_agent(self, rng):
        """Should work with single agent"""
        model = CausalRewardModel(num_agents=1, action_dim=4)
        obs = jax.random.normal(rng, (2, 1, 8, 8, 3))
        actions = jnp.zeros((2, 1), dtype=jnp.int32)
        params = model.init(rng, obs, actions)
        rewards, attn = model.apply(params, obs, actions)

        assert rewards.shape == (2, 1)
        assert attn.shape == (2, 4, 1, 1)

    def test_large_num_agents(self, rng):
        """Should work with large number of agents"""
        num_agents = 10
        model = CausalRewardModel(num_agents=num_agents, action_dim=4)
        obs = jax.random.normal(rng, (2, num_agents, 8, 8, 3))
        actions = jnp.zeros((2, num_agents), dtype=jnp.int32)
        params = model.init(rng, obs, actions)
        rewards, attn = model.apply(params, obs, actions)

        assert rewards.shape == (2, num_agents)
        assert attn.shape == (2, 4, num_agents, num_agents)

    def test_batch_size_one(self, rng):
        """Should work with batch size 1"""
        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (1, 3, 8, 8, 3))
        actions = jnp.zeros((1, 3), dtype=jnp.int32)
        params = model.init(rng, obs, actions)
        rewards, _ = model.apply(params, obs, actions)

        assert rewards.shape == (1, 3)

    def test_extreme_values(self, rng):
        """Should handle extreme input values"""
        model = CausalRewardModel(num_agents=3, action_dim=4)

        # Large positive values
        obs_large = jnp.ones((2, 3, 8, 8, 3)) * 100
        actions = jnp.zeros((2, 3), dtype=jnp.int32)
        params = model.init(rng, obs_large, actions)
        rewards, _ = model.apply(params, obs_large, actions)
        assert not jnp.any(jnp.isnan(rewards))

        # Small values
        obs_small = jnp.ones((2, 3, 8, 8, 3)) * 1e-6
        rewards, _ = model.apply(params, obs_small, actions)
        assert not jnp.any(jnp.isnan(rewards))


class TestCompatibilityWithRewardModel:
    """Test that CausalRewardModel is compatible with RewardModel API"""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def test_output_shape_matches(self, rng):
        """Output shape should match RewardModel: [batch, num_agents]"""
        from socialjax.algorithms.cf.generative_model import RewardModel

        num_agents = 3
        action_dim = 4
        batch_size = 2

        obs = jax.random.normal(rng, (batch_size, num_agents, 8, 8, 3))
        actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)

        # Standard RewardModel
        standard_model = RewardModel(num_agents=num_agents, action_dim=action_dim)
        standard_params = standard_model.init(rng, obs, actions)
        standard_output = standard_model.apply(standard_params, obs, actions)

        # CausalRewardModel
        causal_model = CausalRewardModel(num_agents=num_agents, action_dim=action_dim)
        causal_params = causal_model.init(rng, obs, actions)
        causal_output, _ = causal_model.apply(causal_params, obs, actions)

        # Outputs should have same shape
        assert standard_output.shape == causal_output.shape

    def test_can_use_same_loss_function(self, rng):
        """Should be able to use with generative_model_loss function"""
        from socialjax.algorithms.cf.generative_model import generative_model_loss

        model = CausalRewardModel(num_agents=3, action_dim=4)
        obs = jax.random.normal(rng, (2, 3, 8, 8, 3))
        actions = jnp.zeros((2, 3), dtype=jnp.int32)
        rewards = jax.random.normal(rng, (2, 3))

        params = model.init(rng, obs, actions)
        predicted, _ = model.apply(params, obs, actions)

        # Should work with the same loss function
        loss = generative_model_loss(predicted, rewards)

        assert loss >= 0
        assert loss.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
