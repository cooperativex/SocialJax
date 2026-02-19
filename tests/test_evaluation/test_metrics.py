"""Unit tests for socialjax.evaluation.metrics module."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "socialjax"))

from socialjax.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationMetrics,
    compute_episode_return,
    compute_agent_returns,
    compute_cooperation_rate,
    compute_gini_coefficient,
    compute_social_welfare,
    aggregate_episode_metrics,
    identify_cooperative_action,
    compute_metrics_from_episodes,
)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_init_defaults(self):
        """Test EpisodeMetrics initialization with defaults."""
        metrics = EpisodeMetrics()
        assert metrics.episode_return == 0.0
        assert metrics.episode_length == 0
        assert metrics.agent_returns == {}
        assert metrics.cooperation_actions == 0
        assert metrics.total_actions == 0

    def test_init_with_values(self):
        """Test EpisodeMetrics initialization with values."""
        metrics = EpisodeMetrics(
            episode_return=10.5,
            episode_length=100,
            agent_returns={"agent_0": 5.0, "agent_1": 5.5},
            cooperation_actions=50,
            total_actions=100,
        )
        assert metrics.episode_return == 10.5
        assert metrics.episode_length == 100
        assert metrics.agent_returns["agent_0"] == 5.0
        assert metrics.cooperation_actions == 50

    def test_to_dict(self):
        """Test EpisodeMetrics to_dict conversion."""
        metrics = EpisodeMetrics(
            episode_return=5.0,
            episode_length=50,
            agent_returns={"agent_0": 2.5},
            info={"test": "value"},
        )
        d = metrics.to_dict()
        assert d["episode_return"] == 5.0
        assert d["episode_length"] == 50
        assert d["agent_returns"]["agent_0"] == 2.5
        assert d["info"]["test"] == "value"

    def test_agent_rewards(self):
        """Test agent_rewards tracking."""
        metrics = EpisodeMetrics(
            agent_rewards={"agent_0": [1.0, 2.0, 3.0], "agent_1": [0.5, 0.5, 0.5]}
        )
        assert len(metrics.agent_rewards["agent_0"]) == 3
        assert sum(metrics.agent_rewards["agent_0"]) == 6.0


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_init_defaults(self):
        """Test EvaluationMetrics initialization with defaults."""
        metrics = EvaluationMetrics()
        assert metrics.num_episodes == 0
        assert metrics.mean_return == 0.0
        assert metrics.cooperation_rate == 0.0

    def test_init_with_values(self):
        """Test EvaluationMetrics initialization with values."""
        metrics = EvaluationMetrics(
            num_episodes=10,
            mean_return=5.0,
            std_return=1.0,
            cooperation_rate=0.75,
            gini_coefficient=0.2,
        )
        assert metrics.num_episodes == 10
        assert metrics.mean_return == 5.0
        assert metrics.cooperation_rate == 0.75
        assert metrics.gini_coefficient == 0.2

    def test_to_dict(self):
        """Test EvaluationMetrics to_dict conversion."""
        metrics = EvaluationMetrics(
            num_episodes=5,
            mean_return=3.0,
            episode_returns=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        d = metrics.to_dict()
        assert d["num_episodes"] == 5
        assert d["mean_return"] == 3.0
        assert len(d["episode_returns"]) == 5


class TestComputeEpisodeReturn:
    """Tests for compute_episode_return function."""

    def test_dict_rewards(self):
        """Test with dictionary rewards."""
        rewards = {"agent_0": 1.0, "agent_1": 2.0, "agent_2": 3.0}
        result = compute_episode_return(rewards)
        assert result == 6.0

    def test_dict_rewards_with_all(self):
        """Test with dictionary rewards containing __all__ key."""
        rewards = {"agent_0": 1.0, "agent_1": 2.0, "__all__": 100.0}
        result = compute_episode_return(rewards)
        assert result == 3.0  # Should exclude __all__

    def test_list_rewards(self):
        """Test with list rewards."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        result = compute_episode_return(rewards)
        assert result == 10.0

    def test_array_rewards(self):
        """Test with numpy array rewards."""
        rewards = np.array([1.0, 2.0, 3.0])
        result = compute_episode_return(rewards)
        assert result == 6.0

    def test_scalar_reward(self):
        """Test with scalar reward."""
        result = compute_episode_return(5.0)
        assert result == 5.0


class TestComputeAgentReturns:
    """Tests for compute_agent_returns function."""

    def test_basic(self):
        """Test basic agent return computation."""
        episode_rewards = {
            "agent_0": [1.0, 2.0, 3.0],
            "agent_1": [0.5, 0.5, 0.5],
        }
        result = compute_agent_returns(episode_rewards)
        assert result["agent_0"] == 6.0
        assert result["agent_1"] == 1.5

    def test_empty(self):
        """Test with empty rewards."""
        result = compute_agent_returns({})
        assert result == {}

    def test_negative_rewards(self):
        """Test with negative rewards."""
        episode_rewards = {
            "agent_0": [1.0, -2.0, 3.0],
        }
        result = compute_agent_returns(episode_rewards)
        assert result["agent_0"] == 2.0


class TestComputeCooperationRate:
    """Tests for compute_cooperation_rate function."""

    def test_basic(self):
        """Test basic cooperation rate computation."""
        rate = compute_cooperation_rate(50, 100)
        assert rate == 0.5

    def test_full_cooperation(self):
        """Test with full cooperation."""
        rate = compute_cooperation_rate(100, 100)
        assert rate == 1.0

    def test_no_cooperation(self):
        """Test with no cooperation."""
        rate = compute_cooperation_rate(0, 100)
        assert rate == 0.0

    def test_zero_actions(self):
        """Test with zero total actions."""
        rate = compute_cooperation_rate(0, 0)
        assert rate == 0.0


class TestComputeGiniCoefficient:
    """Tests for compute_gini_coefficient function."""

    def test_perfect_equality(self):
        """Test with perfectly equal distribution."""
        gini = compute_gini_coefficient([10.0, 10.0, 10.0, 10.0])
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_perfect_inequality(self):
        """Test with perfectly unequal distribution."""
        gini = compute_gini_coefficient([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.5  # Should be high

    def test_empty(self):
        """Test with empty list."""
        gini = compute_gini_coefficient([])
        assert gini == 0.0

    def test_single_value(self):
        """Test with single value."""
        gini = compute_gini_coefficient([10.0])
        assert gini == 0.0

    def test_negative_values(self):
        """Test with negative values (should handle gracefully)."""
        gini = compute_gini_coefficient([-1.0, 1.0, 3.0])
        assert 0.0 <= gini <= 1.0

    def test_all_zeros(self):
        """Test with all zeros."""
        gini = compute_gini_coefficient([0.0, 0.0, 0.0])
        assert gini == 0.0


class TestComputeSocialWelfare:
    """Tests for compute_social_welfare function."""

    def test_basic(self):
        """Test basic social welfare computation."""
        welfare = compute_social_welfare({"agent_0": 10.0, "agent_1": 5.0})
        assert welfare == 15.0

    def test_empty(self):
        """Test with empty dict."""
        welfare = compute_social_welfare({})
        assert welfare == 0.0

    def test_negative_values(self):
        """Test with negative values."""
        welfare = compute_social_welfare({"agent_0": 10.0, "agent_1": -5.0})
        assert welfare == 5.0


class TestAggregateEpisodeMetrics:
    """Tests for aggregate_episode_metrics function."""

    def test_basic_aggregation(self):
        """Test basic metrics aggregation."""
        episodes = [
            EpisodeMetrics(episode_return=10.0, episode_length=100),
            EpisodeMetrics(episode_return=20.0, episode_length=200),
            EpisodeMetrics(episode_return=30.0, episode_length=300),
        ]
        result = aggregate_episode_metrics(episodes)

        assert result.num_episodes == 3
        assert result.mean_return == 20.0
        assert result.min_return == 10.0
        assert result.max_return == 30.0
        assert result.mean_length == 200.0

    def test_empty_list(self):
        """Test with empty list."""
        result = aggregate_episode_metrics([])
        assert result.num_episodes == 0

    def test_single_episode(self):
        """Test with single episode."""
        episodes = [EpisodeMetrics(episode_return=5.0, episode_length=50)]
        result = aggregate_episode_metrics(episodes)

        assert result.num_episodes == 1
        assert result.mean_return == 5.0
        assert result.std_return == 0.0

    def test_agent_returns_aggregation(self):
        """Test aggregation of per-agent returns."""
        episodes = [
            EpisodeMetrics(agent_returns={"agent_0": 10.0, "agent_1": 5.0}),
            EpisodeMetrics(agent_returns={"agent_0": 20.0, "agent_1": 10.0}),
        ]
        result = aggregate_episode_metrics(episodes)

        assert result.mean_agent_returns["agent_0"] == 15.0
        assert result.mean_agent_returns["agent_1"] == 7.5

    def test_cooperation_rate_aggregation(self):
        """Test aggregation of cooperation rate."""
        episodes = [
            EpisodeMetrics(cooperation_actions=50, total_actions=100),
            EpisodeMetrics(cooperation_actions=75, total_actions=100),
        ]
        result = aggregate_episode_metrics(episodes)

        assert result.cooperation_rate == 0.625  # 125/200

    def test_episode_returns_preserved(self):
        """Test that individual episode returns are preserved."""
        episodes = [
            EpisodeMetrics(episode_return=1.0),
            EpisodeMetrics(episode_return=2.0),
        ]
        result = aggregate_episode_metrics(episodes)

        assert result.episode_returns == [1.0, 2.0]
        assert result.episode_lengths == [0, 0]


class TestIdentifyCooperativeAction:
    """Tests for identify_cooperative_action function."""

    def test_default_behavior(self):
        """Test default cooperative action identification."""
        # Action 0 is considered cooperative by default
        result = identify_cooperative_action(0, None, None)
        assert result is True

        result = identify_cooperative_action(1, None, None)
        assert result is False

    def test_with_agent_id(self):
        """Test with agent_id parameter."""
        result = identify_cooperative_action(0, None, None, "agent_0")
        assert result is True


class TestComputeMetricsFromEpisodes:
    """Tests for compute_metrics_from_episodes function."""

    def test_basic(self):
        """Test basic metric computation from episode data."""
        episode_data = [
            {"rewards": [1.0, 2.0, 3.0], "length": 3, "info": {}},
            {"rewards": [4.0, 5.0, 6.0], "length": 3, "info": {}},
        ]
        result = compute_metrics_from_episodes(episode_data)

        assert result.num_episodes == 2
        assert result.mean_return == 10.5  # (6 + 15) / 2
        assert result.mean_length == 3.0

    def test_dict_rewards(self):
        """Test with dictionary rewards."""
        episode_data = [
            {
                "rewards": {"agent_0": 1.0, "agent_1": 2.0},
                "length": 10,
                "info": {},
            }
        ]
        result = compute_metrics_from_episodes(episode_data)

        assert result.num_episodes == 1
        assert result.mean_return == 3.0

    def test_empty_list(self):
        """Test with empty episode list."""
        result = compute_metrics_from_episodes([])
        assert result.num_episodes == 0

    def test_with_actions(self):
        """Test with action data for cooperation rate."""
        episode_data = [
            {
                "rewards": [1.0],
                "actions": [[0, 1, 2], [0, 0, 1]],  # Two steps
                "length": 2,
                "info": {},
            }
        ]
        result = compute_metrics_from_episodes(episode_data, cooperation_threshold=0.5)

        assert result.num_episodes == 1
        # Cooperation rate should be computed (between 0 and 1)
        assert 0.0 <= result.cooperation_rate <= 1.0


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_full_workflow(self):
        """Test full metrics workflow."""
        # Create episode metrics
        episodes = []
        for i in range(5):
            episode = EpisodeMetrics(
                episode_return=float(i * 10),
                episode_length=100 + i * 10,
                agent_returns={"agent_0": float(i * 5), "agent_1": float(i * 5)},
                cooperation_actions=50 + i * 5,
                total_actions=100,
            )
            episodes.append(episode)

        # Aggregate
        result = aggregate_episode_metrics(episodes)

        # Verify
        assert result.num_episodes == 5
        assert result.mean_return == 20.0
        assert result.mean_length == 120.0
        assert result.social_welfare == pytest.approx(20.0, abs=0.1)

    def test_gini_with_real_returns(self):
        """Test Gini coefficient with realistic return distributions."""
        # Equal returns
        equal = [10.0, 10.0, 10.0, 10.0]
        gini_equal = compute_gini_coefficient(equal)
        assert gini_equal < 0.1

        # Unequal returns
        unequal = [0.0, 5.0, 15.0, 40.0]
        gini_unequal = compute_gini_coefficient(unequal)
        assert gini_unequal > gini_equal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
