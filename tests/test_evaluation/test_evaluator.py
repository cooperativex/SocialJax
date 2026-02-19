"""Unit tests for socialjax.evaluation.evaluator module."""

import pytest
import numpy as np
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "socialjax"))

from socialjax.evaluation.evaluator import (
    EvaluatorConfig,
    Evaluator,
    save_evaluation_results,
    load_evaluation_results,
    print_evaluation_summary,
)
from socialjax.evaluation.metrics import EpisodeMetrics, EvaluationMetrics


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, num_agents=5, max_steps=100):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self._step_count = 0

    def reset(self, key=None):
        """Reset the environment."""
        self._step_count = 0
        obs = {agent: np.zeros((10, 10, 3)) for agent in self.agents}
        state = {"step": 0, "done": False}
        return obs, state

    def step(self, key_or_state, state_or_actions, actions=None):
        """Step the environment."""
        # Handle different calling conventions
        if actions is None:
            # step(state, actions)
            state = key_or_state
            actions = state_or_actions
        else:
            # step(key, state, actions)
            state = state_or_actions

        self._step_count += 1

        obs = {agent: np.zeros((10, 10, 3)) for agent in self.agents}
        rewards = {agent: 1.0 for agent in self.agents}
        dones = {"__all__": self._step_count >= self.max_steps}
        info = {}

        state = {"step": self._step_count, "done": dones["__all__"]}
        return obs, state, rewards, dones, info

    def action_space(self):
        """Return action space."""
        space = Mock()
        space.n = 8
        return space

    def render(self, state):
        """Render the environment."""
        return np.zeros((64, 64, 3), dtype=np.uint8)


class MockAlgorithm:
    """Mock algorithm for testing."""

    def __init__(self, action_dim=8):
        self.action_dim = action_dim

    def compute_action(self, state, obs, rng, deterministic=True):
        """Compute an action."""
        # Return a random action
        action = np.random.randint(0, self.action_dim)
        return action, {}


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig dataclass."""

    def test_init_defaults(self):
        """Test EvaluatorConfig initialization with defaults."""
        config = EvaluatorConfig()
        assert config.num_episodes == 10
        assert config.deterministic is True
        assert config.max_steps_per_episode is None
        assert config.seed == 42
        assert config.capture_frames is False
        assert config.verbose == 1

    def test_init_with_values(self):
        """Test EvaluatorConfig initialization with custom values."""
        config = EvaluatorConfig(
            num_episodes=50,
            deterministic=False,
            max_steps_per_episode=500,
            seed=123,
            capture_frames=True,
            verbose=2,
        )
        assert config.num_episodes == 50
        assert config.deterministic is False
        assert config.max_steps_per_episode == 500
        assert config.seed == 123
        assert config.capture_frames is True
        assert config.verbose == 2


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_init(self):
        """Test Evaluator initialization."""
        env = MockEnvironment()
        algo = MockAlgorithm()
        evaluator = Evaluator(env, algo)

        assert evaluator.env == env
        assert evaluator.algorithm == algo
        assert evaluator.config.num_episodes == 10

    def test_init_with_config(self):
        """Test Evaluator initialization with config."""
        env = MockEnvironment()
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=25, seed=100)
        evaluator = Evaluator(env, algo, config)

        assert evaluator.config.num_episodes == 25
        assert evaluator.config.seed == 100

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        env = MockEnvironment(max_steps=10)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=3, verbose=0)
        evaluator = Evaluator(env, algo, config)

        metrics = evaluator.evaluate()

        assert metrics.num_episodes == 3
        assert metrics.mean_return > 0
        assert metrics.mean_length == 10

    def test_evaluate_with_override_params(self):
        """Test evaluation with parameter overrides."""
        env = MockEnvironment(max_steps=5)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=10, verbose=0)
        evaluator = Evaluator(env, algo, config)

        # Override num_episodes
        metrics = evaluator.evaluate(num_episodes=2)
        assert metrics.num_episodes == 2

    def test_evaluate_with_max_steps(self):
        """Test evaluation with max_steps_per_episode."""
        env = MockEnvironment(max_steps=100)
        algo = MockAlgorithm()
        config = EvaluatorConfig(
            num_episodes=2,
            max_steps_per_episode=5,
            verbose=0,
        )
        evaluator = Evaluator(env, algo, config)

        metrics = evaluator.evaluate()
        assert metrics.mean_length == 5

    def test_evaluate_deterministic(self):
        """Test deterministic evaluation."""
        env = MockEnvironment(max_steps=5)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=2, seed=42, verbose=0)
        evaluator = Evaluator(env, algo, config)

        # Run twice with same seed - should get same results
        metrics1 = evaluator.evaluate(seed=42)
        metrics2 = evaluator.evaluate(seed=42)

        # Episodes should have same lengths
        assert metrics1.episode_lengths == metrics2.episode_lengths

    def test_evaluate_with_frames(self):
        """Test evaluation with frame capture."""
        env = MockEnvironment(max_steps=5)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=2, verbose=0)
        evaluator = Evaluator(env, algo, config)

        metrics, frames = evaluator.evaluate_with_frames(
            capture_frequency=2,
        )

        assert metrics.num_episodes == 2
        # Should capture frames at step 0, 2, 4, and final
        assert len(frames) > 0

    def test_run_episode(self):
        """Test single episode execution."""
        env = MockEnvironment(max_steps=10)
        algo = MockAlgorithm()
        config = EvaluatorConfig(verbose=0)
        evaluator = Evaluator(env, algo, config)

        rng = None
        metrics = evaluator._run_episode(
            algorithm_state=None,
            rng=rng,
            deterministic=True,
        )

        assert metrics.episode_length == 10
        assert metrics.episode_return > 0
        assert metrics.total_actions == 10 * env.num_agents

    def test_get_actions(self):
        """Test action computation."""
        env = MockEnvironment()
        algo = MockAlgorithm()
        config = EvaluatorConfig(verbose=0)
        evaluator = Evaluator(env, algo, config)

        obs = {agent: np.zeros((10, 10, 3)) for agent in env.agents}
        actions = evaluator._get_actions(
            obs=obs,
            algorithm_state=None,
            rng=None,
            deterministic=True,
        )

        assert len(actions) == env.num_agents
        for action in actions:
            assert isinstance(action, (int, np.integer)) or hasattr(action, '__array__')

    def test_capture_frame(self):
        """Test frame capture."""
        env = MockEnvironment()
        algo = MockAlgorithm()
        config = EvaluatorConfig(verbose=0)
        evaluator = Evaluator(env, algo, config)

        state = {"step": 5}
        frame = evaluator._capture_frame(state)

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)

    def test_evaluate_no_render(self):
        """Test evaluation when render is not available."""
        env_no_render = MockEnvironment()
        env_no_render.render = None  # Remove render method

        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=2, verbose=0)
        evaluator = Evaluator(env_no_render, algo, config)

        metrics = evaluator.evaluate()
        assert metrics.num_episodes == 2


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    def test_save_basic(self):
        """Test basic results saving."""
        metrics = EvaluationMetrics(
            num_episodes=10,
            mean_return=5.0,
            std_return=1.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result_path = save_evaluation_results(metrics, output_path)
            assert Path(result_path).exists()

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data["num_episodes"] == 10
            assert data["mean_return"] == 5.0
        finally:
            Path(output_path).unlink()

    def test_save_with_additional_info(self):
        """Test saving with additional info."""
        metrics = EvaluationMetrics(num_episodes=5)
        additional = {"algorithm": "ippo", "env": "clean_up"}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_evaluation_results(metrics, output_path, additional)

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data["additional_info"]["algorithm"] == "ippo"
        finally:
            Path(output_path).unlink()

    def test_save_creates_directories(self):
        """Test that save creates parent directories."""
        metrics = EvaluationMetrics(num_episodes=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "results.json"
            save_evaluation_results(metrics, str(output_path))
            assert output_path.exists()


class TestLoadEvaluationResults:
    """Tests for load_evaluation_results function."""

    def test_load_basic(self):
        """Test basic results loading."""
        data = {
            "num_episodes": 10,
            "mean_return": 5.0,
            "episode_returns": [1.0, 2.0, 3.0],
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            json.dump(data, f)
            output_path = f.name

        try:
            loaded = load_evaluation_results(output_path)
            assert loaded["num_episodes"] == 10
            assert loaded["mean_return"] == 5.0
        finally:
            Path(output_path).unlink()


class TestPrintEvaluationSummary:
    """Tests for print_evaluation_summary function."""

    def test_print_basic(self, capsys):
        """Test basic summary printing."""
        metrics = EvaluationMetrics(
            num_episodes=10,
            mean_return=5.0,
            std_return=1.0,
            cooperation_rate=0.75,
        )

        print_evaluation_summary(metrics, verbose=1)
        captured = capsys.readouterr()

        assert "Episodes:" in captured.out
        assert "Mean Return:" in captured.out
        assert "5.0000" in captured.out

    def test_print_silent(self, capsys):
        """Test silent mode (verbose=0)."""
        metrics = EvaluationMetrics(num_episodes=5)

        print_evaluation_summary(metrics, verbose=0)
        captured = capsys.readouterr()

        assert "Episodes:" not in captured.out

    def test_print_verbose(self, capsys):
        """Test verbose mode."""
        metrics = EvaluationMetrics(
            num_episodes=5,
            mean_agent_returns={"agent_0": 2.5, "agent_1": 2.5},
            std_agent_returns={"agent_0": 0.5, "agent_1": 0.5},
        )

        print_evaluation_summary(metrics, verbose=2)
        captured = capsys.readouterr()

        assert "Per-Agent Returns" in captured.out


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create environment and algorithm
        env = MockEnvironment(num_agents=3, max_steps=20)
        algo = MockAlgorithm()

        # Create evaluator
        config = EvaluatorConfig(
            num_episodes=5,
            seed=42,
            verbose=0,
        )
        evaluator = Evaluator(env, algo, config)

        # Run evaluation
        metrics = evaluator.evaluate()

        # Verify results
        assert metrics.num_episodes == 5
        assert metrics.mean_return > 0
        assert metrics.mean_length == 20

        # Save results
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_evaluation_results(metrics, output_path)
            loaded = load_evaluation_results(output_path)
            assert loaded["num_episodes"] == 5
        finally:
            Path(output_path).unlink()

    def test_evaluation_with_frame_capture(self):
        """Test evaluation with frame capture and save."""
        env = MockEnvironment(num_agents=2, max_steps=5)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=2, verbose=0)
        evaluator = Evaluator(env, algo, config)

        # Run with frames
        metrics, frames = evaluator.evaluate_with_frames()

        assert metrics.num_episodes == 2
        assert len(frames) > 0

    def test_evaluation_verbose_output(self, capsys):
        """Test evaluation with verbose output."""
        env = MockEnvironment(num_agents=2, max_steps=3)
        algo = MockAlgorithm()
        config = EvaluatorConfig(num_episodes=2, verbose=1)
        evaluator = Evaluator(env, algo, config)

        evaluator.evaluate()

        captured = capsys.readouterr()
        assert "Running 2 evaluation episodes" in captured.out
        assert "Episode" in captured.out

    def test_evaluate_with_random_action_fallback(self):
        """Test evaluation when compute_action fails (should use random action)."""
        env = MockEnvironment(num_agents=2, max_steps=3)

        # Create algorithm that raises exception
        class FailingAlgorithm:
            def compute_action(self, *args, **kwargs):
                raise RuntimeError("Algorithm failed")

        algo = FailingAlgorithm()
        config = EvaluatorConfig(num_episodes=1, verbose=0)
        evaluator = Evaluator(env, algo, config)

        # Should still complete with random actions
        metrics = evaluator.evaluate()
        assert metrics.num_episodes == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
