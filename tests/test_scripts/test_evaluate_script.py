"""Unit tests for scripts/evaluate.py.

Tests cover:
- CLI argument parsing
- Checkpoint loading
- Evaluation execution
- Metrics computation
- GIF output
- Result saving
"""

import pytest
import sys
import os
import json
import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "socialjax"))

# Import the module under test
from scripts.evaluate import (
    parse_args,
    detect_algorithm_from_checkpoint,
    load_checkpoint,
    run_evaluation,
    run_evaluation_with_render,
    print_results,
    save_results_json,
    save_gif,
)


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args_checkpoint_and_env(self):
        """Test that checkpoint and env are required."""
        # Missing both args
        with patch('sys.argv', ['evaluate.py']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing env
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing checkpoint
        with patch('sys.argv', ['evaluate.py', '--env', 'coin_game']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_valid_minimal_args(self):
        """Test minimal valid argument set."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game']):
            args = parse_args()
            assert args.checkpoint == '/path/to/ckpt'
            assert args.env == 'coin_game'
            assert args.episodes == 10
            assert args.seed == 42
            assert args.deterministic is True

    def test_custom_episodes(self):
        """Test custom episodes argument."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--episodes', '50']):
            args = parse_args()
            assert args.episodes == 50

    def test_custom_seed(self):
        """Test custom seed argument."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--seed', '123']):
            args = parse_args()
            assert args.seed == 123

    def test_deterministic_flag(self):
        """Test deterministic flag (default True)."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game']):
            args = parse_args()
            assert args.deterministic is True

    def test_stochastic_flag(self):
        """Test stochastic flag overrides deterministic."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--stochastic']):
            args = parse_args()
            assert args.stochastic is True

    def test_render_options(self):
        """Test render-related arguments."""
        with patch('sys.argv', [
            'evaluate.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--render',
            '--output', 'eval.gif',
            '--fps', '15',
            '--max-frames', '250'
        ]):
            args = parse_args()
            assert args.render is True
            assert args.output == 'eval.gif'
            assert args.fps == 15
            assert args.max_frames == 250

    def test_algorithm_option(self):
        """Test algorithm argument."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--algorithm', 'ippo']):
            args = parse_args()
            assert args.algorithm == 'ippo'

    def test_num_agents_option(self):
        """Test num_agents argument."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--num-agents', '7']):
            args = parse_args()
            assert args.num_agents == 7

    def test_verbose_option(self):
        """Test verbosity level."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--verbose', '2']):
            args = parse_args()
            assert args.verbose == 2

    def test_save_results_option(self):
        """Test save-results argument."""
        with patch('sys.argv', ['evaluate.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game', '--save-results', 'results.json']):
            args = parse_args()
            assert args.save_results == 'results.json'


class TestDetectAlgorithmFromCheckpoint:
    """Tests for algorithm detection from checkpoint path."""

    def test_detect_ippo_from_path(self):
        """Test detecting IPPO from checkpoint path."""
        result = detect_algorithm_from_checkpoint('/checkpoints/ippo_coin_game/ippo_final')
        assert result == 'ippo'

    def test_detect_mappo_from_path(self):
        """Test detecting MAPPO from checkpoint path."""
        result = detect_algorithm_from_checkpoint('/checkpoints/mappo_clean_up/final')
        assert result == 'mappo'

    def test_detect_vdn_from_path(self):
        """Test detecting VDN from checkpoint path."""
        result = detect_algorithm_from_checkpoint('/models/vdn_model/checkpoint')
        assert result == 'vdn'

    def test_detect_svo_from_path(self):
        """Test detecting SVO from checkpoint path."""
        result = detect_algorithm_from_checkpoint('/saved/svo_experiment/ckpt')
        assert result == 'svo'

    def test_detect_algorithm_case_insensitive(self):
        """Test that detection is case-insensitive."""
        result = detect_algorithm_from_checkpoint('/checkpoints/IPPO_Coin_Game/final')
        assert result == 'ippo'

    def test_detect_from_trainer_info(self):
        """Test detecting algorithm from trainer_info.pkl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)
            trainer_info_path = checkpoint_path / "trainer_info.pkl"

            # Create trainer_info.pkl with algorithm name
            trainer_info = {
                "config": {
                    "algorithm": {
                        "name": "mappo"
                    }
                }
            }
            with open(trainer_info_path, "wb") as f:
                pickle.dump(trainer_info, f)

            result = detect_algorithm_from_checkpoint(str(checkpoint_path))
            assert result == 'mappo'

    def test_detect_returns_none_for_unknown(self):
        """Test that detection returns None for unknown paths."""
        result = detect_algorithm_from_checkpoint('/some/random/path')
        assert result is None


class TestLoadCheckpoint:
    """Tests for checkpoint loading."""

    @pytest.fixture
    def mock_checkpoint(self):
        """Create a mock checkpoint directory."""
        tmpdir = tempfile.mkdtemp()
        checkpoint_path = Path(tmpdir)

        # Create algorithm directory
        algo_path = checkpoint_path / "algorithm"
        algo_path.mkdir()

        # Create a minimal checkpoint
        import jax
        import pickle

        # Save algorithm params
        params = {"layer1": {"kernel": np.zeros((10, 10))}}
        with open(algo_path / "params.pkl", "wb") as f:
            pickle.dump(params, f)

        # Save trainer info
        trainer_info = {
            "config": {"algorithm": {"name": "ippo"}},
            "metrics": {
                "episode_returns": [],
                "episode_lengths": [],
                "losses": {},
                "custom_metrics": {},
            }
        }
        with open(checkpoint_path / "trainer_info.pkl", "wb") as f:
            pickle.dump(trainer_info, f)

        yield str(checkpoint_path)

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)

    def test_load_checkpoint_requires_valid_path(self):
        """Test that load_checkpoint raises error for invalid path."""
        with pytest.raises(Exception):
            load_checkpoint(
                "/nonexistent/path",
                "coin_game",
                algorithm_name="ippo",
            )

    def test_load_checkpoint_detects_algorithm(self, mock_checkpoint):
        """Test that load_checkpoint can auto-detect algorithm."""
        # Skip if JAX not available
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")

        # Create checkpoint with IPPO in path
        ippo_path = Path(mock_checkpoint) / "ippo_final"
        ippo_path.mkdir(exist_ok=True)

        # Copy checkpoint files
        import shutil
        for f in Path(mock_checkpoint).glob("*.pkl"):
            shutil.copy(f, ippo_path / f.name)
        if (Path(mock_checkpoint) / "algorithm").exists():
            shutil.copytree(Path(mock_checkpoint) / "algorithm", ippo_path / "algorithm", dirs_exist_ok=True)

        # Detection should find IPPO in path
        result = detect_algorithm_from_checkpoint(str(ippo_path))
        assert result == "ippo"


class TestRunEvaluation:
    """Tests for evaluation execution."""

    def test_run_evaluation_returns_metrics(self):
        """Test that run_evaluation returns expected metrics."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")

        # Create mock trainer and state
        mock_trainer = MagicMock()
        mock_state = MagicMock()

        expected_metrics = {
            "mean_return": 1.5,
            "std_return": 0.5,
            "mean_length": 100.0,
            "num_episodes": 10,
        }
        mock_trainer.evaluate.return_value = expected_metrics

        result = run_evaluation(
            trainer=mock_trainer,
            state=mock_state,
            num_episodes=10,
            deterministic=True,
            verbose=0,
        )

        assert "mean_return" in result
        assert "std_return" in result
        assert "num_episodes" in result
        mock_trainer.evaluate.assert_called_once()

    def test_run_evaluation_deterministic_mode(self):
        """Test that run_evaluation respects deterministic mode."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")

        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_trainer.evaluate.return_value = {"mean_return": 0.0}

        run_evaluation(
            trainer=mock_trainer,
            state=mock_state,
            num_episodes=5,
            deterministic=True,
            verbose=0,
        )

        call_kwargs = mock_trainer.evaluate.call_args[1]
        assert call_kwargs["deterministic"] is True

    def test_run_evaluation_stochastic_mode(self):
        """Test that run_evaluation respects stochastic mode."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")

        mock_trainer = MagicMock()
        mock_state = MagicMock()
        mock_trainer.evaluate.return_value = {"mean_return": 0.0}

        run_evaluation(
            trainer=mock_trainer,
            state=mock_state,
            num_episodes=5,
            deterministic=False,
            verbose=0,
        )

        call_kwargs = mock_trainer.evaluate.call_args[1]
        assert call_kwargs["deterministic"] is False


class TestRunEvaluationWithRender:
    """Tests for evaluation with rendering."""

    def test_run_evaluation_with_render_returns_metrics(self):
        """Test that render evaluation returns metrics."""
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not available")

        # Create mock trainer with render-capable environment
        mock_trainer = MagicMock()
        mock_state = MagicMock()

        # Mock environment
        mock_env = MagicMock()
        mock_env.agents = ['agent_0']
        mock_env.reset.return_value = ({'agent_0': np.zeros((10, 10, 3))}, MagicMock())
        mock_env.step.return_value = (
            {'agent_0': np.zeros((10, 10, 3))},  # obs
            MagicMock(),  # state
            {'agent_0': 1.0},  # rewards
            {'__all__': True},  # dones
            {},  # info
        )
        mock_env.render.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_trainer.env = mock_env

        # Mock algorithm
        mock_algorithm = MagicMock()
        mock_algorithm.compute_action.return_value = (0, {})
        mock_trainer.algorithm = mock_algorithm

        # Mock RNG in state
        mock_state.algorithm_state.rng = jax.random.PRNGKey(42)

        results, gif_path = run_evaluation_with_render(
            trainer=mock_trainer,
            state=mock_state,
            num_episodes=2,
            deterministic=True,
            max_frames=10,
            fps=10,
            output_path=None,
            verbose=0,
        )

        assert "mean_return" in results
        assert "episode_returns" in results
        assert gif_path is None  # No output path specified

    def test_run_evaluation_with_render_saves_gif(self):
        """Test that render evaluation saves GIF when output path specified."""
        try:
            import jax
            from PIL import Image
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

        # Create mock trainer with render-capable environment
        mock_trainer = MagicMock()
        mock_state = MagicMock()

        # Mock environment
        mock_env = MagicMock()
        mock_env.agents = ['agent_0']
        mock_env.reset.return_value = ({'agent_0': np.zeros((10, 10, 3))}, MagicMock())
        mock_env.step.return_value = (
            {'agent_0': np.zeros((10, 10, 3))},  # obs
            MagicMock(),  # state
            {'agent_0': 1.0},  # rewards
            {'__all__': True},  # dones
            {},  # info
        )
        mock_env.render.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_trainer.env = mock_env

        # Mock algorithm
        mock_algorithm = MagicMock()
        mock_algorithm.compute_action.return_value = (0, {})
        mock_trainer.algorithm = mock_algorithm

        # Mock RNG in state
        mock_state.algorithm_state.rng = jax.random.PRNGKey(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_eval.gif")

            results, gif_path = run_evaluation_with_render(
                trainer=mock_trainer,
                state=mock_state,
                num_episodes=1,
                deterministic=True,
                max_frames=5,
                fps=10,
                output_path=output_path,
                verbose=0,
            )

            assert gif_path is not None
            assert os.path.exists(gif_path)


class TestPrintResults:
    """Tests for results printing."""

    def test_print_results_outputs_info(self, capsys):
        """Test that print_results prints metrics."""
        results = {
            "num_episodes": 10,
            "mean_return": 1.5,
            "std_return": 0.5,
            "min_return": 0.5,
            "max_return": 2.5,
            "mean_length": 100.0,
            "std_length": 10.0,
        }

        print_results(results, verbose=1)

        captured = capsys.readouterr()
        assert "10" in captured.out
        assert "1.5" in captured.out
        assert "0.5" in captured.out

    def test_print_results_silent_mode(self, capsys):
        """Test that print_results respects silent mode."""
        results = {
            "num_episodes": 10,
            "mean_return": 1.5,
            "std_return": 0.5,
            "min_return": 0.5,
            "max_return": 2.5,
            "mean_length": 100.0,
            "std_length": 10.0,
        }

        print_results(results, verbose=0)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_results_verbose_shows_episodes(self, capsys):
        """Test that verbose=2 shows individual episodes."""
        results = {
            "num_episodes": 3,
            "mean_return": 1.0,
            "std_return": 0.5,
            "min_return": 0.5,
            "max_return": 1.5,
            "mean_length": 100.0,
            "std_length": 10.0,
            "episode_returns": [0.5, 1.0, 1.5],
        }

        print_results(results, verbose=2)

        captured = capsys.readouterr()
        assert "Episode 1" in captured.out
        assert "Episode 2" in captured.out
        assert "Episode 3" in captured.out


class TestSaveResultsJson:
    """Tests for saving results to JSON."""

    def test_save_results_json_creates_file(self):
        """Test that save_results_json creates a JSON file."""
        results = {
            "mean_return": 1.5,
            "std_return": 0.5,
            "num_episodes": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            save_results_json(results, output_path, verbose=0)

            assert os.path.exists(output_path)

            with open(output_path, "r") as f:
                loaded = json.load(f)

            assert loaded["mean_return"] == 1.5
            assert loaded["num_episodes"] == 10

    def test_save_results_json_creates_directories(self):
        """Test that save_results_json creates parent directories."""
        results = {"mean_return": 1.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "results.json")
            save_results_json(results, output_path, verbose=0)

            assert os.path.exists(output_path)


class TestSaveGif:
    """Tests for GIF saving."""

    def test_save_gif_with_pil(self):
        """Test saving GIF with PIL available."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.gif")
            result = save_gif(frames, output_path, fps=10, verbose=0)

            assert result is not None
            assert os.path.exists(result)

    def test_save_gif_no_frames(self):
        """Test handling empty frame list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.gif")
            result = save_gif([], output_path, fps=10, verbose=0)

            assert result is None

    def test_save_gif_normalizes_float_frames(self):
        """Test that float frames are normalized to uint8."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Float frames in [0, 1] range
        frames = [
            np.random.rand(50, 50, 3).astype(np.float32),
            np.random.rand(50, 50, 3).astype(np.float32),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.gif")
            result = save_gif(frames, output_path, fps=10, verbose=0)

            assert result is not None
            assert os.path.exists(result)


class TestCLIIntegration:
    """Tests for CLI integration."""

    def test_help_command_works(self):
        """Test that --help works without error."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/evaluate.py --help > /dev/null 2>&1'
        )
        assert result == 0

    def test_missing_required_args_exits(self):
        """Test that missing required args causes exit."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/evaluate.py > /dev/null 2>&1'
        )
        assert result != 0  # Should fail

    def test_missing_env_arg_exits(self):
        """Test that missing env arg causes exit."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/evaluate.py --checkpoint /some/path > /dev/null 2>&1'
        )
        assert result != 0  # Should fail

    def test_missing_checkpoint_arg_exits(self):
        """Test that missing checkpoint arg causes exit."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/evaluate.py --env coin_game > /dev/null 2>&1'
        )
        assert result != 0  # Should fail


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_detect_algorithm_with_nested_config(self):
        """Test detecting algorithm with nested config structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir)
            trainer_info_path = checkpoint_path / "trainer_info.pkl"

            # Create deeply nested config
            trainer_info = {
                "config": {
                    "algorithm": {
                        "training": {"lr": 0.001},
                        "network": {"hidden": 64},
                        "name": "vdn"
                    }
                }
            }
            with open(trainer_info_path, "wb") as f:
                pickle.dump(trainer_info, f)

            result = detect_algorithm_from_checkpoint(str(checkpoint_path))
            assert result == 'vdn'

    def test_print_results_with_empty_episode_list(self, capsys):
        """Test print_results handles empty episode list."""
        results = {
            "num_episodes": 0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "min_return": 0.0,
            "max_return": 0.0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "episode_returns": [],
        }

        # Should not raise
        print_results(results, verbose=1)

    def test_save_gif_with_pil_import_error(self):
        """Test that save_gif handles PIL not being available."""
        frames = [np.zeros((10, 10, 3))]

        with patch.dict('sys.modules', {'PIL': None, 'PIL.Image': None}):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "test.gif")
                result = save_gif(frames, output_path, fps=10, verbose=0)

                # Should return None when PIL not available
                assert result is None


class TestCheckpointLoadingIntegration:
    """Integration tests for checkpoint loading."""

    @pytest.fixture
    def minimal_checkpoint(self):
        """Create a minimal valid checkpoint for testing."""
        tmpdir = tempfile.mkdtemp()
        checkpoint_path = Path(tmpdir)

        # Create algorithm subdirectory
        algo_path = checkpoint_path / "algorithm"
        algo_path.mkdir()

        # Create minimal params file
        import pickle
        import jax.numpy as jnp

        params = {"params": {"dense": {"kernel": jnp.zeros((10, 5)), "bias": jnp.zeros(5)}}}
        with open(algo_path / "params.pkl", "wb") as f:
            pickle.dump(params, f)

        # Create optimizer state
        opt_state = {"count": 0, "mu": jnp.zeros(5), "nu": jnp.zeros(5)}
        with open(algo_path / "optimizer_state.pkl", "wb") as f:
            pickle.dump(opt_state, f)

        # Create trainer_info
        trainer_info = {
            "config": {
                "algorithm": {
                    "name": "ippo",
                    "training": {"learning_rate": 0.001, "gamma": 0.99},
                },
                "environment": {"name": "coin_game"},
            },
            "metrics": {
                "episode_returns": [1.0, 2.0, 3.0],
                "episode_lengths": [100, 150, 200],
                "losses": {"total_loss": [0.5, 0.4, 0.3]},
                "custom_metrics": {},
            },
        }
        with open(checkpoint_path / "trainer_info.pkl", "wb") as f:
            pickle.dump(trainer_info, f)

        yield str(checkpoint_path)

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)

    def test_checkpoint_structure(self, minimal_checkpoint):
        """Test that checkpoint has expected structure."""
        checkpoint_path = Path(minimal_checkpoint)

        assert (checkpoint_path / "algorithm").exists()
        assert (checkpoint_path / "algorithm" / "params.pkl").exists()
        assert (checkpoint_path / "trainer_info.pkl").exists()

        # Verify trainer_info structure
        with open(checkpoint_path / "trainer_info.pkl", "rb") as f:
            trainer_info = pickle.load(f)

        assert "config" in trainer_info
        assert "metrics" in trainer_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
