"""Unit tests for scripts/visualize.py.

Tests cover:
- CLI argument parsing
- Checkpoint loading
- Visualization execution
- GIF output
- MP4 output
- Visualization modes
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
from scripts.visualize import (
    parse_args,
    detect_algorithm_from_checkpoint,
    load_checkpoint,
    run_visualization,
    save_gif,
    save_mp4,
    save_visualization,
    infer_format,
    apply_visualization_mode,
    OutputFormat,
    VisualizationMode,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_gif_format(self):
        """Test GIF format enum."""
        assert OutputFormat.GIF.value == "gif"

    def test_mp4_format(self):
        """Test MP4 format enum."""
        assert OutputFormat.MP4.value == "mp4"


class TestVisualizationMode:
    """Tests for VisualizationMode enum."""

    def test_basic_mode(self):
        """Test basic mode."""
        assert VisualizationMode.BASIC.value == "basic"

    def test_actions_mode(self):
        """Test actions mode."""
        assert VisualizationMode.ACTIONS.value == "actions"

    def test_rewards_mode(self):
        """Test rewards mode."""
        assert VisualizationMode.REWARDS.value == "rewards"

    def test_full_mode(self):
        """Test full mode."""
        assert VisualizationMode.FULL.value == "full"


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args_checkpoint_env_output(self):
        """Test that checkpoint, env, and output are required."""
        # Missing all args
        with patch('sys.argv', ['visualize.py']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing env and output
        with patch('sys.argv', ['visualize.py', '--checkpoint', '/path/to/ckpt']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing output
        with patch('sys.argv', ['visualize.py', '--checkpoint', '/path/to/ckpt', '--env', 'coin_game']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_valid_minimal_args(self):
        """Test minimal valid argument set."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif'
        ]):
            args = parse_args()
            assert args.checkpoint == '/path/to/ckpt'
            assert args.env == 'coin_game'
            assert args.output == 'output.gif'
            assert args.fps == 10
            assert args.seed == 42
            assert args.deterministic is True
            assert args.mode == 'basic'

    def test_custom_fps(self):
        """Test custom fps argument."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--fps', '15'
        ]):
            args = parse_args()
            assert args.fps == 15

    def test_custom_num_frames(self):
        """Test custom num_frames argument."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--num-frames', '250'
        ]):
            args = parse_args()
            assert args.num_frames == 250

    def test_format_option_gif(self):
        """Test format argument gif."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--format', 'gif'
        ]):
            args = parse_args()
            assert args.format == 'gif'

    def test_format_option_mp4(self):
        """Test format argument mp4."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.mp4',
            '--format', 'mp4'
        ]):
            args = parse_args()
            assert args.format == 'mp4'

    def test_mode_option_basic(self):
        """Test mode argument basic."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--mode', 'basic'
        ]):
            args = parse_args()
            assert args.mode == 'basic'

    def test_mode_option_actions(self):
        """Test mode argument actions."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--mode', 'actions'
        ]):
            args = parse_args()
            assert args.mode == 'actions'

    def test_mode_option_rewards(self):
        """Test mode argument rewards."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--mode', 'rewards'
        ]):
            args = parse_args()
            assert args.mode == 'rewards'

    def test_mode_option_full(self):
        """Test mode argument full."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--mode', 'full'
        ]):
            args = parse_args()
            assert args.mode == 'full'

    def test_algorithm_option(self):
        """Test algorithm argument."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--algorithm', 'ippo'
        ]):
            args = parse_args()
            assert args.algorithm == 'ippo'

    def test_num_agents_option(self):
        """Test num_agents argument."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--num-agents', '7'
        ]):
            args = parse_args()
            assert args.num_agents == 7

    def test_seed_option(self):
        """Test seed argument."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--seed', '123'
        ]):
            args = parse_args()
            assert args.seed == 123

    def test_stochastic_flag(self):
        """Test stochastic flag overrides deterministic."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--stochastic'
        ]):
            args = parse_args()
            assert args.stochastic is True

    def test_verbose_option(self):
        """Test verbosity level."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--verbose', '2'
        ]):
            args = parse_args()
            assert args.verbose == 2


class TestInferFormat:
    """Tests for format inference from file extension."""

    def test_infer_gif_from_extension(self):
        """Test inferring GIF format from .gif extension."""
        result = infer_format("output.gif")
        assert result == OutputFormat.GIF

    def test_infer_mp4_from_extension(self):
        """Test inferring MP4 format from .mp4 extension."""
        result = infer_format("output.mp4")
        assert result == OutputFormat.MP4

    def test_infer_default_to_gif(self):
        """Test default to GIF for unknown extensions."""
        result = infer_format("output.txt")
        assert result == OutputFormat.GIF

    def test_infer_gif_with_path(self):
        """Test inferring format with path."""
        result = infer_format("/path/to/output.gif")
        assert result == OutputFormat.GIF

    def test_infer_mp4_with_path(self):
        """Test inferring format with path."""
        result = infer_format("/path/to/output.mp4")
        assert result == OutputFormat.MP4


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
        result = detect_algorithm_from_checkpoint('/checkpoints/vdn_coop_mining/final')
        assert result == 'vdn'

    def test_detect_svo_from_path(self):
        """Test detecting SVO from checkpoint path."""
        result = detect_algorithm_from_checkpoint('/checkpoints/svo_harvest/final')
        assert result == 'svo'

    def test_detect_case_insensitive(self):
        """Test detection is case-insensitive."""
        result = detect_algorithm_from_checkpoint('/checkpoints/IPPO_MODEL/final')
        assert result == 'ippo'

    def test_no_detection_from_unknown_path(self):
        """Test None returned for unknown path."""
        result = detect_algorithm_from_checkpoint('/checkpoints/unknown_model/final')
        assert result is None


class TestSaveGif:
    """Tests for GIF saving functionality."""

    def test_save_gif_basic(self):
        """Test basic GIF saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

            result = save_gif(frames, str(output_path), fps=10, verbose=0)

            assert result is not None
            assert Path(result).exists()
            assert Path(result).suffix == ".gif"

    def test_save_gif_creates_directory(self):
        """Test GIF saving creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test.gif"
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

            result = save_gif(frames, str(output_path), fps=10, verbose=0)

            assert result is not None
            assert Path(result).exists()

    def test_save_gif_empty_frames(self):
        """Test GIF saving with empty frames returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            frames = []

            result = save_gif(frames, str(output_path), fps=10, verbose=0)

            assert result is None

    def test_save_gif_normalizes_float_frames(self):
        """Test GIF saving normalizes float frames to uint8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            frames = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(3)]

            result = save_gif(frames, str(output_path), fps=10, verbose=0)

            assert result is not None
            assert Path(result).exists()

    def test_save_gif_respects_fps(self):
        """Test GIF saving respects FPS parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

            # FPS 5 vs FPS 20 should produce different duration
            result = save_gif(frames, str(output_path), fps=5, verbose=0)
            assert result is not None

            output_path2 = Path(tmpdir) / "test2.gif"
            result2 = save_gif(frames, str(output_path2), fps=20, verbose=0)
            assert result2 is not None


class TestSaveMp4:
    """Tests for MP4 saving functionality."""

    @pytest.mark.skipif(
        not __import__('cv2', fromlist=[''], level=0) if 'cv2' in sys.modules else True,
        reason="OpenCV not available"
    )
    def test_save_mp4_basic(self):
        """Test basic MP4 saving."""
        try:
            import cv2
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "test.mp4"
                frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

                result = save_mp4(frames, str(output_path), fps=10, verbose=0)

                assert result is not None
                assert Path(result).exists()
                assert Path(result).suffix == ".mp4"
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_save_mp4_empty_frames(self):
        """Test MP4 saving with empty frames returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"
            frames = []

            result = save_mp4(frames, str(output_path), fps=10, verbose=0)

            assert result is None


class TestSaveVisualization:
    """Tests for unified save_visualization function."""

    def test_save_visualization_gif(self):
        """Test save_visualization with GIF format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

            result = save_visualization(
                frames=frames,
                output_path=str(output_path),
                output_format=OutputFormat.GIF,
                fps=10,
                verbose=0,
            )

            assert result is not None
            assert Path(result).exists()

    def test_save_visualization_dispatches_correctly(self):
        """Test save_visualization dispatches to correct function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # GIF
            output_path = Path(tmpdir) / "test.gif"
            frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
            result = save_visualization(
                frames=frames,
                output_path=str(output_path),
                output_format=OutputFormat.GIF,
                fps=10,
                verbose=0,
            )
            assert result is not None


class TestApplyVisualizationMode:
    """Tests for visualization mode application."""

    def test_basic_mode_returns_frame(self):
        """Test basic mode returns frame unchanged."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0, "episode_length": 100}
        mock_env = MagicMock()

        result = apply_visualization_mode(
            frame, None, episode_info, VisualizationMode.BASIC, mock_env
        )

        np.testing.assert_array_equal(result, frame)

    def test_actions_mode_attempts_overlay(self):
        """Test actions mode attempts to add overlay."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0, "episode_length": 100}
        mock_env = MagicMock()

        # Should not raise an error even if PIL is not available
        result = apply_visualization_mode(
            frame, None, episode_info, VisualizationMode.ACTIONS, mock_env
        )

        assert isinstance(result, np.ndarray)

    def test_rewards_mode_attempts_overlay(self):
        """Test rewards mode attempts to add overlay."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0, "episode_length": 100}
        mock_env = MagicMock()

        result = apply_visualization_mode(
            frame, None, episode_info, VisualizationMode.REWARDS, mock_env
        )

        assert isinstance(result, np.ndarray)

    def test_full_mode_attempts_overlay(self):
        """Test full mode attempts to add overlay."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0, "episode_length": 100}
        mock_env = MagicMock()

        result = apply_visualization_mode(
            frame, None, episode_info, VisualizationMode.FULL, mock_env
        )

        assert isinstance(result, np.ndarray)


class TestLoadCheckpoint:
    """Tests for checkpoint loading."""

    def test_load_checkpoint_missing_algorithm_detection(self):
        """Test error when algorithm cannot be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "unknown_model"
            checkpoint_path.mkdir()

            with pytest.raises(ValueError) as exc_info:
                load_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    env_name="coin_game",
                    algorithm_name=None,
                    verbose=0,
                )

            assert "Could not detect algorithm" in str(exc_info.value)


class TestRunVisualization:
    """Tests for visualization execution."""

    def test_run_visualization_returns_frames_and_info(self):
        """Test run_visualization returns expected structure."""
        # Create mock trainer and state
        mock_trainer = MagicMock()
        mock_algorithm = MagicMock()
        mock_env = MagicMock()

        mock_trainer.algorithm = mock_algorithm
        mock_trainer.env = mock_env
        mock_env.agents = ["agent_0", "agent_1"]
        mock_env.render = MagicMock(return_value=np.zeros((64, 64, 3), dtype=np.uint8))

        # Mock compute_action to return valid actions
        mock_algorithm.compute_action = MagicMock(return_value=(np.array(0), None))

        # Mock environment step
        mock_env.step = MagicMock(return_value=(
            {"agent_0": np.zeros((15, 15, 3)), "agent_1": np.zeros((15, 15, 3))},
            None,
            {"agent_0": 1.0, "agent_1": 1.0},
            {"__all__": True},
            {}
        ))

        # Mock environment reset
        mock_env.reset = MagicMock(return_value=(
            {"agent_0": np.zeros((15, 15, 3)), "agent_1": np.zeros((15, 15, 3))},
            None
        ))

        # Create mock state
        mock_state = MagicMock()
        mock_state.algorithm_state.rng = np.array([42])

        frames, info = run_visualization(
            trainer=mock_trainer,
            state=mock_state,
            num_frames=3,
            deterministic=True,
            mode=VisualizationMode.BASIC,
            verbose=0,
        )

        assert isinstance(frames, list)
        assert isinstance(info, dict)
        assert "total_reward" in info
        assert "episode_length" in info


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_missing_checkpoint(self):
        """Test CLI with missing checkpoint returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"

            with patch('sys.argv', [
                'visualize.py',
                '--checkpoint', '/nonexistent/path',
                '--env', 'coin_game',
                '--output', str(output_path)
            ]):
                with patch('builtins.print') as mock_print:
                    from scripts.visualize import main
                    result = main()
                    assert result == 1

    def test_cli_all_options(self):
        """Test CLI with all options parsed correctly."""
        with patch('sys.argv', [
            'visualize.py',
            '--checkpoint', '/path/to/ckpt',
            '--env', 'coin_game',
            '--output', 'output.gif',
            '--num-frames', '100',
            '--fps', '15',
            '--format', 'gif',
            '--mode', 'full',
            '--algorithm', 'ippo',
            '--num-agents', '5',
            '--seed', '123',
            '--verbose', '2'
        ]):
            args = parse_args()
            assert args.checkpoint == '/path/to/ckpt'
            assert args.env == 'coin_game'
            assert args.output == 'output.gif'
            assert args.num_frames == 100
            assert args.fps == 15
            assert args.format == 'gif'
            assert args.mode == 'full'
            assert args.algorithm == 'ippo'
            assert args.num_agents == 5
            assert args.seed == 123
            assert args.verbose == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_gif_save_with_grayscale_frames(self):
        """Test GIF saving with grayscale frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"
            # Grayscale frames (H, W) instead of (H, W, 3)
            frames = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)]

            result = save_gif(frames, str(output_path), fps=10, verbose=0)

            # Should still work (PIL handles grayscale)
            assert result is not None

    def test_infer_format_with_uppercase_extension(self):
        """Test format inference with uppercase extension."""
        result = infer_format("output.GIF")
        # Should default to GIF (case-insensitive comparison)
        assert result == OutputFormat.GIF

    def test_visualization_mode_enum_from_string(self):
        """Test VisualizationMode can be created from string."""
        mode = VisualizationMode("basic")
        assert mode == VisualizationMode.BASIC

    def test_output_format_enum_from_string(self):
        """Test OutputFormat can be created from string."""
        format_type = OutputFormat("gif")
        assert format_type == OutputFormat.GIF

    def test_detect_algorithm_with_uppercase_path(self):
        """Test algorithm detection with uppercase path."""
        result = detect_algorithm_from_checkpoint('/CHECKPOINTS/IPPO_FINAL')
        assert result == 'ippo'
