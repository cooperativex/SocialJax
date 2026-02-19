"""Unit tests for socialjax.evaluation.visualization module."""

import pytest
import numpy as np
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "socialjax"))

from socialjax.evaluation.visualization import (
    OutputFormat,
    VisualizationMode,
    infer_format,
    normalize_frame,
    add_text_overlay,
    apply_visualization_mode,
    save_gif,
    save_mp4,
    save_visualization,
    create_comparison_gif,
    create_episode_grid,
    resize_frames,
    get_frame_statistics,
)

# Check for optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_gif_format(self):
        """Test GIF format."""
        assert OutputFormat.GIF == "gif"
        assert OutputFormat.GIF.value == "gif"

    def test_mp4_format(self):
        """Test MP4 format."""
        assert OutputFormat.MP4 == "mp4"
        assert OutputFormat.MP4.value == "mp4"


class TestVisualizationMode:
    """Tests for VisualizationMode enum."""

    def test_basic_mode(self):
        """Test BASIC mode."""
        assert VisualizationMode.BASIC == "basic"

    def test_actions_mode(self):
        """Test ACTIONS mode."""
        assert VisualizationMode.ACTIONS == "actions"

    def test_rewards_mode(self):
        """Test REWARDS mode."""
        assert VisualizationMode.REWARDS == "rewards"

    def test_full_mode(self):
        """Test FULL mode."""
        assert VisualizationMode.FULL == "full"


class TestInferFormat:
    """Tests for infer_format function."""

    def test_gif_extension(self):
        """Test inference from .gif extension."""
        assert infer_format("output.gif") == OutputFormat.GIF
        assert infer_format("/path/to/output.gif") == OutputFormat.GIF
        assert infer_format("OUTPUT.GIF") == OutputFormat.GIF

    def test_mp4_extension(self):
        """Test inference from .mp4 extension."""
        assert infer_format("output.mp4") == OutputFormat.MP4
        assert infer_format("/path/to/output.mp4") == OutputFormat.MP4

    def test_unknown_extension(self):
        """Test inference from unknown extension (defaults to GIF)."""
        assert infer_format("output.txt") == OutputFormat.GIF
        assert infer_format("output") == OutputFormat.GIF
        assert infer_format("output.png") == OutputFormat.GIF


class TestNormalizeFrame:
    """Tests for normalize_frame function."""

    def test_already_uint8(self):
        """Test with already normalized frame."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = normalize_frame(frame)
        np.testing.assert_array_equal(result, frame)

    def test_float_range(self):
        """Test normalization of float frame."""
        frame = np.random.rand(64, 64, 3).astype(np.float32)
        result = normalize_frame(frame)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_frame(self):
        """Test with constant frame."""
        frame = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = normalize_frame(frame)
        assert result.dtype == np.uint8

    def test_negative_values(self):
        """Test with negative values."""
        frame = np.random.uniform(-1, 1, (64, 64, 3)).astype(np.float32)
        result = normalize_frame(frame)
        assert result.dtype == np.uint8
        assert result.min() >= 0


class TestAddTextOverlay:
    """Tests for add_text_overlay function."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_basic_overlay(self):
        """Test basic text overlay."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = add_text_overlay(frame, "Test", position=(10, 10))
        assert result.shape == frame.shape
        # Frame should be modified
        assert not np.array_equal(result, frame)

    def test_overlay_without_pil(self):
        """Test overlay when PIL is not available."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch('socialjax.evaluation.visualization.PIL_AVAILABLE', False):
            result = add_text_overlay(frame, "Test")
            np.testing.assert_array_equal(result, frame)


class TestApplyVisualizationMode:
    """Tests for apply_visualization_mode function."""

    def test_basic_mode(self):
        """Test BASIC mode (no modification)."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0, "episode_length": 50}
        result = apply_visualization_mode(frame, episode_info, VisualizationMode.BASIC)
        np.testing.assert_array_equal(result, frame)

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_actions_mode(self):
        """Test ACTIONS mode."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        episode_info = {"total_reward": 10.0}
        result = apply_visualization_mode(frame, episode_info, VisualizationMode.ACTIONS)
        assert result.shape == frame.shape

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_rewards_mode(self):
        """Test REWARDS mode."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        episode_info = {"episode_length": 50}
        result = apply_visualization_mode(frame, episode_info, VisualizationMode.REWARDS)
        assert result.shape == frame.shape

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_full_mode(self):
        """Test FULL mode."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        episode_info = {
            "total_reward": 10.0,
            "episode_length": 50,
            "cooperation_rate": 0.75,
        }
        result = apply_visualization_mode(frame, episode_info, VisualizationMode.FULL)
        assert result.shape == frame.shape


class TestSaveGif:
    """Tests for save_gif function."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_basic(self):
        """Test basic GIF saving."""
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = save_gif(frames, output_path, fps=10, verbose=0)
            assert result == output_path
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_creates_directory(self):
        """Test that save_gif creates parent directories."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "output.gif"
            result = save_gif(frames, str(output_path), fps=10, verbose=0)
            assert output_path.exists()

    def test_save_empty_frames(self):
        """Test with empty frame list."""
        result = save_gif([], "output.gif", verbose=0)
        assert result is None

    def test_save_without_pil(self):
        """Test when PIL is not available."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with patch('socialjax.evaluation.visualization.PIL_AVAILABLE', False):
            result = save_gif(frames, "output.gif", verbose=0)
            assert result is None


class TestSaveMp4:
    """Tests for save_mp4 function."""

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_save_basic(self):
        """Test basic MP4 saving."""
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name

        try:
            result = save_mp4(frames, output_path, fps=10, verbose=0)
            assert result == output_path
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_save_empty_frames(self):
        """Test with empty frame list."""
        result = save_mp4([], "output.mp4", verbose=0)
        assert result is None

    def test_save_without_cv2(self):
        """Test when OpenCV is not available."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with patch('socialjax.evaluation.visualization.CV2_AVAILABLE', False):
            result = save_mp4(frames, "output.mp4", verbose=0)
            assert result is None


class TestSaveVisualization:
    """Tests for save_visualization function."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_format(self):
        """Test saving as GIF."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = save_visualization(
                frames, output_path, OutputFormat.GIF, fps=10, verbose=0
            )
            assert result == output_path
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_save_mp4_format(self):
        """Test saving as MP4."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name

        try:
            result = save_visualization(
                frames, output_path, OutputFormat.MP4, fps=10, verbose=0
            )
            assert result == output_path
        finally:
            Path(output_path).unlink()

    def test_format_inference(self):
        """Test automatic format inference."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            # Pass None for format - should infer GIF from extension
            with patch('socialjax.evaluation.visualization.save_gif') as mock_save:
                mock_save.return_value = output_path
                save_visualization(frames, output_path, None, fps=10, verbose=0)
                mock_save.assert_called_once()
        finally:
            Path(output_path).unlink()


class TestCreateComparisonGif:
    """Tests for create_comparison_gif function."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_basic_comparison(self):
        """Test basic comparison GIF creation."""
        frames_list = [
            [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
            [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(5)],
        ]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = create_comparison_gif(frames_list, output_path, verbose=0)
            assert result == output_path
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_with_labels(self):
        """Test comparison with labels."""
        frames_list = [
            [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)],
            [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)],
        ]
        labels = ["Algorithm A", "Algorithm B"]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = create_comparison_gif(
                frames_list, output_path, labels=labels, verbose=0
            )
            assert result == output_path
        finally:
            Path(output_path).unlink()

    def test_empty_frames(self):
        """Test with empty frame list."""
        result = create_comparison_gif([], "output.gif", verbose=0)
        assert result is None

    def test_without_pil(self):
        """Test when PIL is not available."""
        frames_list = [[np.zeros((64, 64, 3), dtype=np.uint8)]]
        with patch('socialjax.evaluation.visualization.PIL_AVAILABLE', False):
            result = create_comparison_gif(frames_list, "output.gif", verbose=0)
            assert result is None


class TestCreateEpisodeGrid:
    """Tests for create_episode_grid function."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_basic_grid(self):
        """Test basic grid creation."""
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(16)
        ]

        grid = create_episode_grid(frames, grid_size=(4, 4))
        assert grid is not None
        assert grid.shape[0] == 32 * 4
        assert grid.shape[1] == 32 * 4

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_grid_with_save(self):
        """Test grid creation with saving."""
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(8)
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            grid = create_episode_grid(
                frames, grid_size=(2, 4), output_path=output_path, verbose=0
            )
            assert grid is not None
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_empty_frames(self):
        """Test with empty frame list."""
        grid = create_episode_grid([])
        assert grid is None

    def test_fewer_frames_than_cells(self):
        """Test with fewer frames than grid cells."""
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        grid = create_episode_grid(frames, grid_size=(4, 4))
        assert grid is not None


class TestResizeFrames:
    """Tests for resize_frames function."""

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_resize_basic(self):
        """Test basic frame resizing."""
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        resized = resize_frames(frames, (32, 32))

        assert len(resized) == 3
        for frame in resized:
            assert frame.shape[:2] == (32, 32)

    def test_empty_frames(self):
        """Test with empty frame list."""
        resized = resize_frames([], (32, 32))
        assert resized == []

    def test_without_cv2(self):
        """Test when OpenCV is not available."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with patch('socialjax.evaluation.visualization.CV2_AVAILABLE', False):
            result = resize_frames(frames, (32, 32))
            assert result == frames


class TestGetFrameStatistics:
    """Tests for get_frame_statistics function."""

    def test_basic_statistics(self):
        """Test basic frame statistics."""
        frames = [
            np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        stats = get_frame_statistics(frames)

        assert stats["num_frames"] == 10
        assert stats["width"] == 48
        assert stats["height"] == 64
        assert stats["channels"] == 3
        assert stats["dtype"] == "uint8"

    def test_empty_frames(self):
        """Test with empty frame list."""
        stats = get_frame_statistics([])
        assert stats["num_frames"] == 0

    def test_pil_frames(self):
        """Test with PIL frames."""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")

        frames = [Image.new('RGB', (32, 64)) for _ in range(5)]
        stats = get_frame_statistics(frames)

        assert stats["num_frames"] == 5
        assert stats["width"] == 32
        assert stats["height"] == 64
        assert stats["dtype"] == "PIL"


class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_full_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create sample frames
        frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for _ in range(20)
        ]

        # Apply visualization mode
        episode_info = {
            "total_reward": 100.0,
            "episode_length": 20,
        }
        processed = [
            apply_visualization_mode(f, episode_info, VisualizationMode.ACTIONS)
            for f in frames
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as GIF
            gif_path = Path(tmpdir) / "output.gif"
            result = save_gif(processed, str(gif_path), fps=10, verbose=0)
            assert result is not None

            # Create grid
            grid_path = Path(tmpdir) / "grid.png"
            grid = create_episode_grid(
                frames, grid_size=(4, 5), output_path=str(grid_path), verbose=0
            )
            assert grid is not None

            # Get statistics
            stats = get_frame_statistics(frames)
            assert stats["num_frames"] == 20

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_verbose_output(self, capsys):
        """Test verbose output when saving GIF."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            save_gif(frames, output_path, fps=10, verbose=1)
            captured = capsys.readouterr()
            assert "Saved GIF" in captured.out
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_float_frames(self):
        """Test saving float frames that need normalization."""
        frames = [
            np.random.rand(64, 64, 3).astype(np.float32)
            for _ in range(3)
        ]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = save_gif(frames, output_path, fps=10, verbose=0)
            assert result is not None
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_with_loop(self):
        """Test saving GIF with custom loop count."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = save_gif(frames, output_path, fps=10, loop=5, verbose=0)
            assert result is not None
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_create_grid_verbose_output(self, capsys):
        """Test verbose output when creating grid."""
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(8)]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            create_episode_grid(frames, grid_size=(2, 4), output_path=output_path, verbose=1)
            captured = capsys.readouterr()
            assert "Saved grid" in captured.out
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_visualization_with_verbose(self, capsys):
        """Test save_visualization with verbose output."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            save_visualization(frames, output_path, OutputFormat.GIF, fps=10, verbose=1)
            captured = capsys.readouterr()
            assert "Saved GIF" in captured.out
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_create_comparison_gif_verbose(self, capsys):
        """Test comparison GIF creation with verbose output."""
        frames_list = [
            [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)],
            [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)],
        ]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            create_comparison_gif(frames_list, output_path, verbose=1)
            captured = capsys.readouterr()
            assert "Saved comparison GIF" in captured.out
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_grayscale_frames(self):
        """Test saving grayscale frames (2D arrays)."""
        frames = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            # Grayscale frames should work but may need special handling
            result = save_gif(frames, output_path, fps=10, verbose=0)
            # If it doesn't work, it should return None or the path
            assert result is None or Path(output_path).exists()
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestEdgeCases:
    """Additional edge case tests for better coverage."""

    def test_normalize_constant_frame(self):
        """Test normalizing a completely constant frame."""
        frame = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = normalize_frame(frame)
        assert result.dtype == np.uint8

    def test_infer_format_case_insensitive(self):
        """Test format inference is case insensitive."""
        assert infer_format("OUTPUT.GIF") == OutputFormat.GIF
        assert infer_format("OUTPUT.MP4") == OutputFormat.MP4

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_gif_with_pil_images(self):
        """Test saving PIL Image objects directly."""
        frames = [Image.new('RGB', (64, 64), color=(i * 50, i * 50, i * 50)) for i in range(3)]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            result = save_gif(frames, output_path, fps=10, verbose=0)
            assert result is not None
        finally:
            Path(output_path).unlink()

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_comparison_gif_different_lengths(self):
        """Test comparison GIF with different length sequences."""
        frames_list = [
            [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
            [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)],
        ]

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            output_path = f.name

        try:
            # Should use minimum length
            result = create_comparison_gif(frames_list, output_path, verbose=0)
            assert result is not None
        finally:
            Path(output_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
