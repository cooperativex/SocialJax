"""Visualization utilities for multi-agent RL evaluation.

This module provides functions for generating GIFs and videos from
evaluation episodes, including frame capture, processing, and saving.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import os

import numpy as np

# Check for optional dependencies
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class OutputFormat(str, Enum):
    """Supported output formats for visualization."""
    GIF = "gif"
    MP4 = "mp4"


class VisualizationMode(str, Enum):
    """Visualization modes for frame processing."""
    BASIC = "basic"          # Just render the environment
    ACTIONS = "actions"      # Overlay action information
    REWARDS = "rewards"      # Overlay reward information
    FULL = "full"            # Full visualization with all info


def infer_format(output_path: str) -> OutputFormat:
    """Infer output format from file extension.

    Args:
        output_path: Output file path

    Returns:
        Output format enum
    """
    ext = Path(output_path).suffix.lower()
    if ext == ".mp4":
        return OutputFormat.MP4
    return OutputFormat.GIF


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame values to uint8 range [0, 255].

    Args:
        frame: Input frame array

    Returns:
        Normalized frame as uint8
    """
    if frame.dtype == np.uint8:
        return frame

    # Normalize to [0, 1]
    min_val = frame.min()
    max_val = frame.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(frame, dtype=np.uint8)

    normalized = (frame - min_val) / (max_val - min_val)

    # Scale to [0, 255]
    return (normalized * 255).astype(np.uint8)


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 10),
    color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 12,
) -> np.ndarray:
    """Add text overlay to a frame.

    Args:
        frame: Input frame
        text: Text to add
        position: (x, y) position for text
        color: RGB color tuple
        font_size: Font size (ignored if PIL not available)

    Returns:
        Frame with text overlay
    """
    if not PIL_AVAILABLE:
        return frame

    try:
        # Normalize frame
        if frame.dtype != np.uint8:
            frame = normalize_frame(frame)

        # Convert to PIL Image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        draw.text(position, text, fill=color, font=font)

        return np.array(img)

    except Exception:
        return frame


def apply_visualization_mode(
    frame: np.ndarray,
    episode_info: Dict[str, Any],
    mode: VisualizationMode = VisualizationMode.BASIC,
) -> np.ndarray:
    """Apply visualization mode overlays to a frame.

    Args:
        frame: Input frame
        episode_info: Episode information dictionary
        mode: Visualization mode

    Returns:
        Frame with overlays
    """
    if mode == VisualizationMode.BASIC:
        return frame

    if not PIL_AVAILABLE:
        return frame

    try:
        result = frame.copy() if isinstance(frame, np.ndarray) else np.array(frame)
        y_offset = 10

        if mode in [VisualizationMode.ACTIONS, VisualizationMode.FULL]:
            if "total_reward" in episode_info:
                text = f"Reward: {episode_info['total_reward']:.2f}"
                result = add_text_overlay(result, text, (10, y_offset))
                y_offset += 20

        if mode in [VisualizationMode.REWARDS, VisualizationMode.FULL]:
            if "episode_length" in episode_info:
                text = f"Step: {episode_info['episode_length']}"
                result = add_text_overlay(result, text, (10, y_offset))
                y_offset += 20

        if mode == VisualizationMode.FULL:
            if "cooperation_rate" in episode_info:
                text = f"Coop: {episode_info['cooperation_rate']:.1%}"
                result = add_text_overlay(result, text, (10, y_offset))

        return result

    except Exception:
        return frame


def save_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    loop: int = 0,
    verbose: int = 1,
) -> Optional[str]:
    """Save frames as GIF.

    Args:
        frames: List of frames to save
        output_path: Output path for GIF
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        verbose: Verbosity level

    Returns:
        Path to saved GIF or None if failed
    """
    if not PIL_AVAILABLE:
        if verbose >= 1:
            print("PIL not available, cannot save GIF. Install with: pip install Pillow")
        return None

    if not frames:
        if verbose >= 1:
            print("No frames to save")
        return None

    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                normalized = normalize_frame(frame)
                pil_frames.append(Image.fromarray(normalized))
            elif isinstance(frame, Image.Image):
                pil_frames.append(frame)
            else:
                pil_frames.append(Image.fromarray(normalize_frame(np.array(frame))))

        if not pil_frames:
            if verbose >= 1:
                print("No valid frames to save")
            return None

        # Calculate duration in milliseconds
        duration = 1000 // fps

        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop,
        )

        if verbose >= 1:
            print(f"Saved GIF to: {output_path} ({len(pil_frames)} frames)")

        return str(output_path)

    except Exception as e:
        if verbose >= 1:
            print(f"Failed to save GIF: {e}")
        return None


def save_mp4(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    verbose: int = 1,
) -> Optional[str]:
    """Save frames as MP4 video.

    Args:
        frames: List of frames to save
        output_path: Output path for MP4
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved MP4 or None if failed
    """
    if not CV2_AVAILABLE:
        if verbose >= 1:
            print("OpenCV not available, cannot save MP4. Install with: pip install opencv-python")
        return None

    if not frames:
        if verbose >= 1:
            print("No frames to save")
        return None

    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get frame dimensions from first frame
        first_frame = frames[0]
        if isinstance(first_frame, np.ndarray):
            if first_frame.dtype != np.uint8:
                first_frame = normalize_frame(first_frame)
            h, w = first_frame.shape[:2]
        else:
            h, w = first_frame.height, first_frame.width

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not out.isOpened():
            if verbose >= 1:
                print("Failed to open video writer")
            return None

        # Write frames
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = normalize_frame(frame)
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                # PIL Image
                frame_np = np.array(frame)
                if frame_np.dtype != np.uint8:
                    frame_np = normalize_frame(frame_np)
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_np)

        out.release()

        if verbose >= 1:
            print(f"Saved MP4 to: {output_path} ({len(frames)} frames)")

        return str(output_path)

    except Exception as e:
        if verbose >= 1:
            print(f"Failed to save MP4: {e}")
        return None


def save_visualization(
    frames: List[np.ndarray],
    output_path: str,
    output_format: Optional[OutputFormat] = None,
    fps: int = 10,
    verbose: int = 1,
) -> Optional[str]:
    """Save visualization frames to file.

    Args:
        frames: List of frames
        output_path: Output path
        output_format: Output format (inferred from path if None)
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved file or None if failed
    """
    if output_format is None:
        output_format = infer_format(output_path)

    if output_format == OutputFormat.MP4:
        return save_mp4(frames, output_path, fps, verbose)
    else:
        return save_gif(frames, output_path, fps, verbose=verbose)


def create_comparison_gif(
    frames_list: List[List[np.ndarray]],
    output_path: str,
    labels: Optional[List[str]] = None,
    fps: int = 10,
    verbose: int = 1,
) -> Optional[str]:
    """Create a side-by-side comparison GIF from multiple frame sequences.

    Args:
        frames_list: List of frame sequences to compare
        output_path: Output path for GIF
        labels: Optional labels for each sequence
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved GIF or None if failed
    """
    if not PIL_AVAILABLE:
        if verbose >= 1:
            print("PIL not available for comparison GIF")
        return None

    if not frames_list or not frames_list[0]:
        if verbose >= 1:
            print("No frames to compare")
        return None

    try:
        # Find minimum length across all sequences
        min_length = min(len(frames) for frames in frames_list)

        # Normalize all frames
        normalized_sequences = []
        for frames in frames_list:
            normalized = []
            for frame in frames[:min_length]:
                if isinstance(frame, np.ndarray):
                    normalized.append(normalize_frame(frame))
                else:
                    normalized.append(normalize_frame(np.array(frame)))
            normalized_sequences.append(normalized)

        # Get dimensions
        h, w = normalized_sequences[0][0].shape[:2]
        n_sequences = len(frames_list)

        # Create combined frames
        combined_frames = []
        for i in range(min_length):
            combined = np.zeros((h, w * n_sequences, 3), dtype=np.uint8)
            for j, seq in enumerate(normalized_sequences):
                combined[:, j*w:(j+1)*w] = seq[i]
            combined_frames.append(Image.fromarray(combined))

        # Add labels if provided
        if labels:
            labeled_frames = []
            for frame in combined_frames:
                img = frame.copy()
                draw = ImageDraw.Draw(img)
                for j, label in enumerate(labels):
                    x = j * w + 10
                    draw.text((x, 10), label, fill=(255, 255, 255))
                labeled_frames.append(img)
            combined_frames = labeled_frames

        # Save GIF
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = 1000 // fps
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=duration,
            loop=0,
        )

        if verbose >= 1:
            print(f"Saved comparison GIF to: {output_path}")

        return str(output_path)

    except Exception as e:
        if verbose >= 1:
            print(f"Failed to create comparison GIF: {e}")
        return None


def create_episode_grid(
    frames: List[np.ndarray],
    grid_size: Tuple[int, int] = (4, 4),
    output_path: Optional[str] = None,
    verbose: int = 1,
) -> Optional[np.ndarray]:
    """Create a grid of frames from an episode.

    Args:
        frames: List of frames
        grid_size: (rows, cols) for the grid
        output_path: Optional path to save the grid image
        verbose: Verbosity level

    Returns:
        Grid image as numpy array or None if failed
    """
    if not frames:
        return None

    rows, cols = grid_size
    total_cells = rows * cols

    # Select evenly spaced frames
    if len(frames) <= total_cells:
        selected = frames
    else:
        indices = np.linspace(0, len(frames) - 1, total_cells, dtype=int)
        selected = [frames[i] for i in indices]

    # Normalize frames
    normalized = [normalize_frame(f) if isinstance(f, np.ndarray) else f for f in selected]

    # Get frame dimensions
    h, w = normalized[0].shape[:2]

    # Create grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for i, frame in enumerate(normalized):
        if i >= total_cells:
            break
        row = i // cols
        col = i % cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame

    # Save if path provided
    if output_path and PIL_AVAILABLE:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(output_path)
        if verbose >= 1:
            print(f"Saved grid to: {output_path}")

    return grid


def resize_frames(
    frames: List[np.ndarray],
    target_size: Tuple[int, int],
) -> List[np.ndarray]:
    """Resize frames to target size.

    Args:
        frames: List of frames
        target_size: (width, height) target size

    Returns:
        List of resized frames
    """
    if not CV2_AVAILABLE:
        return frames

    resized = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            resized.append(cv2.resize(frame, target_size))
        else:
            # PIL Image
            resized.append(np.array(frame.resize(target_size)))
    return resized


def get_frame_statistics(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Get statistics about a sequence of frames.

    Args:
        frames: List of frames

    Returns:
        Dictionary of statistics
    """
    if not frames:
        return {"num_frames": 0}

    first_frame = frames[0]
    if isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
        dtype = str(first_frame.dtype)
    else:
        h, w = first_frame.height, first_frame.width
        dtype = "PIL"

    return {
        "num_frames": len(frames),
        "width": w,
        "height": h,
        "dtype": dtype,
        "channels": first_frame.shape[2] if hasattr(first_frame, 'shape') and len(first_frame.shape) > 2 else 1,
    }
