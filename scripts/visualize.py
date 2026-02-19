#!/usr/bin/env python
"""Visualization script for SocialJax trained policies.

This script generates visualizations (GIFs, videos) of trained RL
policies operating in multi-agent environments.

Example usage:
    # Generate a GIF from a checkpoint
    python scripts/visualize.py --checkpoint checkpoints/ippo_final --env coin_game --output output.gif

    # Custom frame count and FPS
    python scripts/visualize.py --checkpoint X --env clean_up --output output.gif --num-frames 250 --fps 15

    # Generate video (MP4) instead of GIF
    python scripts/visualize.py --checkpoint X --env coin_game --output output.mp4 --format mp4

    # Visualization mode: overlay actions
    python scripts/visualize.py --checkpoint X --env coin_game --output output.gif --mode actions
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "socialjax"))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.training import Trainer
from socialjax.config.manager import ConfigManager
from socialjax.algorithms.registry import get_algorithm, list_algorithms

# Import algorithm modules to register them
try:
    import socialjax.algorithms.ippo
except ImportError:
    pass
try:
    import socialjax.algorithms.mappo
except ImportError:
    pass
try:
    import socialjax.algorithms.vdn
except ImportError:
    pass
try:
    import socialjax.algorithms.svo
except ImportError:
    pass


class OutputFormat(str, Enum):
    """Supported output formats for visualization."""
    GIF = "gif"
    MP4 = "mp4"


class VisualizationMode(str, Enum):
    """Visualization modes."""
    BASIC = "basic"  # Just render the environment
    ACTIONS = "actions"  # Overlay action information
    REWARDS = "rewards"  # Overlay reward information
    FULL = "full"  # Full visualization with all info


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate visualizations of trained SocialJax policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize.py --checkpoint checkpoints/ippo_final --env coin_game --output output.gif
  python scripts/visualize.py --checkpoint X --env clean_up --output output.gif --num-frames 250
  python scripts/visualize.py --checkpoint X --env coin_game --output output.mp4 --format mp4
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name (e.g., coin_game, clean_up, harvest_common_open)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for visualization (GIF or MP4)",
    )

    # Frame control
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to capture (default: run until episode end)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output (default: 10)",
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["gif", "mp4"],
        default=None,
        help="Output format (default: inferred from output file extension)",
    )

    # Visualization mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "actions", "rewards", "full"],
        default="basic",
        help="Visualization mode (default: basic)",
    )

    # Checkpoint info
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Number of agents (auto-detected from environment if not specified)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for visualization (default: 42)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (overrides --deterministic)",
    )

    # Output control
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=normal, 2=debug (default: 1)",
    )

    return parser.parse_args()


def detect_algorithm_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Detect algorithm name from checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Detected algorithm name or None
    """
    path = Path(checkpoint_path)

    # Try to extract from path name
    path_str = str(path).lower()
    for algo in ["ippo", "mappo", "vdn", "svo"]:
        if algo in path_str:
            return algo

    # Try to read from config if available
    config_path = path / "trainer_info.pkl"
    if config_path.exists():
        try:
            import pickle
            with open(config_path, "rb") as f:
                trainer_info = pickle.load(f)
            config = trainer_info.get("config", {})
            if "algorithm" in config:
                if isinstance(config["algorithm"], dict):
                    return config["algorithm"].get("name")
        except Exception:
            pass

    return None


def load_checkpoint(
    checkpoint_path: str,
    env_name: str,
    algorithm_name: Optional[str] = None,
    num_agents: Optional[int] = None,
    seed: int = 42,
    verbose: int = 1,
) -> Tuple[Trainer, Any]:
    """Load a trainer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        env_name: Environment name
        algorithm_name: Algorithm name (auto-detected if None)
        num_agents: Number of agents (uses default if None)
        seed: Random seed
        verbose: Verbosity level

    Returns:
        Tuple of (trainer, state)
    """
    # Detect algorithm if not specified
    if algorithm_name is None:
        algorithm_name = detect_algorithm_from_checkpoint(checkpoint_path)
        if algorithm_name is None:
            raise ValueError(
                f"Could not detect algorithm from checkpoint. "
                f"Please specify --algorithm. Available: {list_algorithms()}"
            )
        if verbose >= 1:
            print(f"Detected algorithm: {algorithm_name}")

    # Create environment
    if verbose >= 1:
        print(f"Creating environment: {env_name}")

    if num_agents is None:
        default_agents = {
            "coin_game": 5,
            "clean_up": 7,
            "harvest_common_open": 7,
            "coop_mining": 4,
            "territory_open": 4,
            "pd_arena": 4,
            "mushrooms": 5,
            "gift": 5,
        }
        num_agents = default_agents.get(env_name, 5)

    env = socialjax.make(env_name, num_agents=num_agents)

    # Create trainer
    trainer = Trainer(
        algorithm=algorithm_name,
        env=env,
        seed=seed,
    )

    # Load checkpoint
    if verbose >= 1:
        print(f"Loading checkpoint: {checkpoint_path}")

    state = trainer.load(checkpoint_path)

    return trainer, state


def run_visualization(
    trainer: Trainer,
    state: Any,
    num_frames: Optional[int] = None,
    deterministic: bool = True,
    mode: VisualizationMode = VisualizationMode.BASIC,
    verbose: int = 1,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Run visualization episode and collect frames.

    Args:
        trainer: Trainer instance
        state: Trainer state
        num_frames: Maximum number of frames (None for episode end)
        deterministic: Whether to use deterministic actions
        mode: Visualization mode
        verbose: Verbosity level

    Returns:
        Tuple of (frames, episode_info)
    """
    env = trainer.env
    algorithm = trainer.algorithm

    frames = []
    episode_rewards = []
    episode_actions = []
    episode_info = {
        "total_reward": 0.0,
        "episode_length": 0,
        "agent_rewards": {},
    }

    rng = jax.random.PRNGKey(
        state.algorithm_state.rng[0] if hasattr(state.algorithm_state.rng, '__getitem__') else 42
    )

    if verbose >= 1:
        print(f"\nRunning visualization episode...")
        print(f"Mode: {mode.value}")
        print(f"Deterministic actions: {deterministic}")

    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)

    frame_count = 0
    done = False

    while not done and (num_frames is None or frame_count < num_frames):
        # Render current state
        if hasattr(env, 'render'):
            try:
                frame = env.render(env_state)
                if frame is not None:
                    # Apply visualization mode overlays
                    frame = apply_visualization_mode(
                        frame, env_state, episode_info, mode, env
                    )
                    frames.append(frame)
            except Exception as e:
                if verbose >= 2:
                    print(f"Render error: {e}")

        # Compute actions for all agents
        actions = []
        for agent in env.agents:
            rng, action_rng = jax.random.split(rng)
            action, _ = algorithm.compute_action(
                state.algorithm_state,
                obs[agent],
                action_rng,
                deterministic=deterministic,
            )
            actions.append(np.array(action))

        episode_actions.append(actions.copy())

        # Step environment
        rng, step_rng = jax.random.split(rng)
        next_obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions)

        # Track rewards
        step_reward = 0.0
        if isinstance(rewards, dict):
            for agent_id, reward in rewards.items():
                if agent_id != "__all__":
                    step_reward += float(reward)
                    if agent_id not in episode_info["agent_rewards"]:
                        episode_info["agent_rewards"][agent_id] = 0.0
                    episode_info["agent_rewards"][agent_id] += float(reward)
        else:
            step_reward = float(np.sum(rewards))

        episode_rewards.append(step_reward)
        episode_info["total_reward"] += step_reward
        episode_info["episode_length"] += 1
        frame_count += 1

        # Check done
        if isinstance(dones, dict):
            done = dones.get("__all__", False)
        else:
            done = bool(dones)

        obs = next_obs

    episode_info["mean_reward"] = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    episode_info["frame_count"] = len(frames)

    if verbose >= 1:
        print(f"Captured {len(frames)} frames")
        print(f"Episode length: {episode_info['episode_length']}")
        print(f"Total reward: {episode_info['total_reward']:.4f}")

    return frames, episode_info


def apply_visualization_mode(
    frame: np.ndarray,
    env_state: Any,
    episode_info: Dict[str, Any],
    mode: VisualizationMode,
    env: Any,
) -> np.ndarray:
    """Apply visualization mode overlays to frame.

    Args:
        frame: Rendered frame
        env_state: Current environment state
        episode_info: Episode information
        mode: Visualization mode
        env: Environment instance

    Returns:
        Modified frame with overlays
    """
    # For now, just return the frame as-is
    # Future enhancement: add text overlays for actions, rewards, etc.
    if mode == VisualizationMode.BASIC:
        return frame

    # Try to add overlays for other modes
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
            img = Image.fromarray(frame)
        else:
            img = frame

        draw = ImageDraw.Draw(img)

        # Add overlay text based on mode
        y_offset = 10
        if mode in [VisualizationMode.ACTIONS, VisualizationMode.FULL]:
            text = f"Reward: {episode_info['total_reward']:.2f}"
            draw.text((10, y_offset), text, fill=(255, 255, 255))
            y_offset += 20

        if mode in [VisualizationMode.REWARDS, VisualizationMode.FULL]:
            text = f"Length: {episode_info['episode_length']}"
            draw.text((10, y_offset), text, fill=(255, 255, 255))

        return np.array(img)

    except ImportError:
        # PIL not available, return frame as-is
        return frame
    except Exception:
        return frame


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
    return OutputFormat.GIF  # Default to GIF


def save_gif(frames: List[np.ndarray], output_path: str, fps: int, verbose: int) -> Optional[str]:
    """Save frames as GIF.

    Args:
        frames: List of frames to save
        output_path: Output path for GIF
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved GIF or None if failed
    """
    try:
        from PIL import Image

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(frame))
            elif isinstance(frame, Image.Image):
                pil_frames.append(frame)

        if not pil_frames:
            if verbose >= 1:
                print("No frames to save")
            return None

        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps,
            loop=0,
        )

        if verbose >= 1:
            print(f"Saved GIF to: {output_path}")

        return str(output_path)

    except ImportError:
        if verbose >= 1:
            print("PIL not available, cannot save GIF. Install with: pip install Pillow")
        return None
    except Exception as e:
        if verbose >= 1:
            print(f"Failed to save GIF: {e}")
        return None


def save_mp4(frames: List[np.ndarray], output_path: str, fps: int, verbose: int) -> Optional[str]:
    """Save frames as MP4 video.

    Args:
        frames: List of frames to save
        output_path: Output path for MP4
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved MP4 or None if failed
    """
    try:
        import cv2

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            if verbose >= 1:
                print("No frames to save")
            return None

        # Get frame dimensions
        first_frame = frames[0]
        if isinstance(first_frame, np.ndarray):
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
                    frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                # PIL Image
                frame_np = np.array(frame)
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_np)

        out.release()

        if verbose >= 1:
            print(f"Saved MP4 to: {output_path}")

        return str(output_path)

    except ImportError:
        if verbose >= 1:
            print("OpenCV not available, cannot save MP4. Install with: pip install opencv-python")
        return None
    except Exception as e:
        if verbose >= 1:
            print(f"Failed to save MP4: {e}")
        return None


def save_visualization(
    frames: List[np.ndarray],
    output_path: str,
    output_format: OutputFormat,
    fps: int,
    verbose: int = 1,
) -> Optional[str]:
    """Save visualization frames to file.

    Args:
        frames: List of frames
        output_path: Output path
        output_format: Output format (GIF or MP4)
        fps: Frames per second
        verbose: Verbosity level

    Returns:
        Path to saved file or None if failed
    """
    if output_format == OutputFormat.MP4:
        return save_mp4(frames, output_path, fps, verbose)
    else:
        return save_gif(frames, output_path, fps, verbose)


def print_visualization_info(args: argparse.Namespace, episode_info: Dict[str, Any]):
    """Print visualization information.

    Args:
        args: Parsed CLI arguments
        episode_info: Episode information dictionary
    """
    print("\n" + "=" * 60)
    print("Visualization Complete")
    print("=" * 60)
    print(f"Output:       {args.output}")
    print(f"Frames:       {episode_info.get('frame_count', 0)}")
    print(f"FPS:          {args.fps}")
    print(f"Format:       {args.format or 'gif'}")
    print(f"Mode:         {args.mode}")
    print(f"Episode length: {episode_info.get('episode_length', 0)}")
    print(f"Total reward: {episode_info.get('total_reward', 0.0):.4f}")
    print("=" * 60)


def main():
    """Main visualization entry point."""
    args = parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    # Determine output format
    if args.format:
        output_format = OutputFormat(args.format)
    else:
        output_format = infer_format(args.output)

    # Determine deterministic mode
    deterministic = not args.stochastic if args.stochastic else args.deterministic

    # Visualization mode
    mode = VisualizationMode(args.mode)

    # Print visualization info
    if args.verbose >= 1:
        print("\n" + "=" * 60)
        print("SocialJax Visualization")
        print("=" * 60)
        print(f"Checkpoint:    {args.checkpoint}")
        print(f"Environment:   {args.env}")
        print(f"Output:        {args.output}")
        print(f"Format:        {output_format.value}")
        print(f"Mode:          {mode.value}")
        print(f"FPS:           {args.fps}")
        if args.num_frames:
            print(f"Max frames:    {args.num_frames}")
        print(f"Seed:          {args.seed}")
        print(f"Deterministic: {deterministic}")
        print("=" * 60)

    # Load checkpoint
    try:
        trainer, state = load_checkpoint(
            checkpoint_path=str(checkpoint_path),
            env_name=args.env,
            algorithm_name=args.algorithm,
            num_agents=args.num_agents,
            seed=args.seed,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run visualization
    start_time = time.time()

    try:
        frames, episode_info = run_visualization(
            trainer=trainer,
            state=state,
            num_frames=args.num_frames,
            deterministic=deterministic,
            mode=mode,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save visualization
    if not frames:
        print("Error: No frames captured")
        return 1

    output_file = save_visualization(
        frames=frames,
        output_path=args.output,
        output_format=output_format,
        fps=args.fps,
        verbose=args.verbose,
    )

    if not output_file:
        print("Error: Failed to save visualization")
        return 1

    elapsed = time.time() - start_time
    episode_info["elapsed_time"] = elapsed

    # Print summary
    if args.verbose >= 1:
        print_visualization_info(args, episode_info)
        print(f"\nVisualization time: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
