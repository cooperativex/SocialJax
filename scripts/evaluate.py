#!/usr/bin/env python
"""Unified evaluation script for SocialJax.

This script provides a command-line interface for evaluating trained RL
policies on multi-agent environments.

Example usage:
    # Evaluate a checkpoint on an environment
    python scripts/evaluate.py --checkpoint checkpoints/ippo_coin_game/ippo_final --env coin_game

    # Evaluate with custom number of episodes
    python scripts/evaluate.py --checkpoint X --env clean_up --episodes 50

    # Generate evaluation GIF
    python scripts/evaluate.py --checkpoint X --env coin_game --render --output eval.gif

    # Evaluate with specific seed for reproducibility
    python scripts/evaluate.py --checkpoint X --env coin_game --seed 42
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SocialJax policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --checkpoint checkpoints/ippo_final --env coin_game
  python scripts/evaluate.py --checkpoint X --env clean_up --episodes 50
  python scripts/evaluate.py --checkpoint X --env coin_game --render --output eval.gif
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

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation (default: 42)",
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

    # Rendering
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for GIF/video when --render is set",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for rendered output (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to render per episode (default: 500)",
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

    # Output
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=normal, 2=debug (default: 1)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
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
            # Navigate nested config
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
        # Default number of agents by environment
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

    # Create trainer with minimal config
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


def run_evaluation(
    trainer: Trainer,
    state: Any,
    num_episodes: int = 10,
    deterministic: bool = True,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Run evaluation episodes.

    Args:
        trainer: Trainer instance
        state: Trainer state
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        verbose: Verbosity level

    Returns:
        Dictionary of evaluation results
    """
    if verbose >= 1:
        print(f"\nRunning {num_episodes} evaluation episodes...")
        print(f"Deterministic actions: {deterministic}")

    # Call the trainer's evaluate method
    eval_metrics = trainer.evaluate(
        state=state,
        num_episodes=num_episodes,
        deterministic=deterministic,
    )

    return eval_metrics


def run_evaluation_with_render(
    trainer: Trainer,
    state: Any,
    num_episodes: int = 10,
    deterministic: bool = True,
    max_frames: int = 500,
    fps: int = 10,
    output_path: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run evaluation with rendering.

    Args:
        trainer: Trainer instance
        state: Trainer state
        num_episodes: Number of episodes
        deterministic: Whether to use deterministic actions
        max_frames: Maximum frames per episode
        fps: Frames per second for output
        output_path: Path to save GIF/video
        verbose: Verbosity level

    Returns:
        Tuple of (evaluation_results, output_file_path)
    """
    import numpy as np

    env = trainer.env
    algorithm = trainer.algorithm

    all_returns = []
    all_lengths = []
    frames = []

    rng = jax.random.PRNGKey(state.algorithm_state.rng[0] if hasattr(state.algorithm_state.rng, '__getitem__') else 42)

    if verbose >= 1:
        print(f"\nRunning {num_episodes} evaluation episodes with rendering...")

    for episode in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)

        episode_return = 0.0
        episode_length = 0
        done = False
        episode_frames = []

        while not done and episode_length < max_frames:
            # Render current state
            if hasattr(env, 'render'):
                try:
                    frame = env.render(env_state)
                    if frame is not None:
                        episode_frames.append(frame)
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

            # Step environment
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, rewards, dones, info = env.step(step_rng, env_state, actions)

            # Accumulate rewards
            if isinstance(rewards, dict):
                episode_return += sum(rewards.values())
            else:
                episode_return += float(np.sum(rewards))

            episode_length += 1

            # Check done
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = bool(dones)

            obs = next_obs

        all_returns.append(episode_return)
        all_lengths.append(episode_length)
        frames.extend(episode_frames)

        if verbose >= 1:
            print(f"  Episode {episode + 1}/{num_episodes}: Return={episode_return:.2f}, Length={episode_length}")

    # Compute metrics
    results = {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "min_return": float(np.min(all_returns)),
        "max_return": float(np.max(all_returns)),
        "mean_length": float(np.mean(all_lengths)),
        "std_length": float(np.std(all_lengths)),
        "num_episodes": num_episodes,
        "episode_returns": [float(r) for r in all_returns],
        "episode_lengths": [int(l) for l in all_lengths],
    }

    # Save GIF if output path specified
    output_file = None
    if output_path and frames:
        output_file = save_gif(frames, output_path, fps, verbose)

    return results, output_file


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
        import io

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Normalize if needed
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


def print_results(results: Dict[str, Any], verbose: int = 1):
    """Print evaluation results.

    Args:
        results: Dictionary of evaluation results
        verbose: Verbosity level
    """
    if verbose < 1:
        return

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes:        {results['num_episodes']}")
    print(f"Mean Return:     {results['mean_return']:.4f} +/- {results['std_return']:.4f}")
    print(f"Min Return:      {results['min_return']:.4f}")
    print(f"Max Return:      {results['max_return']:.4f}")
    print(f"Mean Length:     {results['mean_length']:.1f} +/- {results['std_length']:.1f}")

    if 'episode_returns' in results and verbose >= 2:
        print(f"\nIndividual Returns:")
        for i, ret in enumerate(results['episode_returns']):
            print(f"  Episode {i + 1}: {ret:.4f}")

    print("=" * 60)


def save_results_json(results: Dict[str, Any], output_path: str, verbose: int = 1):
    """Save results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save JSON
        verbose: Verbosity level
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose >= 1:
        print(f"Saved results to: {output_path}")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    # Determine deterministic mode
    deterministic = not args.stochastic if args.stochastic else args.deterministic

    # Print evaluation info
    if args.verbose >= 1:
        print("\n" + "=" * 60)
        print("SocialJax Evaluation")
        print("=" * 60)
        print(f"Checkpoint:    {args.checkpoint}")
        print(f"Environment:   {args.env}")
        print(f"Episodes:      {args.episodes}")
        print(f"Seed:          {args.seed}")
        print(f"Deterministic: {deterministic}")
        if args.render:
            print(f"Render:        True")
            print(f"Max frames:    {args.max_frames}")
            print(f"FPS:           {args.fps}")
            if args.output:
                print(f"Output:        {args.output}")
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

    # Run evaluation
    start_time = time.time()

    try:
        if args.render:
            results, gif_path = run_evaluation_with_render(
                trainer=trainer,
                state=state,
                num_episodes=args.episodes,
                deterministic=deterministic,
                max_frames=args.max_frames,
                fps=args.fps,
                output_path=args.output,
                verbose=args.verbose,
            )
            if gif_path:
                results["gif_path"] = gif_path
        else:
            results = run_evaluation(
                trainer=trainer,
                state=state,
                num_episodes=args.episodes,
                deterministic=deterministic,
                verbose=args.verbose,
            )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["episodes_per_second"] = args.episodes / elapsed if elapsed > 0 else 0

    # Print results
    print_results(results, args.verbose)

    if args.verbose >= 1:
        print(f"\nEvaluation time: {elapsed:.2f}s ({results['episodes_per_second']:.2f} eps/s)")

    # Save results to JSON if requested
    if args.save_results:
        save_results_json(results, args.save_results, args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
