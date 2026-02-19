"""SocialJax evaluation module.

This module provides tools for evaluating trained multi-agent policies,
including metrics computation, episode evaluation, and visualization.

Key components:
- EpisodeMetrics: Metrics from a single evaluation episode
- EvaluationMetrics: Aggregated metrics across multiple episodes
- Evaluator: Main class for running evaluation episodes
- Visualization utilities: GIF/MP4 generation

Example usage:
    ```python
    import socialjax
    from socialjax.evaluation import Evaluator, EvaluatorConfig

    # Create environment
    env = socialjax.make('clean_up', num_agents=7)

    # Create evaluator
    evaluator = Evaluator(
        env=env,
        algorithm=trained_algorithm,
        config=EvaluatorConfig(num_episodes=50)
    )

    # Run evaluation
    metrics = evaluator.evaluate()

    # Print results
    print(f"Mean return: {metrics.mean_return:.2f} +/- {metrics.std_return:.2f}")
    print(f"Cooperation rate: {metrics.cooperation_rate:.2%}")

    # Generate visualization
    from socialjax.evaluation import save_gif
    metrics, frames = evaluator.evaluate_with_frames()
    save_gif(frames, "evaluation.gif", fps=10)
    ```
"""

# Metrics
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

# Evaluator
from socialjax.evaluation.evaluator import (
    EvaluatorConfig,
    Evaluator,
    save_evaluation_results,
    load_evaluation_results,
    print_evaluation_summary,
)

# Visualization
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

__all__ = [
    # Metrics
    "EpisodeMetrics",
    "EvaluationMetrics",
    "compute_episode_return",
    "compute_agent_returns",
    "compute_cooperation_rate",
    "compute_gini_coefficient",
    "compute_social_welfare",
    "aggregate_episode_metrics",
    "identify_cooperative_action",
    "compute_metrics_from_episodes",
    # Evaluator
    "EvaluatorConfig",
    "Evaluator",
    "save_evaluation_results",
    "load_evaluation_results",
    "print_evaluation_summary",
    # Visualization
    "OutputFormat",
    "VisualizationMode",
    "infer_format",
    "normalize_frame",
    "add_text_overlay",
    "apply_visualization_mode",
    "save_gif",
    "save_mp4",
    "save_visualization",
    "create_comparison_gif",
    "create_episode_grid",
    "resize_frames",
    "get_frame_statistics",
]
