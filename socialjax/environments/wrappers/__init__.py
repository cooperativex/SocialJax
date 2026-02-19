"""Environment wrappers for SocialJax.

This module provides various wrappers for augmenting environment behavior:
- BaseWrapper: Base class for all wrappers
- NormalizationWrapper: Normalize observations and rewards
- FrameStackWrapper: Stack multiple frames for temporal information
- TimeLimitWrapper: Enforce maximum episode length
"""

from socialjax.environments.wrappers.base_wrapper import BaseWrapper
from socialjax.environments.wrappers.normalization import NormalizationWrapper
from socialjax.environments.wrappers.frame_stack import FrameStackWrapper
from socialjax.environments.wrappers.time_limit import TimeLimitWrapper

__all__ = [
    "BaseWrapper",
    "NormalizationWrapper",
    "FrameStackWrapper",
    "TimeLimitWrapper",
]
