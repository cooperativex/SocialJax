"""Harvest Common (Closed) Environment.

A variant of Harvest Common (Open) with walls dividing the space,
creating separate regions for agents to harvest apples.
"""

from .harvest_open import Harvest_open


# Default map for closed variant - has walls dividing the space
CLOSED_MAP = [
    "AAA    A  WW  A    AAA",
    "AA    AAA WW AAA    AA",
    "A    AAAAAWWAAAAA    A",
    "      AAA WW AAA      ",
    "       A  WW  A       ",
    "  A       WW       A  ",
    " AAA  Q   WW   Q  AAA ",
    "AAAAA     WW     AAAAA",
    " AAA  WWWWWWWWWW  AAA ",
    "  A       WW       A  ",
    "WWWWWWWWW WW WWWWWWWWW",
    "          WW          ",
    "  WWWWWWWWWWWWWWWWWW  ",
    "  P P P P P P P P P   ",
    " P P P P P P P P P P  ",
]


class Harvest_closed(Harvest_open):
    """
    Harvest Common (Closed) environment.
    
    This is a variant of Harvest Common (Open) with walls dividing the space.
    The walls create separate regions, making cooperation more challenging
    as agents cannot easily move between regions.
    
    Inherits all functionality from Harvest_open, only changing the default map.
    """
    
    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=7,
        shared_rewards=True,
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        enable_smooth_rewards=False,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        interest=False,
        s_interest=0.5,
        s_interest_schedule=None,
        s_interest_change_every=30000000,
        grid_size=(16, 22),
        jit=True,
        obs_size=11,
        cnn=True,
        map_ASCII=None,
    ):
        """Initialize Harvest Closed environment.
        
        Args:
            All args same as Harvest_open, except:
            map_ASCII: If None, uses the default closed map with walls
        """
        # Use closed map by default if not specified
        if map_ASCII is None:
            map_ASCII = CLOSED_MAP
            
        super().__init__(
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            num_agents=num_agents,
            shared_rewards=shared_rewards,
            inequity_aversion=inequity_aversion,
            inequity_aversion_target_agents=inequity_aversion_target_agents,
            inequity_aversion_alpha=inequity_aversion_alpha,
            inequity_aversion_beta=inequity_aversion_beta,
            enable_smooth_rewards=enable_smooth_rewards,
            svo=svo,
            svo_target_agents=svo_target_agents,
            svo_w=svo_w,
            svo_ideal_angle_degrees=svo_ideal_angle_degrees,
            interest=interest,
            s_interest=s_interest,
            s_interest_schedule=s_interest_schedule,
            s_interest_change_every=s_interest_change_every,
            grid_size=grid_size,
            jit=jit,
            obs_size=obs_size,
            cnn=cnn,
            map_ASCII=map_ASCII,
        )
