"""
Rendering utilities for Level-Based Foraging environment
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Color scheme
COLORS = {
    'background': (240, 240, 240),    # Light gray
    'wall': (60, 60, 60),              # Dark gray
    'grid_line': (200, 200, 200),      # Light gray for grid
    'food': [
        (100, 200, 100),               # Light green (level 1)
        (50, 180, 50),                 # Medium green (level 2)
        (0, 150, 0),                   # Dark green (level 3)
    ],
    'agents': [
        (255, 100, 100),               # Red
        (100, 100, 255),               # Blue
        (255, 200, 100),               # Orange
        (200, 100, 255),               # Purple
        (100, 255, 255),               # Cyan
        (255, 255, 100),               # Yellow
        (255, 150, 200),               # Pink
        (150, 255, 150),               # Light green
    ]
}


def render_lb_foraging(state, grid_size, cell_size=40):
    """
    Render the LB Foraging environment state as an RGB image

    Args:
        state: Environment state object
        grid_size: Size of the grid
        cell_size: Size of each cell in pixels

    Returns:
        RGB numpy array of shape (H, W, 3)
    """
    # Create image
    img_size = grid_size * cell_size
    img = Image.new('RGB', (img_size, img_size), COLORS['background'])
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=int(cell_size * 0.4))
        font_small = ImageFont.truetype("arial.ttf", size=int(cell_size * 0.25))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw grid lines
    for i in range(grid_size + 1):
        # Vertical lines
        draw.line([(i * cell_size, 0), (i * cell_size, img_size)],
                 fill=COLORS['grid_line'], width=1)
        # Horizontal lines
        draw.line([(0, i * cell_size), (img_size, i * cell_size)],
                 fill=COLORS['grid_line'], width=1)

    # Draw walls (borders)
    wall_width = 3
    # Top wall
    draw.rectangle([0, 0, img_size, cell_size],
                  fill=COLORS['wall'], outline=COLORS['wall'])
    # Bottom wall
    draw.rectangle([0, img_size - cell_size, img_size, img_size],
                  fill=COLORS['wall'], outline=COLORS['wall'])
    # Left wall
    draw.rectangle([0, 0, cell_size, img_size],
                  fill=COLORS['wall'], outline=COLORS['wall'])
    # Right wall
    draw.rectangle([img_size - cell_size, 0, img_size, img_size],
                  fill=COLORS['wall'], outline=COLORS['wall'])

    # Convert state arrays to numpy for easier indexing
    food_pos = np.array(state.food_pos)
    food_levels = np.array(state.food_levels)
    food_active = np.array(state.food_active)
    agent_pos = np.array(state.agent_pos)
    agent_levels = np.array(state.agent_levels)

    # Draw food items
    for i in range(len(food_pos)):
        if food_active[i]:
            x, y = food_pos[i]
            level = food_levels[i]

            # Calculate cell center
            cx = y * cell_size + cell_size // 2
            cy = x * cell_size + cell_size // 2

            # Draw food as a circle
            radius = cell_size * 0.3
            color_idx = min(int(level) - 1, len(COLORS['food']) - 1)
            food_color = COLORS['food'][color_idx]

            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                        fill=food_color, outline=(0, 0, 0), width=2)

            # Draw level number
            text = str(int(level))
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((cx - text_width // 2, cy - text_height // 2),
                     text, fill=(255, 255, 255), font=font)

    # Draw agents
    for i in range(len(agent_pos)):
        x, y = agent_pos[i]
        level = agent_levels[i]

        # Calculate cell center
        cx = y * cell_size + cell_size // 2
        cy = x * cell_size + cell_size // 2

        # Draw agent as a square
        size = cell_size * 0.35
        color_idx = i % len(COLORS['agents'])
        agent_color = COLORS['agents'][color_idx]

        draw.rectangle([cx - size, cy - size, cx + size, cy + size],
                      fill=agent_color, outline=(0, 0, 0), width=2)

        # Draw agent ID
        text = f"A{i}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((cx - text_width // 2, cy - text_height // 2 - size * 0.3),
                 text, fill=(255, 255, 255), font=font_small)

        # Draw level below
        level_text = f"L{int(level)}"
        bbox = draw.textbbox((0, 0), level_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((cx - text_width // 2, cy - text_height // 2 + size * 0.3),
                 level_text, fill=(255, 255, 255), font=font_small)

    # Draw info text at bottom
    info_text = f"Step: {int(state.step_count)}"
    draw.text((10, img_size - 25), info_text, fill=(0, 0, 0), font=font_small)

    # Convert to numpy array
    return np.array(img)
