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
        (255, 150, 150),               # Light red (level 1)
        (255, 100, 100),               # Medium red (level 2)
        (220, 50, 50),                 # Dark red (level 3)
    ],
    'agents': [
        (100, 100, 255),               # Blue
        (255, 200, 100),               # Orange
        (150, 100, 255),               # Purple
        (100, 255, 200),               # Cyan
        (255, 255, 100),               # Yellow
        (255, 150, 200),               # Pink
        (100, 200, 100),               # Green
        (200, 150, 255),               # Lavender
    ]
}


def render_lb_foraging(state, field_size, cell_size=40, cumulative_rewards=None):
    """
    Render the LB Foraging environment state as an RGB image

    Args:
        state: Environment state object
        field_size: Size of the field (rows, cols) tuple
        cell_size: Size of each cell in pixels
        cumulative_rewards: Optional dict or list of cumulative rewards for each agent

    Returns:
        RGB numpy array of shape (H, W, 3)
    """
    # Handle field_size as tuple
    if isinstance(field_size, tuple):
        rows, cols = field_size
    else:
        rows = cols = field_size

    # Create image
    img_width = cols * cell_size
    img_height = rows * cell_size
    img = Image.new('RGB', (img_width, img_height), COLORS['background'])
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=int(cell_size * 0.4))
        font_small = ImageFont.truetype("arial.ttf", size=int(cell_size * 0.25))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw grid lines
    for i in range(rows + 1):
        # Horizontal lines
        draw.line([(0, i * cell_size), (img_width, i * cell_size)],
                 fill=COLORS['grid_line'], width=1)
    for j in range(cols + 1):
        # Vertical lines
        draw.line([(j * cell_size, 0), (j * cell_size, img_height)],
                 fill=COLORS['grid_line'], width=1)

    # Note: The original lb-foraging environment doesn't draw walls at borders
    # Grid cells are accessible from 0 to rows-1 and 0 to cols-1

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
    draw.text((10, img_height - 25), info_text, fill=(0, 0, 0), font=font_small)

    # Draw cumulative rewards if provided
    if cumulative_rewards is not None:
        # Add extra space below for rewards display
        reward_panel_height = 80
        new_img = Image.new('RGB', (img_width, img_height + reward_panel_height), COLORS['background'])
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        # Draw reward title
        title_text = "Cumulative Rewards:"
        draw.text((10, img_height + 10), title_text, fill=(0, 0, 0), font=font_small)

        # Draw each agent's reward with their color
        y_offset = img_height + 30
        num_agents = len(agent_pos)

        # Convert cumulative_rewards to list if it's a dict
        if isinstance(cumulative_rewards, dict):
            rewards_list = [cumulative_rewards.get(str(i), 0.0) for i in range(num_agents)]
        else:
            rewards_list = cumulative_rewards

        # Calculate layout: 3 agents per row
        agents_per_row = 3
        x_spacing = img_width // agents_per_row

        for i in range(num_agents):
            reward_val = rewards_list[i] if i < len(rewards_list) else 0.0
            color_idx = i % len(COLORS['agents'])
            agent_color = COLORS['agents'][color_idx]

            # Calculate position in grid
            row_idx = i // agents_per_row
            col_idx = i % agents_per_row

            x_pos = 10 + col_idx * x_spacing
            current_y = y_offset + row_idx * 20

            # Draw colored square indicator
            square_size = 12
            draw.rectangle(
                [x_pos, current_y - 2, x_pos + square_size, current_y + square_size - 2],
                fill=agent_color,
                outline=(0, 0, 0),
                width=1
            )

            # Draw agent reward text
            reward_text = f"A{i}: {reward_val:.4f}"
            draw.text((x_pos + square_size + 5, current_y), reward_text, fill=(0, 0, 0), font=font_small)

        img = new_img

    # Convert to numpy array
    return np.array(img)
