from __future__ import annotations

import math
from enum import IntEnum

import numpy as np


class Items(IntEnum):
    empty = 0
    wall = 1
    ore_wait = 2
    spawn_point = 3
    iron_ore = 4
    gold_ore = 5
    gold_partial = 6


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c, border=0):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle or on border
        return (u >= 0 - border) and (v >= 0 - border) and (u + v) < 1 + border

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img


def overlay_agent(tile_img, agent_color, orientation):
    """
    Draw an agent triangle with the given orientation onto tile_img in-place.
    """
    tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
    if orientation is not None:
        # rotate triangle by (orientation * 90 deg) - 90 deg
        tri_fn = rotate_fn(tri_fn, 0.5, 0.5, 0.5 * math.pi * orientation - 0.5 * math.pi)
    fill_coords(tile_img, tri_fn, agent_color)


def get_base_item_tile(item, tile_size, item_colors, item_tile_cache, items):
    """
    Cache only the base item tile here (no agent, highlight, etc.).
    """
    key = (item, tile_size)
    if key in item_tile_cache:
        return item_tile_cache[key]

    # Build tile base image
    tile_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    base_color = item_colors.get(item, (0, 0, 0))

    # Drawing for iron ore
    if item == items.iron_ore:
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1),
                    item_colors.get(items.ore_wait, (200, 200, 170)))
        fill_coords(tile_img, point_in_circle(0.5, 0.5, 0.5), base_color)

    elif item == items.gold_ore or item == items.gold_partial:
        # Also fill background tan
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1),
                    item_colors.get(items.ore_wait, (200, 200, 170)))
        # Gold circle
        fill_coords(tile_img, point_in_circle(0.5, 0.5, 0.5), base_color)

    else:
        fill_coords(tile_img, point_in_rect(0, 1, 0, 1), base_color)

    item_tile_cache[key] = tile_img
    return tile_img


def render_tile(
        item: int,
        tile_size: int,
        item_colors: dict,
        base_tile_cache: dict,  # cache for base items
        final_tile_cache: dict,  # cache for fully composed tiles
        items,
        highlight: bool = False,
        beam: bool = False,
        agent_id: int = -1,
        agent_dir: int = None,
        agent_colors=None,
):
    """
    Build or retrieve from cache the fully composed tile
    (base item + agent overlay + beam + highlight).
    """
    # Key that includes all features that change the final tile
    key = (item, agent_id, agent_dir, highlight, beam, tile_size)
    if key in final_tile_cache:
        return final_tile_cache[key]

    # Start with base item tile (cached)
    base_tile = get_base_item_tile(item, tile_size, item_colors, base_tile_cache, items)
    tile_img = base_tile.copy()

    # Overlay agent if present
    if agent_id >= 0 and agent_colors is not None:
        color = agent_colors[agent_id % len(agent_colors)]
        overlay_agent(tile_img, color, agent_dir)

    # If beam, highlight in gold
    if beam:
        highlight_img(tile_img, color=(220, 180, 20), alpha=0.4)

    # If partial-observation highlight
    if highlight:
        highlight_img(tile_img, color=(255, 255, 255), alpha=0.3)

    final_tile_cache[key] = tile_img
    return tile_img


def render_time(inner_t, outer_t, max_inner, max_outer, width_px):
    """
    Render a 2-row time bar (first row for inner time, second row for outer).
    """
    tile_height = 32
    img = np.zeros((2 * tile_height, width_px, 3), dtype=np.uint8)

    # Top row for inner time
    if max_inner > 0:
        tile_width_inner = width_px // max_inner
        for i in range(inner_t):
            y_min = 0
            y_max = tile_height
            x_min = i * tile_width_inner
            x_max = (i + 1) * tile_width_inner
            img[y_min:y_max, x_min:x_max, :] = 255  # white

    # Bottom row for outer time
    if max_outer > 0:
        tile_width_outer = width_px // max_outer
        for i in range(outer_t):
            y_min = tile_height
            y_max = 2 * tile_height
            x_min = i * tile_width_outer
            x_max = (i + 1) * tile_width_outer
            img[y_min:y_max, x_min:x_max, :] = 255  # white

    return img


#################################################################
# ---------------------------- JAX ---------------------------- #
#################################################################

import jax
import jax.numpy as jnp
from jax import lax


ITEM_COLORS_JAX = jnp.array([
    [220, 220, 220],  # empty=0
    [127, 127, 127],  # wall=1
    [200, 200, 170],  # ore_wait=2
    [180, 180, 250],  # spawn_point=3
    [139, 69, 19],    # iron_ore=4
    [180, 180, 40],   # gold_ore=5
    [190, 190, 80],   # gold_partial=6
], dtype=jnp.uint8)


def downsample_jax(img: jnp.ndarray, factor: int) -> jnp.ndarray:
    """
    Downsample an image along both spatial dimensions by `factor`.
    Equivalent to your NumPy version, but using JAX.

    img: shape (H, W, C)
    factor: integer
    returns: shape (H//factor, W//factor, C)
    """
    H, W, C = img.shape
    # (no Python asserts in jitted code, but you can keep them outside if you like)
    # Reshape and take mean in two steps
    img_reshaped = jnp.reshape(img, (H // factor, factor, W // factor, factor, C))
    # Mean over the factor dims: axis=3 then axis=1
    img_ds = img_reshaped.mean(axis=3).mean(axis=1)
    return img_ds


def pixel_coords(H, W):
    """
    Returns (xf, yf) each of shape (H, W), where
      xf[i,j] = (j + 0.5)/W,
      yf[i,j] = (i + 0.5)/H.
    """
    y_idx = jnp.arange(H)
    x_idx = jnp.arange(W)
    yy, xx = jnp.meshgrid(y_idx, x_idx, indexing='ij')
    # Convert to float in [0,1]
    xf = (xx + 0.5) / W
    yf = (yy + 0.5) / H
    return xf, yf


def point_in_circle_jax(H, W, cx, cy, r):
    """
    Return a (H, W) bool mask: True where (x - cx)^2 + (y - cy)^2 <= r^2
    (x,y) in normalized [0,1] coords.
    """
    xf, yf = pixel_coords(H, W)
    dist2 = (xf - cx)**2 + (yf - cy)**2
    return dist2 <= r*r


def point_in_rect_jax(H, W, xmin, xmax, ymin, ymax):
    """
    (H, W) bool mask: True where xmin <= x <= xmax and ymin <= y <= ymax
    in normalized coords.
    """
    xf, yf = pixel_coords(H, W)
    inside_x = (xf >= xmin) & (xf <= xmax)
    inside_y = (yf >= ymin) & (yf <= ymax)
    return inside_x & inside_y


def point_in_triangle_jax(H, W, a, b, c, border=0.0):
    """
    Returns a (H, W) boolean mask for points in or on the triangle
    defined by (ax,ay), (bx,by), (cx,cy).
    a,b,c are 2D tuples in normalized coords, e.g. (0.12,0.19).
    """
    # Convert to jnp arrays
    a = jnp.array(a, dtype=jnp.float32)
    b = jnp.array(b, dtype=jnp.float32)
    c = jnp.array(c, dtype=jnp.float32)

    xf, yf = pixel_coords(H, W)
    # Stack as (H,W,2)
    pts = jnp.stack([xf, yf], axis=-1)

    # We'll do the barycentric test for each pixel
    # v0 = c - a, v1 = b - a, v2 = p - a
    v0 = c - a
    v1 = b - a

    # Expand them so we can do vector ops vs. each pixel
    v0 = v0[None,None,:]  # shape (1,1,2)
    v1 = v1[None,None,:]  # shape (1,1,2)
    a_ = a[None,None,:]   # shape (1,1,2)
    v2 = pts - a_         # shape (H,W,2)

    dot00 = jnp.sum(v0*v0, axis=-1)  # shape(H,W)
    dot01 = jnp.sum(v0*v1, axis=-1)
    dot02 = jnp.sum(v0*v2, axis=-1)
    dot11 = jnp.sum(v1*v1, axis=-1)
    dot12 = jnp.sum(v1*v2, axis=-1)

    denom = (dot00 * dot11 - dot01*dot01)
    # For numerical safety, clamp denominator away from zero:
    denom = jnp.where(denom==0, 1e-12, denom)
    inv_denom = 1.0 / denom

    u = (dot11*dot02 - dot01*dot12)*inv_denom
    v = (dot00*dot12 - dot01*dot02)*inv_denom

    inside = (u >= 0 - border) & (v >= 0 - border) & ((u+v) <= 1 + border)
    return inside


def point_in_line_jax(H, W, x0, y0, x1, y1, r):
    """
    Returns (H, W) bool mask for points within distance r of the line segment (x0,y0)->(x1,y1).
    All coords are in normalized [0,1].
    """
    xf, yf = pixel_coords(H, W)
    # direction
    dirx = x1 - x0
    diry = y1 - y0
    dist_seg = jnp.sqrt(dirx*dirx + diry*diry)

    # For each pixel, we do a line-distance test
    # We'll define a local function to compute the distance from a point to the segment
    # but we'll do it in a vectorized manner:
    def segment_dist(px, py):
        # param: p0=(x0,y0), p1=(x1,y1), p=(px,py)
        # "project" p onto the line p0->p1, clamp t in [0,dist_seg], then measure distance
        # but we do it in normalized param space 0..1
        # t = dot((px-x0, py-y0), (dirx,diry)) / dist_seg
        vx = px - x0
        vy = py - y0
        dotv = vx*dirx + vy*diry
        t = dotv / (dist_seg*dist_seg)  # normalized [0..1] if inside segment
        t_clamped = jnp.clip(t, 0., 1.)
        # closest point on segment
        cx = x0 + t_clamped * dirx
        cy = y0 + t_clamped * diry
        # distance
        dx = px - cx
        dy = py - cy
        return jnp.sqrt(dx*dx + dy*dy)

    dist_func = jax.vmap(jax.vmap(segment_dist, in_axes=(0,None)), in_axes=(None,0))
    # dist_func expects (px, py) to be broadcast. We'll do:
    dists = dist_func(xf, yf)
    return dists <= r


def fill_coords_jax(img: jnp.ndarray, mask: jnp.ndarray, color) -> jnp.ndarray:
    """
    Fill the pixels of `img` (shape (H,W,3)) where `mask` (shape (H,W)) is True
    with the given color (3,). Returns a new jnp array.
    """
    color_arr = jnp.array(color, dtype=jnp.uint8)
    # shape(1,1,3) for broadcasting
    color_arr = color_arr[None, None, :]
    # Where mask is True, use color, else old pixel
    return jnp.where(mask[..., None], color_arr, img)


def rotate_mask_jax(mask: jnp.ndarray, cx: float, cy: float, theta: float) -> jnp.ndarray:
    """
    Rotates the 'True' region of mask by -theta around (cx, cy),
    in normalized coordinates.
    mask shape: (H, W)
    """
    H, W = mask.shape
    xf, yf = pixel_coords(H, W)  # shape(H, W)

    # We'll apply the inverse rotation to (xf,yf) to see
    # if that 'landed' in a True region in the original (unrotated) mask.
    # So we do the opposite transform on each pixel's location to find
    # where it maps back in the original mask.

    # Shift
    x2 = xf - cx
    y2 = yf - cy
    # Rotate by +theta to 'undo' the negative rotation we used in your code
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x3 = cx + x2*cos_t - y2*sin_t
    y3 = cy + y2*cos_t + x2*sin_t

    # Now we see if (x3,y3) was True in the original mask, i.e.
    # we need to see if that coordinate maps to a valid pixel (i0,j0)
    # in the original. We'll do a “reverse pixel mapping” if needed.
    # Or simpler: define a function that said “(x,y) => True if inside shape,”
    # so we can re-run it.
    # If your original mask is a static shape, you'd need to do an interpolation approach.

    # Usually, we do the entire shape in continuous coords. That is more advanced
    # and depends on how you define fin.

    # If your approach is to keep the same discrete mask, you'd do a nearest-neighbor or
    # bilinear sampling from the original 'mask'. For example:

    # Convert x3,y3 in [0,1] => pixel coords
    i3 = (y3 * H) - 0.5
    j3 = (x3 * W) - 0.5

    # Round to nearest pixel
    i3_rounded = jnp.round(i3).astype(jnp.int32)
    j3_rounded = jnp.round(j3).astype(jnp.int32)

    # Check bounds
    in_bounds = (
            (i3_rounded >= 0) & (i3_rounded < H) &
            (j3_rounded >= 0) & (j3_rounded < W)
    )
    # Where in-bounds, sample original mask
    old_vals = jnp.where(
        in_bounds,
        mask[i3_rounded, j3_rounded],  # gather
        False
    )
    return old_vals


def alpha_blend(tile, color, alpha=0.3):
    """
    tile, color are jnp arrays of shape (..., 3).
    alpha-blend in place: new_tile = tile + alpha * (color - tile).
    """
    # Convert tile to float for blending, then back to uint8
    tile_f = tile.astype(jnp.float32)
    color_f = color.astype(jnp.float32)
    blended = tile_f + alpha * (color_f - tile_f)
    blended = jnp.clip(blended, 0, 255)
    return blended.astype(jnp.uint8)


def highlight_jax(tile, color=(255, 255, 255), alpha=0.3):
    """
    Blend entire tile with a given color, factor alpha.
    """
    color_arr = jnp.array(color, dtype=jnp.uint8)
    return alpha_blend(tile, color_arr, alpha)


def circle_mask(H, W, center=(0.5, 0.5), radius=0.4):
    """
    Returns a (H, W) bool mask for a circle at (center) with radius in fraction coords.
    Example: if H=tile_size, W=tile_size
    """
    ys = jnp.linspace(0, 1, H)[:, None]
    xs = jnp.linspace(0, 1, W)[None, :]
    dist = jnp.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    return dist < radius


def rectangle_mask(H, W, x0, x1, y0, y1):
    """
    Returns (H, W) bool mask for rectangle [x0,x1] x [y0,y1] in normalized coords [0,1].
    E.g. x0=0.0, x1=1.0 => entire width, y0=0.0, y1=0.03 => top bar, etc.
    """
    ys = jnp.linspace(0, 1, H)[:, None]
    xs = jnp.linspace(0, 1, W)[None, :]
    in_x = (xs >= x0) & (xs <= x1)
    in_y = (ys >= y0) & (ys <= y1)
    return in_x & in_y


def create_base_item_tile_jax(item, tile_size, item_colors, base_tile_cache):
    """
    Return a jax array (tile_size, tile_size, 3) representing the base color or shape.
    """
    key = (item, tile_size)
    if key in base_tile_cache:
        return base_tile_cache[key]

    # Start with a blank tile
    tile = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.uint8)

    # Branch function for iron_ore
    def fill_iron_ore(_):
        tile_iron = fill_coords_jax(
            tile,
            jnp.ones((tile_size, tile_size), dtype=bool),
            jnp.array(item_colors.get(Items.ore_wait, (200, 200, 170)), dtype=jnp.uint8)
        )
        mask = circle_mask(tile_size, tile_size, radius=0.5)
        iron_color = jnp.array(item_colors.get(Items.iron_ore, (0, 0, 0)), dtype=jnp.uint8)
        return fill_coords_jax(tile_iron, mask, iron_color)

    def fill_gold_ore(_):
        # Tan background
        tile_bg = fill_coords_jax(
            tile,
            jnp.ones((tile_size, tile_size), dtype=bool),
            jnp.array(item_colors.get(Items.ore_wait, (200, 200, 170)), dtype=jnp.uint8)
        )
        # Gold circle
        mask = circle_mask(tile_size, tile_size, radius=0.5)
        gold_color_idx = jnp.clip(item, 0, 6)  # ensures valid index for ITEM_COLORS_JAX
        gold_color = ITEM_COLORS_JAX[gold_color_idx]
        return fill_coords_jax(tile_bg, mask, gold_color)

    # Branch function for all other items
    def fill_other(_):
        idx = jnp.clip(item, 0, 6)
        color = ITEM_COLORS_JAX[idx]
        mask = jnp.ones((tile_size, tile_size), dtype=bool)
        return fill_coords_jax(tile, mask, color)

    # Use jax.lax.cond to avoid Python if-check on a Traced bool
    tile_out = jax.lax.cond(
        jnp.equal(item, Items.iron_ore),
        fill_iron_ore,
        lambda _: jax.lax.cond(
            (item == Items.gold_ore) | (item == Items.gold_partial),
            fill_gold_ore,
            fill_other,
            operand=None
        ),
        operand=None
    )

    base_tile_cache[key] = tile_out
    return tile_out


def overlay_agent_jax(tile: jnp.ndarray, agent_color: jnp.ndarray, orientation: int) -> jnp.ndarray:
    """
    Draw an agent triangle with the given orientation onto the tile (H, W, 3).
    Uses a triangle mask centered in the tile and rotates it based on the orientation.

    Args:
        tile: The image tile to overlay on, shape (H, W, 3).
        agent_color: RGB color of the agent, shape (3,).
        orientation: Integer (0: up, 1: right, 2: down, 3: left).

    Returns:
        Updated tile with the agent overlay.
    """
    H, W, _ = tile.shape
    cx, cy = 0.5, 0.5  # Center of the tile in normalized coordinates

    # Define an unrotated triangle mask (pointing up by default)
    triangle_mask = point_in_triangle_jax(
        H, W,
        a=(cx, cy - 0.3),  # Top point (upward)
        b=(cx - 0.25, cy + 0.2),  # Bottom-left
        c=(cx + 0.25, cy + 0.2),  # Bottom-right
    )

    # Rotate the triangle mask according to the orientation
    theta = -orientation * (jnp.pi / 2)  # 0, 90, 180, or 270 degrees
    rotated_mask = rotate_mask_jax(triangle_mask, cx, cy, theta)

    # Overlay the rotated triangle onto the tile
    updated_tile = fill_coords_jax(tile, rotated_mask, agent_color)
    return updated_tile


def render_tile_jax(
        item: jnp.int32,               # scalar item ID as a JAX array
        tile_size: int,
        item_colors: dict,            # (Python dict) maps item -> (R,G,B)
        base_tile_cache: dict,        # Python dict for caching base tile
        final_tile_cache: dict,       # Python dict for caching fully composed tiles
        highlight: jnp.bool_,         # JAX boolean or Python bool
        beam: jnp.bool_,              # JAX boolean or Python bool
        agent_id: jnp.int32,          # scalar agent index as a JAX array
        agent_dir: jnp.int32,         # orientation
        agent_colors: jnp.ndarray     # shape (N,3) in jnp.uint8 (a JAX array!)
) -> jnp.ndarray:
    """
    Creates a fully-composed tile in JAX and caches it.
    (base item + possibly agent overlay + beam highlight + partial highlight).

    Args:
      item: e.g. jnp.int32(Items.ore_wait), ...
      tile_size: int
      item_colors: Python dict { item_id -> (r,g,b) } used by create_base_item_tile_jax
      base_tile_cache: Python dict for base tiles
      final_tile_cache: Python dict for final tiles
      highlight, beam: booleans (could be jnp.bool_ or Python bool)
      agent_id: jnp.int32 scalar (>=0 means agent present)
      agent_dir: jnp.int32 orientation
      agent_colors: jnp array of shape (num_agents,3) in uint8

    Returns:
      jnp.uint8 array of shape (tile_size, tile_size, 3)
    """

    # Because item, agent_id, etc. are JAX arrays (possibly traced), we cannot
    # do a Python dictionary lookup with them directly as the key. Instead,
    # we can do partial or no caching if these values are truly dynamic.
    # Below attempts to convert them to Python ints if they are static:
    try:
        item_int = int(item)
        agent_id_int = int(agent_id)
        highlight_bool = bool(highlight)
        beam_bool = bool(beam)
        agent_dir_int = int(agent_dir)
        key = (item_int, agent_id_int, agent_dir_int, highlight_bool, beam_bool, tile_size)
    except:
        # If they're not static, skip caching or use a fallback key
        key = None

    # If we have a valid key and it's in final_tile_cache, reuse
    if key is not None and key in final_tile_cache:
        return final_tile_cache[key]

    # 1) Build the base tile
    tile = create_base_item_tile_jax(item, tile_size, item_colors, base_tile_cache)

    # 2) Possibly overlay agent. We do a jax.lax.cond if agent_id >= 0
    #    'agent_id >= 0' is a JAX boolean expression => we can't do normal `if`.
    def overlay_agent_fn(tile_in):
        # agent_id % num_agents
        mod_id = lax.rem(agent_id, agent_colors.shape[0])
        color = agent_colors[mod_id]  # shape (3,)
        return overlay_agent_jax(tile_in, color, agent_dir)

    def no_agent_fn(tile_in):
        return tile_in

    tile = lax.cond(
        agent_id >= 0,      # jnp.bool_ expression
        overlay_agent_fn,
        no_agent_fn,
        operand=tile
    )

    # 3) Possibly highlight beam
    def do_beam(tile_in):
        return highlight_jax(tile_in, color=(20, 180, 200), alpha=0.2)
    def no_beam(tile_in):
        return tile_in

    tile = lax.cond(
        beam,
        do_beam,
        no_beam,
        operand=tile
    )

    # 4) Possibly highlight partial observation
    def do_highlight(tile_in):
        return highlight_jax(tile_in, color=(255,255,255), alpha=0.3)
    def no_highlight(tile_in):
        return tile_in

    tile = lax.cond(
        highlight,
        do_highlight,
        no_highlight,
        operand=tile
    )

    # Cache final tile if we have a valid key
    if key is not None:
        final_tile_cache[key] = tile

    return tile


def render_time_jax(inner_t, outer_t, max_inner, max_outer, width_px):
    """
    JAX-based time bar. Returns (2*32, width_px, 3) jnp.uint8
    """
    tile_height = 32
    bar = jnp.zeros((2 * tile_height, width_px, 3), dtype=jnp.uint8)

    # For simplicity, we'll do a naive approach:
    # Fill columns for each 't' with white. This can be done with a
    # scatter update or a loop using `lax.fori_loop`.

    def fill_bar(arry, start_y, end_y, start_x, end_x):
        # jnp.where-based approach:
        mask_y = (jnp.arange(arry.shape[0])[:, None] >= start_y) & (jnp.arange(arry.shape[0])[:, None] < end_y)
        mask_x = (jnp.arange(arry.shape[1])[None, :] >= start_x) & (jnp.arange(arry.shape[1])[None, :] < end_x)
        mask = mask_y & mask_x
        return jnp.where(mask[..., None], jnp.array([255, 255, 255], dtype=jnp.uint8), arry)

    # Fill top row (inner time)
    if max_inner > 0:
        tile_width_inner = width_px // max_inner

        def body_inner(i, arr):
            # fill columns from x=i*tile_width_inner to x=(i+1)*tile_width_inner
            x_min = i * tile_width_inner
            x_max = (i + 1) * tile_width_inner
            arr_new = fill_bar(arr, 0, tile_height, x_min, x_max)
            return arr_new

        bar = jax.lax.fori_loop(0, inner_t, body_inner, bar)

    # Fill bottom row (outer time)
    if max_outer > 0:
        tile_width_outer = width_px // max_outer

        def body_outer(i, arr):
            x_min = i * tile_width_outer
            x_max = (i + 1) * tile_width_outer
            arr_new = fill_bar(arr, tile_height, 2 * tile_height, x_min, x_max)
            return arr_new

        bar = jax.lax.fori_loop(0, outer_t, body_outer, bar)

    return bar


def render_jax(state, tile_size, item_colors, agent_colors, base_tile_cache, final_tile_cache, highlight_mask, beam_mask):
    """
    Render the entire grid in JAX, returning a jnp.uint8 array of shape
    (rows*tile_size, cols*tile_size, 3).
    """
    rows, cols = state.grid.shape

    # For each cell, we need (item, agent_id, agent_dir, highlight, beam).
    # We can build agent_map, orientation_map as jax arrays too.
    # Then we vmap over row & col. We'll do a double-vmap approach.

    # shape (N,)
    agent_rows = state.agent_locs[:, 0]
    agent_cols = state.agent_locs[:, 1]
    agent_dirs = state.agent_locs[:, 2]

    # Build agent_map in JAX: for each (row, col), which agent (if any) occupies?
    # This might be done with a scatter:
    #   - Start with -1 array
    #   - scatter indices=agent_rows, agent_cols, updates=jnp.arange(num_agents)
    # Or you can store occupant_grid in state. For demonstration, let's do scatter:
    num_agents = agent_rows.shape[0]
    occupant_init = -1 * jnp.ones((rows, cols), dtype=jnp.int32)
    occupant_map = occupant_init.at[agent_rows, agent_cols].set(jnp.arange(num_agents, dtype=jnp.int32))

    # Similarly orientation_map:
    orient_init = -1 * jnp.ones((rows, cols), dtype=jnp.int32)
    orientation_map = orient_init.at[agent_rows, agent_cols].set(agent_dirs)

    def render_cell(rr, cc):
        """
        Return a (tile_size, tile_size, 3) jax array for cell (rr, cc).
        """
        item = state.grid[rr, cc]
        a_id = occupant_map[rr, cc]
        a_dir = jnp.where(a_id < 0, -1, orientation_map[rr, cc])
        # highlight, beam are bool
        hl = highlight_mask[rr, cc]
        bm = beam_mask[rr, cc]

        return render_tile_jax(
            item=item,
            tile_size=tile_size,
            item_colors=item_colors,
            base_tile_cache=base_tile_cache,
            final_tile_cache=final_tile_cache,
            highlight=hl,
            beam=bm,
            agent_id=a_id,
            agent_dir=a_dir,
            agent_colors=agent_colors
        )

    # Now we do vmap over rows, and inside that vmap over cols.
    # We'll get an array of shape (rows, cols, tile_size, tile_size, 3).
    render_row = jax.vmap(
        lambda rr: jax.vmap(
            lambda cc: render_cell(rr, cc)
        )(jnp.arange(cols))
    )(jnp.arange(rows))

    # render_row has shape (rows, cols, tile_size, tile_size, 3)

    # We need to stitch these tiles into a single big image.
    # A typical approach is:
    #  - first combine along columns within each row
    #  - then stack the rows.

    # Combine each row of tiles horizontally: shape (rows, tile_size, cols*tile_size, 3)
    def combine_row(tiles_4d):
        # tiles_4d is shape (cols, tile_size, tile_size, 3)
        # We want shape (tile_size, cols*tile_size, 3)
        return jnp.concatenate(tiles_4d, axis=1)  # concat along width

    row_images = jax.vmap(combine_row)(render_row)
    # row_images: shape (rows, tile_size, cols*tile_size, 3)

    # Now stack rows vertically
    full_image = jnp.concatenate(row_images, axis=0)
    # shape (rows*tile_size, cols*tile_size, 3)

    return full_image


def render_time_jax(inner_t, outer_t, max_inner, max_outer, width_px):
    """
    JAX-based time bar. Returns (2*32, width_px, 3) jnp.uint8
    """
    tile_height = 32
    bar = jnp.zeros((2 * tile_height, width_px, 3), dtype=jnp.uint8)

    # For simplicity, we'll do a naive approach:
    # Fill columns for each 't' with white. This can be done with a
    # scatter update or a loop using `lax.fori_loop`.

    def fill_bar(arry, start_y, end_y, start_x, end_x):
        # jnp.where-based approach:
        mask_y = (jnp.arange(arry.shape[0])[:, None] >= start_y) & (jnp.arange(arry.shape[0])[:, None] < end_y)
        mask_x = (jnp.arange(arry.shape[1])[None, :] >= start_x) & (jnp.arange(arry.shape[1])[None, :] < end_x)
        mask = mask_y & mask_x
        return jnp.where(mask[..., None], jnp.array([255, 255, 255], dtype=jnp.uint8), arry)

    # Fill top row (inner time)
    if max_inner > 0:
        tile_width_inner = width_px // max_inner

        def body_inner(i, arr):
            # fill columns from x=i*tile_width_inner to x=(i+1)*tile_width_inner
            x_min = i * tile_width_inner
            x_max = (i + 1) * tile_width_inner
            arr_new = fill_bar(arr, 0, tile_height, x_min, x_max)
            return arr_new

        bar = jax.lax.fori_loop(0, inner_t, body_inner, bar)

    # Fill bottom row (outer time)
    if max_outer > 0:
        tile_width_outer = width_px // max_outer

        def body_outer(i, arr):
            x_min = i * tile_width_outer
            x_max = (i + 1) * tile_width_outer
            arr_new = fill_bar(arr, tile_height, 2 * tile_height, x_min, x_max)
            return arr_new

        bar = jax.lax.fori_loop(0, outer_t, body_outer, bar)

    return bar