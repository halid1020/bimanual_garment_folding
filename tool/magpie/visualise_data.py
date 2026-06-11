"""
Actoris-Harena Trajectory Visualization Utility.

This script provides diagnostic visualization for robotic manipulation trajectories
stored in Zarr or HDF5 formats. It projects continuous 9D action vectors into
human-readable geometric overlays (arrows, gradient lines, points) and plots
semantic keypoints directly onto RGB observations.

It stacks the current temporal observation with its corresponding goal state 
to visually verify progression and alignment prior to VLA model training.

Dependencies:
    - numpy
    - opencv-python (cv2)
    - matplotlib
    - zarr (optional, preferred)
    - h5py (fallback)

Usage:
    python tool/magpie/visualise_data.py --dataset_path ./data/datasets/my_data
"""

import os
import argparse
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# Constants & Theming
# ==========================================
TEXT_Y_STEP: int = 35
TEXT_BG_ALPHA: float = 0.6

# OpenCV uses BGR color format
BLUE: Tuple[int, int, int] = (200, 50, 50)        # Left manipulator
RED: Tuple[int, int, int]  = (50, 50, 200)        # Right manipulator

# Semantic color mapping for action text backgrounds
PRIMITIVE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "norm-pixel-pick-and-place": (255, 180, 80),        # Orange
    "norm-pixel-dual-pick-and-place": (255, 180, 80),   # Orange
    "norm-pixel-single-pick-and-place": (255, 180, 80), # Orange
    "norm-pixel-pick-and-fling": (80, 200, 255),        # Cyan
    "no-operation": (255, 255, 255),                    # White
    "default": (255, 255, 255),                         # White
}

# ==========================================
# Action Configuration Schema
# ==========================================
PRIMITIVES: List[Dict[str, Any]] = [
    {"name": "norm-pixel-pick-and-fling", "dim": 4},
    {"name": "norm-pixel-dual-pick-and-place", "dim": 8},
    {"name": "norm-pixel-single-pick-and-place", "dim": 4},
    {"name": "no-operation", "dim": 0}
]

# ==========================================
# Core Drawing Utilities
# ==========================================

def norm_to_px(v: np.ndarray, W: int, H: int) -> Tuple[int, int]:
    """
    Converts normalized coordinates [-1, 1] to absolute pixel coordinates.

    Args:
        v (np.ndarray): A 2D vector or array slice representing [x_norm, y_norm].
        W (int): The width of the image in pixels.
        H (int): The height of the image in pixels.

    Returns:
        Tuple[int, int]: Absolute (x, y) pixel coordinates.
    """
    x = int((v[0] + 1) * 0.5 * W)
    y = int((v[1] + 1) * 0.5 * H)
    return x, y

def swap(p: Tuple[int, int]) -> Tuple[int, int]:
    """
    Swaps (x, y) coordinates to (y, x).
    OpenCV point format expects (x, y), but numpy indexing uses (row, col) / (y, x).
    
    Args:
        p (Tuple[int, int]): Input coordinates.
        
    Returns:
        Tuple[int, int]: Swapped coordinates.
    """
    return (p[1], p[0])

def draw_big_arrowhead(img: np.ndarray, p_from: Tuple[int, int], p_to: Tuple[int, int], color: Tuple[int, int, int], size: int = 28) -> None:
    """
    Draws a highly visible geometric arrowhead indicating directional vectors.

    Calculates the orthogonal vector to the direction of motion to construct
    the left and right vertices of the arrowhead triangle.

    Args:
        img (np.ndarray): The image canvas (modified in-place).
        p_from (Tuple[int, int]): Origin coordinate.
        p_to (Tuple[int, int]): Destination coordinate (where the tip goes).
        color (Tuple[int, int, int]): BGR color tuple.
        size (int): Size of the arrowhead base in pixels.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6

    # Unit direction vector
    ux, uy = dx / norm, dy / norm      
    # Orthogonal vector for arrowhead width
    px, py = -uy, ux                   

    tip = np.array([p_to[0], p_to[1]])
    left = np.array([
        p_to[0] - size * ux + size * 0.6 * px,
        p_to[1] - size * uy + size * 0.6 * py
    ])
    right = np.array([
        p_to[0] - size * ux - size * 0.6 * px,
        p_to[1] - size * uy - size * 0.6 * py
    ])

    pts = np.array([
        (int(tip[0]),   int(tip[1])),
        (int(left[0]),  int(left[1])),
        (int(right[0]), int(right[1]))
    ], dtype=np.int32)
    
    cv2.fillPoly(img, [pts], color)

def draw_colored_line(img: np.ndarray, p_start: Tuple[int, int], p_end: Tuple[int, int], cmap: int, thickness: int = 8, num_samples: int = 20) -> None:
    """
    Draws a trajectory line with a color gradient to indicate motion direction.

    Interpolates `num_samples` segments along the line, applying the colormap
    gradient progressively from start to end.

    Args:
        img (np.ndarray): The image canvas (modified in-place).
        p_start (Tuple[int, int]): Trajectory start point.
        p_end (Tuple[int, int]): Trajectory end point.
        cmap (int): OpenCV Colormap ID (e.g., cv2.COLORMAP_COOL).
        thickness (int): Stroke thickness in pixels.
        num_samples (int): Resolution of the gradient segments.
    """
    xs = np.linspace(p_start[0], p_end[0], num_samples).astype(int)
    ys = np.linspace(p_start[1], p_end[1], num_samples).astype(int)

    for i in range(1, num_samples):
        alpha = i / (num_samples - 1)
        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
        color = cv2.applyColorMap(value, cmap)[0, 0].tolist()
        cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color, thickness)

    # Attach final arrowhead at the destination using the terminal gradient color
    draw_big_arrowhead(img, (xs[-2], ys[-2]), (xs[-1], ys[-1]), color, size=30)

def draw_text_with_bg(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float = 1.0, thickness: int = 2) -> None:
    """
    Renders highly legible text onto the image by backing it with a semi-transparent rectangle.

    Args:
        img (np.ndarray): The image canvas (modified in-place).
        text (str): The string to display.
        org (Tuple[int, int]): Bottom-left origin point for the text bounding box.
        color (Tuple[int, int, int]): BGR color for the text stroke.
        scale (float): OpenCV font scale multiplier.
        thickness (int): Font stroke thickness.
    """
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    
    # Create semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, TEXT_BG_ALPHA, img, 1 - TEXT_BG_ALPHA, 0, img)
    
    # Draw actual text
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_keypoints(img: np.ndarray, keypoints_norm: Optional[np.ndarray]) -> None:
    """
    Projects and renders semantic keypoints onto the image using a distinct matplotlib colormap.

    Ignores padding values (assumed implicitly if keypoints_norm is None, though padding logic
    should be handled upstream if values are exactly [-1, -1]).

    Args:
        img (np.ndarray): The image canvas (modified in-place).
        keypoints_norm (Optional[np.ndarray]): Array of shape (K, 2) in normalized [-1, 1] space.
    """
    if keypoints_norm is None: 
        return
    
    H, W = img.shape[:2]
    keypoints_norm = keypoints_norm.reshape(-1, 2)
    num_kps = len(keypoints_norm)
    
    # Utilize tab20 for high-contrast, distinct mapping across up to 20 keypoints
    cmap = plt.cm.get_cmap('tab20')
    
    for i, kp in enumerate(keypoints_norm):
        y_norm, x_norm = kp
        
        # Skip dummy padding values generated during dataset merge
        if np.allclose(kp, [-1.0, -1.0]):
            continue
            
        x = int((x_norm + 1.0) / 2.0 * W)
        y = int((y_norm + 1.0) / 2.0 * H)
        
        # Convert Matplotlib RGBA (0-1) to OpenCV BGR (0-255)
        rgba = cmap(i / max(1, num_kps - 1))
        color_bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        
        cv2.circle(img, (x, y), 8, color_bgr, -1)        # Filled interior
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)   # High-contrast border

def draw_action(img: np.ndarray, prim_name: str, params: np.ndarray) -> None:
    """
    Maps geometric primitive parameters onto the 2D image space.
    
    Routes drawing logic based on the string primitive name extracted from the action vector.

    Args:
        img (np.ndarray): The image canvas (modified in-place).
        prim_name (str): Identifier for the action type (e.g., "norm-pixel-dual-pick-and-place").
        params (np.ndarray): Sliced continuous parameter vector excluding the primitive ID.
    """
    H, W = img.shape[:2]
    cmap_left  = cv2.COLORMAP_COOL
    cmap_right = cv2.COLORMAP_AUTUMN
    
    # Base colors derived from colormaps for points
    BLUE_MAP = cv2.applyColorMap(np.uint8([[[0]]]), cmap_left)[0, 0].tolist()
    RED_MAP = cv2.applyColorMap(np.uint8([[[0]]]), cmap_right)[0, 0].tolist()

    if "pick-and-place" in prim_name:
        if len(params) >= 8:  # Dual Manipulator Pick & Place
            left_pick  = norm_to_px(params[:2], W, H)
            right_pick = norm_to_px(params[2:4], W, H)
            left_place = norm_to_px(params[4:6], W, H)
            right_place= norm_to_px(params[6:8], W, H)
            
            # Enforce spatial consistency: Left manipulator should always be higher in Y (lower numerically in image space)
            if left_pick[1] > right_pick[1]: 
                left_pick, right_pick = right_pick, left_pick
                left_place, right_place = right_place, left_place

            draw_colored_line(img, swap(left_pick), swap(left_place), cmap_left, thickness=6)
            draw_colored_line(img, swap(right_pick), swap(right_place), cmap_right, thickness=6)
            cv2.circle(img, swap(left_pick), 8, BLUE_MAP, 3)
            cv2.circle(img, swap(right_pick), 8, RED_MAP, 3)
            
        elif len(params) >= 4:  # Single Manipulator Pick & Place
            pick = norm_to_px(params[:2], W, H)
            place = norm_to_px(params[2:4], W, H)
            
            draw_colored_line(img, swap(pick), swap(place), cmap_left, thickness=6)
            cv2.circle(img, swap(pick), 8, BLUE_MAP, 3)

    elif prim_name == "norm-pixel-pick-and-fling":
        left_pick = norm_to_px(params[:2], W, H)
        right_pick = norm_to_px(params[2:4], W, H)
        
        if left_pick[1] > right_pick[1]: 
            left_pick, right_pick = right_pick, left_pick
            
        cv2.circle(img, swap(left_pick), 8, BLUE_MAP, 3)
        cv2.circle(img, swap(right_pick), 8, RED_MAP, 3)
        
        # Fling relies on an arbitrary pixel offset to visualize vertical lift
        fling_offset = 60
        draw_big_arrowhead(img, swap(left_pick), (swap(left_pick)[0], swap(left_pick)[1] - fling_offset), BLUE_MAP, size=20)
        draw_big_arrowhead(img, swap(right_pick), (swap(right_pick)[0], swap(right_pick)[1] - fling_offset), RED_MAP, size=20)

# ==========================================
# Dataset Parsing Logic
# ==========================================

def decode_action_vector(action_vec: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Decodes the 9D continuous action array back into discrete primitive semantics.

    Args:
        action_vec (np.ndarray): The continuous action vector from the dataset.

    Returns:
        Tuple[str, np.ndarray]: The primitive name and its relevant parameter slice.
    """
    K = len(PRIMITIVES)
    prim_val = action_vec[0]
    
    # Reverse the continuous normalization mapping: prim_act = (1.0 * (prim_id + 0.5) / K * 2 - 1)
    prim_idx = int(np.clip(((prim_val + 1) / 2) * K - 1e-6, 0, K - 1))
    
    prim_info = PRIMITIVES[prim_idx]
    prim_name = prim_info["name"]
    params = action_vec[1 : 1 + prim_info["dim"]]
    
    return prim_name, params

def load_dataset(dataset_dir: str) -> Any:
    """
    Instantiates the storage reader based on directory structure.
    Prefers Zarr for chunked data access, falls back to HDF5.

    Args:
        dataset_dir (str): Path to the dataset.

    Returns:
        Any: Zarr group root or h5py File object.
        
    Raises:
        RuntimeError: If neither Zarr nor h5py can parse the target path.
    """
    if dataset_dir.endswith('.zarr') or os.path.isdir(dataset_dir):
        try:
            import zarr
            return zarr.open(dataset_dir, mode='r')
        except ImportError: 
            pass
            
    try:
        import h5py
        return h5py.File(dataset_dir, 'r')
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset. Check path or permissions: {e}")

def format_image(img_data: np.ndarray, img_size: int) -> np.ndarray:
    """
    Prepares a raw dataset observation array for OpenCV rendering.

    Converts (C,H,W) to (H,W,C), handles float [0, 1] to uint8 [0, 255]
    scaling, converts RGB to OpenCV's BGR, and resizes to target dimension.

    Args:
        img_data (np.ndarray): Raw image array slice from dataset.
        img_size (int): Target width and height in pixels.

    Returns:
        np.ndarray: Formatted BGR image array ready for cv2 operations.
    """
    img = img_data.copy()
    
    if img.shape[0] == 3: # Handle Channel-First schemas
        img = np.transpose(img, (1, 2, 0))
        
    if img.max() <= 1.0:  # Handle float normalizations
        img = (img * 255).astype(np.uint8)
        
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return img

# ==========================================
# Main Execution & File I/O
# ==========================================

def main() -> None:
    """
    Main visualization pipeline.
    Parses CLI args, loads the dataset, chunks episodes to manage memory, 
    compiles visual grids, and writes them to disk.
    """
    parser = argparse.ArgumentParser(description="Visualize saved trajectory dataset geometries and keypoints.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the .zarr directory or .hdf5 dataset")
    parser.add_argument('--num_episodes', type=int, default=100, help="Maximum number of episodes to visualize overall")
    parser.add_argument('--episodes_per_file', type=int, default=10, help="Number of episodes chunked per output image (prevents OOM)")
    parser.add_argument('--output_path', type=str, default='./tmp/dataset_visualization_goals.png', help="Output file base path")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_path}")
    root = load_dataset(args.dataset_path)

    # Validate dataset structure complies with Actoris-Harena schema
    if 'trajectory_lengths' not in root:
        print("ERROR: Could not find 'trajectory_lengths' array. Invalid dataset structure.")
        return
        
    traj_lengths = root['trajectory_lengths'][:]
    num_total_episodes = len(traj_lengths)
    
    if num_total_episodes == 0:
        print("ERROR: Dataset exists but contains 0 trajectories.")
        return

    # Calculate absolute index offsets for slicing
    traj_starts = np.concatenate(([0], np.cumsum(traj_lengths)[:-1]))
    num_episodes_to_vis = min(args.num_episodes, num_total_episodes)
    print(f"Visualizing {num_episodes_to_vis} episodes in chunks of {args.episodes_per_file}...")

    # Load data groups into memory pointers (Zarr handles lazy loading dynamically)
    obs_group = root['observation']
    rgb_arr = obs_group.get('rgb')
    goal_rgb_arr = obs_group.get('goal_rgb')
    semkey_arr = obs_group.get('semkey_norm_pixel')
    goal_semkey_arr = obs_group.get('flattened_semkey_norm_pixel') 
    act_arr = root['action']['default']

    img_size = 512
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    base_name, ext = os.path.splitext(args.output_path)

    num_chunks = int(np.ceil(num_episodes_to_vis / args.episodes_per_file))

    # Process grid generation in bounded chunks to constrain RAM usage
    for chunk_idx in range(num_chunks):
        chunk_start_ep = chunk_idx * args.episodes_per_file
        chunk_end_ep = min((chunk_idx + 1) * args.episodes_per_file, num_episodes_to_vis)
        
        print(f"Processing part {chunk_idx + 1}/{num_chunks} (Episodes {chunk_start_ep} to {chunk_end_ep - 1})...")

        chunk_episode_rows = []
        chunk_max_steps = 0

        # Compile episode rows for the current chunk
        for ep_idx in range(chunk_start_ep, chunk_end_ep):
            start_idx = traj_starts[ep_idx]
            length = traj_lengths[ep_idx]
            end_idx = start_idx + length
            
            chunk_max_steps = max(chunk_max_steps, length)
            
            # Slice required tensors for the episode
            ep_rgb = rgb_arr[start_idx:end_idx]
            ep_act = act_arr[start_idx:end_idx]
            ep_goal_rgb = goal_rgb_arr[start_idx:end_idx] if goal_rgb_arr is not None else None
            ep_semkey = semkey_arr[start_idx:end_idx] if semkey_arr is not None else None
            ep_goal_semkey = goal_semkey_arr[start_idx:end_idx] if goal_semkey_arr is not None else None
            
            row_images = []
            
            # Render individual temporal frames
            for t in range(length):
                img_cur = format_image(ep_rgb[t], img_size)
                img_goal = format_image(ep_goal_rgb[t], img_size) if ep_goal_rgb is not None else np.zeros_like(img_cur)
                    
                if ep_semkey is not None: draw_keypoints(img_cur, ep_semkey[t])
                if ep_goal_semkey is not None: draw_keypoints(img_goal, ep_goal_semkey[t])

                # The N+1 constraint means the final observation has no corresponding action
                if t < length - 1:
                    action_vec = ep_act[t]
                    prim_name, params = decode_action_vector(action_vec)
                    
                    step_text = f"Step {t+1}: "
                    if prim_name == "norm-pixel-dual-pick-and-place": step_text += "Dual P&P"
                    elif prim_name == "norm-pixel-single-pick-and-place": step_text += "Single P&P"
                    elif prim_name == "norm-pixel-pick-and-fling": step_text += "Pick & Fling"
                    elif prim_name == "no-operation": step_text += "No-Op"
                    else: step_text += prim_name
                    
                    color = PRIMITIVE_COLORS.get(prim_name, PRIMITIVE_COLORS["default"])
                    
                    draw_action(img_cur, prim_name, params)
                    draw_text_with_bg(img_cur, step_text, (10, TEXT_Y_STEP), color, scale=1.0, thickness=2)
                else:
                    draw_text_with_bg(img_cur, f"Step {t+1}: Done", (10, TEXT_Y_STEP), (0, 255, 0), scale=1.0, thickness=2)
                
                draw_text_with_bg(img_goal, "Goal State", (10, TEXT_Y_STEP), (200, 200, 200), scale=1.0, thickness=2)

                # Vertical concatenation: Current State above Goal State
                step_combined = cv2.vconcat([img_cur, img_goal])
                row_images.append(step_combined)
                
            chunk_episode_rows.append(row_images)

        # Assemble the final visual grid for the chunk
        grid_rows = []
        blank_img = np.zeros((img_size * 2, img_size, 3), dtype=np.uint8)
        
        for ep_idx_in_chunk, row_images in enumerate(chunk_episode_rows):
            global_ep_idx = chunk_start_ep + ep_idx_in_chunk
            
            if len(row_images) > 0:
                draw_text_with_bg(row_images[0], f"Ep: {global_ep_idx}", (10, (img_size * 2) - 20), (255, 255, 0), scale=1.2, thickness=3)

            # Pad shorter episodes with blank images to ensure grid rectangle uniformity
            while len(row_images) < chunk_max_steps:
                row_images.append(blank_img.copy())
                
            row_concat = cv2.hconcat(row_images)
            grid_rows.append(row_concat)
            
        final_grid = cv2.vconcat(grid_rows)
        
        # Write output block to disk
        chunk_save_path = f"{base_name}_part{chunk_idx + 1:02d}{ext}"
        cv2.imwrite(chunk_save_path, final_grid)
        print(f"-> Saved: {chunk_save_path}")

if __name__ == "__main__":
    main()