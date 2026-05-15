import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Constants & Colors ---
TEXT_Y_STEP = 35
TEXT_BG_ALPHA = 0.6

BLUE = (200, 50, 50)        # Left picker (BGR for OpenCV)
RED  = (50, 50, 200)        # Right picker

PRIMITIVE_COLORS = {
    "norm-pixel-pick-and-place": (255, 180, 80),        # Orange
    "norm-pixel-dual-pick-and-place": (255, 180, 80),   # Orange
    "norm-pixel-single-pick-and-place": (255, 180, 80), # Orange
    "norm-pixel-pick-and-fling": (80, 200, 255),        # Cyan
    "no-operation": (255, 255, 255),                    # Gray
    "default": (255, 255, 255),                         # White
}

# --- Action Configuration ---
PRIMITIVES = [
    {"name": "norm-pixel-pick-and-fling", "dim": 4},
    {"name": "norm-pixel-dual-pick-and-place", "dim": 8},
    {"name": "norm-pixel-single-pick-and-place", "dim": 4},
    {"name": "no-operation", "dim": 0}
]

# --- Drawing Utilities ---
def norm_to_px(v, W, H):
    """Convert normalized [-1, 1] coordinates to pixel coordinates."""
    x = int((v[0] + 1) * 0.5 * W)
    y = int((v[1] + 1) * 0.5 * H)
    return x, y

def swap(p):
    """OpenCV uses (col, row) aka (x, y) for drawing."""
    return (p[1], p[0])

def draw_big_arrowhead(img, p_from, p_to, color, size=28):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6

    ux, uy = dx / norm, dy / norm      
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

def draw_colored_line(img, p_start, p_end, cmap, thickness=8, num_samples=20):
    xs = np.linspace(p_start[0], p_end[0], num_samples).astype(int)
    ys = np.linspace(p_start[1], p_end[1], num_samples).astype(int)

    for i in range(1, num_samples):
        alpha = i / (num_samples - 1)
        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
        color = cv2.applyColorMap(value, cmap)[0, 0].tolist()
        cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color, thickness)

    draw_big_arrowhead(img, (xs[-2], ys[-2]), (xs[-1], ys[-1]), color, size=30)

def draw_text_with_bg(img, text, org, color, scale=1.0, thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, TEXT_BG_ALPHA, img, 1 - TEXT_BG_ALPHA, 0, img)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_keypoints(img, keypoints_norm):
    """Draws semantic keypoints using a distinct colormap."""
    if keypoints_norm is None: return
    
    H, W = img.shape[:2]
    keypoints_norm = keypoints_norm.reshape(-1, 2)
    num_kps = len(keypoints_norm)
    
    # Generate distinct colors using matplotlib
    cmap = plt.cm.get_cmap('tab20')
    
    for i, kp in enumerate(keypoints_norm):
        y_norm, x_norm = kp
        x = int((x_norm + 1.0) / 2.0 * W)
        y = int((y_norm + 1.0) / 2.0 * H)
        
        # Get color and convert RGB (matplotlib) to BGR (OpenCV)
        rgba = cmap(i / max(1, num_kps - 1))
        color_bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        
        cv2.circle(img, (x, y), 8, color_bgr, -1)        # Filled colored circle
        cv2.circle(img, (x, y), 8, (255, 255, 255), 2)   # White border for visibility

def draw_action(img, prim_name, params):
    """Draws the geometric action representation on the image."""
    H, W = img.shape[:2]
    cmap_left  = cv2.COLORMAP_COOL
    cmap_right = cv2.COLORMAP_AUTUMN
    BLUE_MAP = cv2.applyColorMap(np.uint8([[[0]]]), cmap_left)[0, 0].tolist()
    RED_MAP = cv2.applyColorMap(np.uint8([[[0]]]), cmap_right)[0, 0].tolist()

    if "pick-and-place" in prim_name:
        if len(params) >= 8:  # Dual Pick and Place
            left_pick  = norm_to_px(params[:2], W, H)
            right_pick = norm_to_px(params[2:4], W, H)
            left_place = norm_to_px(params[4:6], W, H)
            right_place= norm_to_px(params[6:8], W, H)
            
            if left_pick[1] > right_pick[1]: 
                left_pick, right_pick = right_pick, left_pick
                left_place, right_place = right_place, left_place

            draw_colored_line(img, swap(left_pick), swap(left_place), cmap_left, thickness=6)
            draw_colored_line(img, swap(right_pick), swap(right_place), cmap_right, thickness=6)
            cv2.circle(img, swap(left_pick), 8, BLUE_MAP, 3)
            cv2.circle(img, swap(right_pick), 8, RED_MAP, 3)
            
        elif len(params) >= 4:  # Single Pick and Place
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
        
        fling_offset = 60
        draw_big_arrowhead(img, swap(left_pick), (swap(left_pick)[0], swap(left_pick)[1] - fling_offset), BLUE_MAP, size=20)
        draw_big_arrowhead(img, swap(right_pick), (swap(right_pick)[0], swap(right_pick)[1] - fling_offset), RED_MAP, size=20)

# --- Dataset Logic ---
def decode_action_vector(action_vec):
    K = len(PRIMITIVES)
    prim_val = action_vec[0]
    prim_idx = int(np.clip(((prim_val + 1) / 2) * K - 1e-6, 0, K - 1))
    
    prim_info = PRIMITIVES[prim_idx]
    prim_name = prim_info["name"]
    params = action_vec[1 : 1 + prim_info["dim"]]
    
    return prim_name, params

def load_dataset(dataset_dir):
    if dataset_dir.endswith('.zarr') or os.path.isdir(dataset_dir):
        try:
            import zarr
            return zarr.open(dataset_dir, mode='r')
        except ImportError: pass
    try:
        import h5py
        return h5py.File(dataset_dir, 'r')
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset. Check path: {e}")

def format_image(img_data, img_size):
    """Formats raw dataset image arrays to OpenCV BGR."""
    img = img_data.copy()
    if img.shape[0] == 3: # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize saved trajectory dataset")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the .zarr or .hdf5 dataset")
    parser.add_argument('--num_episodes', type=int, default=100, help="Number of episodes to visualize")
    parser.add_argument('--episodes_per_file', type=int, default=10, help="Max number of episodes per PNG image file")
    parser.add_argument('--output_path', type=str, default='./tmp/dataset_visualization_goals.png', help="Output image base path")
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"Loading dataset from: {args.dataset_path}")
    root = load_dataset(args.dataset_path)

    if 'trajectory_lengths' not in root:
        print("ERROR: Could not find 'trajectory_lengths' in the dataset.")
        return
        
    traj_lengths = root['trajectory_lengths'][:]
    num_total_episodes = len(traj_lengths)
    
    if num_total_episodes == 0:
        print("ERROR: Dataset contains 0 episodes.")
        return

    traj_starts = np.concatenate(([0], np.cumsum(traj_lengths)[:-1]))
    num_episodes_to_vis = min(args.num_episodes, num_total_episodes)
    print(f"Visualizing {num_episodes_to_vis} episodes in chunks of {args.episodes_per_file}...")

    obs_group = root['observation']
    rgb_arr = obs_group.get('rgb')
    goal_rgb_arr = obs_group.get('goal_rgb')
    semkey_arr = obs_group.get('semkey_norm_pixel')
    goal_semkey_arr = obs_group.get('flattened_semkey_norm_pixel') 
    act_arr = root['action']['default']

    img_size = 512
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    base_name, ext = os.path.splitext(args.output_path)

    # 2. Chunking Logic
    num_chunks = int(np.ceil(num_episodes_to_vis / args.episodes_per_file))

    for chunk_idx in range(num_chunks):
        chunk_start_ep = chunk_idx * args.episodes_per_file
        chunk_end_ep = min((chunk_idx + 1) * args.episodes_per_file, num_episodes_to_vis)
        
        print(f"Processing part {chunk_idx + 1}/{num_chunks} (Episodes {chunk_start_ep} to {chunk_end_ep - 1})...")

        chunk_episode_rows = []
        chunk_max_steps = 0

        # Process Episodes for this Chunk
        for ep_idx in range(chunk_start_ep, chunk_end_ep):
            start_idx = traj_starts[ep_idx]
            length = traj_lengths[ep_idx]
            end_idx = start_idx + length
            
            chunk_max_steps = max(chunk_max_steps, length)
            
            ep_rgb = rgb_arr[start_idx:end_idx]
            ep_act = act_arr[start_idx:end_idx]
            ep_goal_rgb = goal_rgb_arr[start_idx:end_idx] if goal_rgb_arr is not None else None
            ep_semkey = semkey_arr[start_idx:end_idx] if semkey_arr is not None else None
            ep_goal_semkey = goal_semkey_arr[start_idx:end_idx] if goal_semkey_arr is not None else None
            
            row_images = []
            for t in range(length):
                img_cur = format_image(ep_rgb[t], img_size)
                img_goal = format_image(ep_goal_rgb[t], img_size) if ep_goal_rgb is not None else np.zeros_like(img_cur)
                    
                if ep_semkey is not None: draw_keypoints(img_cur, ep_semkey[t])
                if ep_goal_semkey is not None: draw_keypoints(img_goal, ep_goal_semkey[t])

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

                # Stack Current on top of Goal vertically
                step_combined = cv2.vconcat([img_cur, img_goal])
                row_images.append(step_combined)
                
            chunk_episode_rows.append(row_images)

        # Compile Grid for this Chunk
        grid_rows = []
        blank_img = np.zeros((img_size * 2, img_size, 3), dtype=np.uint8)
        
        for ep_idx_in_chunk, row_images in enumerate(chunk_episode_rows):
            global_ep_idx = chunk_start_ep + ep_idx_in_chunk
            
            if len(row_images) > 0:
                draw_text_with_bg(row_images[0], f"Ep: {global_ep_idx}", (10, (img_size * 2) - 20), (255, 255, 0), scale=1.2, thickness=3)

            # Pad short episodes
            while len(row_images) < chunk_max_steps:
                row_images.append(blank_img.copy())
                
            row_concat = cv2.hconcat(row_images)
            grid_rows.append(row_concat)
            
        final_grid = cv2.vconcat(grid_rows)
        
        # Save Chunk
        chunk_save_path = f"{base_name}_part{chunk_idx + 1:02d}{ext}"
        cv2.imwrite(chunk_save_path, final_grid)
        print(f"-> Saved: {chunk_save_path}")

if __name__ == "__main__":
    main()