import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

import os
import ast
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Adjust these imports if needed based on your project structure
import actoris_harena.api as ag_ar
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task

CACHE_DIR = './analysis'
CACHE_FILE = os.path.join(CACHE_DIR, 'garment_metrics_cache.csv')

def collect_data(garments):
    """Initialises arenas and collects metrics across all splits."""
    register_agents()
    register_arenas()
    
    data = []

    for g_label, arena_name in garments.items():
        print(f"Collecting data for {g_label}...")
        
        # 1. Build Arena
        arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
        arena_cfg = OmegaConf.load(arena_conf_path)
        arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
        
        # 2. Build and set Task
        task_conf_path = os.path.join('./conf/', "task", "flattening_overstretch_penalty_1_no_big_bonus.yaml")
        task_cfg = OmegaConf.load(task_conf_path)
        arena.set_task(build_task(task_cfg))
        
        # 3. Retrieve configs for all three splits
        splits = {
            "Train": arena.get_train_configs(),
            "Val": arena.get_val_configs(),
            "Eval": arena.get_eval_configs()
        }
        
        # 4. Loop through each split and config to extract the metrics
        for split_name, configs in splits.items():
            print(f"  -> Processing {len(configs)} {split_name} episodes")
            for config in configs:
                info = arena.reset(config)
                
                # NOTE: Adjust these keys based on how 'info' is structured in your environment.
                # E.g., info['metrics']['max_iou'], or info['state']['color']
                coverage = info['evaluation']['normalised_coverage'] 
                max_iou = info['evaluation']['max_IoU_to_flattened']
                colour = arena._get_garment_colour()
                
                data.append({
                    "Garment": g_label,
                    "Split": split_name,
                    "Normalised Coverage": coverage,
                    "Max IoU": max_iou,
                    "Colour": str(colour) # Cast to string in case it returns an RGB tuple
                })
                
        arena.close()
        
    return pd.DataFrame(data)

def plot_violin(df, y_col, save_name):
    """Generates and saves a violin plot for continuous metrics without a title."""
    plt.figure(figsize=(8, 6))
    
    sns.violinplot(
        data=df, 
        x="Garment", 
        y=y_col, 
        hue="Split", 
        split=False,
        inner="quartile",
        palette="Set2",
        density_norm="width",
        linewidth=1.5
    )
    
    # plt.xlabel("Garment Type")
    plt.ylabel(y_col)
    plt.xlabel("")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(CACHE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {y_col} violin plot to: {save_path}")
    plt.close()

import os
import ast
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_on_palette(df, save_name):
    """Plots Train split colours as a scatter plot over a circular HSL palette."""
    
    # 1. Filter for the Train split only
    df_train = df[df['Split'] == 'Train'].copy()
    
    # 2. Helper to parse string colours into normalized RGB arrays
    def parse_to_rgb(c_str):
        try:
            c = np.array(ast.literal_eval(c_str), dtype=float)
            if c.max() > 1.0:
                c = c / 255.0
            return c[:3] 
        except Exception:
            return np.array([0.5, 0.5, 0.5])

    df_train['RGB'] = df_train['Colour'].apply(parse_to_rgb)
    
    # 3. Convert RGB to HSL to get Hue and Lightness
    def extract_hl(rgb):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        return pd.Series({'Hue': h, 'Lightness': l})
        
    df_train[['Hue', 'Lightness']] = df_train['RGB'].apply(extract_hl)

    # 4. Create a Polar subplot instead of a standard Cartesian one
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # 5. Create the Circular HSL gradient background
    theta = np.linspace(0, 2 * np.pi, 400) # Angle = Hue
    r = np.linspace(0, 1, 200)             # Radius = Lightness
    Theta, R_grid = np.meshgrid(theta, r)
    
    H_grid = Theta / (2 * np.pi)
    L_grid = R_grid
    S_grid = np.ones_like(H_grid) # Fixed saturation
    
    hls_to_rgb_vec = np.vectorize(colorsys.hls_to_rgb)
    R_c, G_c, B_c = hls_to_rgb_vec(H_grid, L_grid, S_grid)
    RGB_bg = np.dstack((R_c, G_c, B_c))
    
    # Paint the background using pcolormesh
    ax.pcolormesh(Theta, R_grid, RGB_bg, shading='auto')
    
    # 6. Assign fixed colours and markers for each garment type
    garments = df_train['Garment'].unique()
    markers = ['o', 's', '^', 'D', 'v'] 
    
    point_colors = sns.color_palette("Set1", n_colors=len(garments)) 
    
    garment_styles = {
        g: {'marker': markers[i % len(markers)], 'color': point_colors[i]} 
        for i, g in enumerate(garments)
    }
    
    # 7. Plot the scattered points
    for g in garments:
        subset = df_train[df_train['Garment'] == g]
        
        ax.scatter(
            subset['Hue'] * 2 * np.pi,  # Map Hue [0,1] to Angle [0, 2π]
            subset['Lightness'],        # Map Lightness [0,1] to Radius [0, 1]
            color=garment_styles[g]['color'],   
            marker=garment_styles[g]['marker'], 
            edgecolors='black', # Changed back to black for better contrast on white edges
            linewidth=1.5,
            s=120,                              
            zorder=3,
            label=g                             
        )

    # 8. Format the Polar Axis
    ax.set_ylim(0, 1) # Lock the radius from 0 to 1
    
    # Hide the default angular/radial numbers to keep the palette visually clean
    ax.set_yticklabels([]) 
    ax.set_xticklabels([]) 
    
    # Add a faint grid so you can still judge radius and angle distances
    ax.grid(color='white', alpha=1.0, linestyle='--') 
    
    # Generate the legend, pushed slightly outside the circle
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncols=2, fontsize=20)
    plt.tight_layout()
    
    save_path = os.path.join(CACHE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Circular HSL scatter colour plot to: {save_path}")
    plt.close()

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    garments = {
        "Longsleeve": "multi_longsleeve_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Trousers": "multi_trousers_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Skirt": "multi_skirt_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Dress": "multi_dress_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200"
    }

    # ---------------------------------------------------------
    # DATA HANDLING (LOAD OR CACHE)
    # ---------------------------------------------------------
    if os.path.exists(CACHE_FILE):
        print(f"Found existing cache at '{CACHE_FILE}'. Loading data...")
        df = pd.read_csv(CACHE_FILE)
    else:
        print("No cache found. Starting environment to collect data...")
        df = collect_data(garments)
        df.to_csv(CACHE_FILE, index=False)
        print(f"Data cached successfully to '{CACHE_FILE}'.")

    # ---------------------------------------------------------
    # PLOTTING PREFERENCES
    # ---------------------------------------------------------
    print("\nGenerating Plots...")
    sns.set_theme(style="whitegrid")
    
    # Globally increase font sizes for academic readability
    plt.rcParams.update({
        'font.size': 18, 
        'axes.labelsize': 22, 
        'xtick.labelsize': 18, 
        'ytick.labelsize': 18, 
        'legend.fontsize': 16, 
        'legend.title_fontsize': 18
    })
    
    # Generate the three requested plots
    plot_violin(df, "Normalised Coverage", "normalized_coverage_distribution.png")
    plot_violin(df, "Max IoU", "max_iou_distribution.png")
    plot_scatter_on_palette(df, "colour_distribution.png")
    
    print("\nAll plotting complete!")

if __name__ == "__main__":
    main()