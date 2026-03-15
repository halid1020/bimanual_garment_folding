import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

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
    plt.figure(figsize=(14, 8))
    
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
    
    plt.xlabel("Garment Type")
    plt.ylabel(y_col)
    plt.legend(title="Dataset Split", loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(CACHE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {y_col} violin plot to: {save_path}")
    plt.close()

def plot_colour_distribution(df, save_name):
    """Generates and saves a count plot for categorical colour distributions."""
    plt.figure(figsize=(14, 8))
    
    # We use a countplot grouped by garment and split to see the color distribution
    sns.countplot(
        data=df, 
        x="Garment", 
        hue="Colour", 
        palette="tab10", # Distinct colors for different categorical values
        edgecolor="black"
    )
    
    plt.xlabel("Garment Type")
    plt.ylabel("Count")
    plt.legend(title="Colour", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    
    save_path = os.path.join(CACHE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Colour distribution plot to: {save_path}")
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
    plot_colour_distribution(df, "colour_distribution.png")
    
    print("\nAll plotting complete!")

if __name__ == "__main__":
    main()