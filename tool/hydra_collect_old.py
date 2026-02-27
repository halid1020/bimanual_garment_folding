
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import ray
import numpy as np
import cv2
from dotmap import DotMap
from tqdm import tqdm
import copy

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import actoris_harena as ag_ar
from actoris_harena.utilities.perform_parallel \
    import setup_arenas_with_class, step_arenas
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task
from tool.lagarnet_utils import * 


def get_actions(agents, arena_ids, ready_infos, actions):
    """
    Collects actions from all ready agents.
    """
    ready_actions = []
    for i, info in enumerate(ready_infos):
        # Map the info back to the correct agent index using the arena_id
        idx = arena_ids.index(info['arena_id'])
        # Agents act based on the observation
        ready_actions.append(agents[idx].act([info], updates=[True])[0]) 
    
    # Store actions in the buffer
    for info, a in zip(ready_infos, ready_actions):
        arena_id = info['arena_id']
        idx = arena_ids.index(arena_id)
        actions[idx]['norm-pixel-pick-and-place'].append(a)

    return ready_actions

def update_observations(observations, idx, info):
    """
    Resizes and appends observations to the buffer.
    """
    # Standard resize to 128x128
    rgb = cv2.resize(info['observation']['rgb'], (128, 128))
    depth = cv2.resize(info['observation']['depth'], (128, 128))
    mask = cv2.resize(info['observation']['mask'].astype(np.float32), (128, 128))
    mask = mask > 0.9 # Binarize

    goal_rgb = cv2.resize(info['goal']['rgb'], (128, 128))
    goal_depth = cv2.resize(info['goal']['depth'], (128, 128))
    goal_mask = cv2.resize(info['goal']['mask'].astype(np.float32), (128, 128))
    goal_mask = goal_mask > 0.9

    observations[idx]['rgb'].append(rgb)
    observations[idx]['depth'].append(depth)
    observations[idx]['mask'].append(mask)
    observations[idx]['goal-rgb'].append(goal_rgb)
    observations[idx]['goal-depth'].append(goal_depth)
    observations[idx]['goal-mask'].append(goal_mask)

    # --- UPDATED LOGIC ---
    # Only append success if the key exists in the observation buffer
    if 'success' in observations[idx]:
        is_success = info.get('success', False)
        observations[idx]['success'].append(is_success)
    # ---------------------
    
    for key in reward_names:
        if key in info['reward'] and key in observations[idx]:
            observations[idx][key].append(info['reward'][key])
            
    for key in evaluation_names:
        if key in info['evaluation'] and key in observations[idx]:
            observations[idx][key].append(info['evaluation'][key])

def build_single_agent(cfg):
    """
    Builds an agent instance based on the Hydra config.
    """
    # 1. Identify Agent Name
    # Priority: cfg.agent.name -> cfg.exp_name
    agent_name = cfg.agent.name

    print(f"Building Agent: {agent_name}...")
    
    # 3. Build & Setup Logging
    save_dir = os.path.join(cfg.agent_save_root, cfg.agent_exp_name)
    agent = ag_ar.build_agent(
        agent_name, 
        config=cfg.agent, 
        save_dir=save_dir)
    
    # Save logs to the experiment directory

    
    # 4. Load Checkpoint (if not Oracle)
    if isinstance(agent, ag_ar.TrainableAgent):
        agent.load_best()
            
    return agent

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="data_collection/collect_dataset_01", version_base=None)
def main(cfg: DictConfig):
    register_agents()
    register_arenas()

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("---------------------")

    ray.init(ignore_reinit_error=True)

    # --- NEW: Modify obs_config based on config name ---
    # Create a local copy to avoid modifying the global import
    local_obs_config = copy.deepcopy(obs_config)
    
    # Attempt to retrieve the Hydra config name
    try:
        hydra_conf_name = HydraConfig.get().job.config_name
    except:
        hydra_conf_name = ""

    # Check config name OR data_path for 'mask-biased'
    if 'mask-biased' in hydra_conf_name or 'mask-biased' in cfg.get('data_path', ''):
        print("!!! 'mask-biased' detected: Removing 'success' from observation config !!!")
        if 'success' in local_obs_config:
            del local_obs_config['success']
    # ---------------------------------------------------

    # 1. Setup Arenas
    arena_class = ag_ar.get_arena_class(cfg.arena.name)
    arenas = setup_arenas_with_class(
        arena_class, cfg.arena,
        num_processes=cfg.parallel_processes)
    
    ray.get([e.set_task.remote(build_task(cfg.task)) for e in arenas])

    process = []
    for i, arena in enumerate(arenas):
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        process.append(arena.set_id.remote(arena_id))
        process.append(arena.set_train.remote())
    
    ray.get(process)

    # Get IDs for mapping
    arena_ids = ray.get([e.get_id.remote() for e in arenas])
    
    # 2. Setup Agents
    if cfg.single_agent:
        print('--- Single Agent Mode ---')
        agent = build_single_agent(cfg)
        agents = [agent for _ in range(len(arenas))]
        agent.reset(arena_ids)
    else:
        print('--- Multi Agent Mode (Independent Agents) ---')
        agents = [build_single_agent(cfg) for _ in range(len(arenas))]
        for i, agent in enumerate(agents):
            agent.reset([arena_ids[i]])
    
    # 3. Initialize Buffers
    # Use local_obs_config here so the buffer doesn't have the 'success' key if removed
    observations = [{k: [] for k in local_obs_config.keys()} for _ in range(len(arenas))]
    actions = [{'norm-pixel-pick-and-place': []} for _ in range(len(arenas))]

    # 4. Dataset Setup
    dataset = TrajectoryDataset(
        data_path=cfg.data_path,
        data_dir=cfg.data_dir,
        io_mode='a',
        obs_config=local_obs_config, # Pass the modified config
        act_config=action_config,
        whole_trajectory=True
    )
    
    # 5. Data Collection Loop
    waiting_infos = [e.reset.remote() for e in arenas]
    ready_arenas = []
    ready_actions = []
    ready_infos = []
    
    existing_trajs = dataset.num_trajectories()
    print(f'Starting with {existing_trajs} existing trajectories.')
    
    pbar = tqdm(total=cfg.trials, desc='Collecting Trajectories')
    
    while dataset.num_trajectories() < cfg.trials:
        
        # 1. Get Actions from Agents
        ready_actions = get_actions(agents, arena_ids, ready_infos, actions)
        
        # 2. Step Environments (Parallel)
        ready_arenas, ready_infos, waiting_infos = step_arenas(
            arenas, arena_ids, 
            ready_arenas, ready_actions, 
            waiting_infos
        )
        
        new_ready_infos = []
        for info in ready_infos:
            arena_id = info['arena_id']
            idx = arena_ids.index(arena_id)

            update_observations(observations, idx, info)
            
            if info['done']:
                # Save completed trajectory
                dataset.add_trajectory(observations[idx], actions[idx])
                pbar.update(1)
                
                # Reset Buffers (using local_obs_config keys)
                observations[idx] = {k: [] for k in local_obs_config.keys()}
                actions[idx] = {'norm-pixel-pick-and-place': []}
                
                ready_arenas.remove(arenas[idx])
                
                if cfg.single_agent:
                    agents[idx].reset([arena_id])
                else:
                    agents[idx] = build_single_agent(cfg)
                    agents[idx].reset([arena_id])
                
                waiting_infos.append(arenas[idx].reset.remote())
            else:
                new_ready_infos.append(info)
        
        ready_infos = new_ready_infos

if __name__ == '__main__':
    main()