
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import ray
import numpy as np
import cv2
from dotmap import DotMap
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import agent_arena as ag_ar
from agent_arena.utilities.perform_parallel import setup_arenas, step_arenas
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
from lagarnet.utils import obs_config, action_config, reward_names, evaluation_names

def get_actions(agents, arena_ids, ready_infos, actions):
    """
    Collects actions from all ready agents.
    """
    ready_actions = []
    for i, info in enumerate(ready_infos):
        # Map the info back to the correct agent index using the arena_id
        idx = arena_ids.index(info['arena_id'])
        # Agents act based on the observation
        ready_actions.append(agents[idx].act([info], update=True)[0]) 
    
    # Store actions in the buffer
    for info, a in zip(ready_infos, ready_actions):
        arena_id = info['arena_id']
        idx = arena_ids.index(arena_id)
        
        # Handle specific action keys for your domain
        if 'norm-pixel-pick-and-place' in a:
            action = a['norm-pixel-pick-and-place']
        else:
            action = a
            
        # Stack pick/place for storage [2, ...]
        action = np.stack([action['pick_0'], action['place_0']])
        actions[idx]['norm-pixel-pick-and-place'].append(action)

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
    
    for key in reward_names:
        if key in info['reward']:
            observations[idx][key].append(info['reward'][key])
            
    for key in evaluation_names:
        if key in info['evaluation']:
            observations[idx][key].append(info['evaluation'][key])

def build_single_agent(cfg):
    """
    Builds an agent instance based on the Hydra config.
    """
    # 1. Identify Agent Name
    # Priority: cfg.agent.name -> cfg.exp_name
    agent_name = cfg.agent.get('name', cfg.exp_name)

    print(f"Building Agent: {agent_name}...")

    # 2. Configure Agent
    if 'oracle' not in agent_name:
        # Retrieve config path details from Hydra, defaulting if missing
        agent_config_name = cfg.agent.get('config_name', 'default')
        # Some agents need the arena name they were trained on
        agent_trained_arena = cfg.agent.get('trained_arena_name', 'default')
        
        agent_config = ag_ar.retrieve_config(
            agent_name, 
            agent_trained_arena, 
            agent_config_name,
            config_dir='../configuration'
        )
    else:
        # Oracles usually just need a flag
        agent_config = DotMap({'oracle': True})
    
    # 3. Build & Setup Logging
    agent = ag_ar.build_agent(agent_name, config=agent_config)
    
    # Save logs to the experiment directory
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    agent.set_log_dir(save_dir)
    
    # 4. Load Checkpoint (if not Oracle)
    if 'oracle' not in agent_name:
        ckpt = cfg.get('checkpoint', -1)
        if ckpt != -1:
            print(f'Loading checkpoint {ckpt}')
            agent.load_checkpoint(ckpt)
        else:
            print('Loading the latest checkpoint')
            agent.load()
            
    return agent

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="data_collection/collect_dataset_01", version_base=None)
def main(cfg: DictConfig):
    
    print("--- Configuration ---")
    # Resolve=True ensures all ${variables} are expanded
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("---------------------")

    ray.init(ignore_reinit_error=True)

    # 1. Setup Arenas
    # IMPORTANT: setup_arenas usually expects a String ID or a Config Dict depending on implementation.
    # We pass cfg.arena.name (the registry string) if available, otherwise we might pass the dict.
    arena_arg = cfg.arena.get('name') 
    if not arena_arg:
         # Fallback: if name isn't explicit, pass the whole config object if setup_arenas supports it
         arena_arg = cfg.arena 

    print(f"Setting up arenas with ID/Config: {arena_arg}")
    
    # Note: If setup_arenas expects the Hydra DictConfig, you might need: setup_arenas(cfg.arena, ...)
    # But usually it expects a string ID like "softgym|domain:..."
    arenas = setup_arenas(arena_arg, num_processes=cfg.parallel_processes)
    
    process = []
    for i, arena in enumerate(arenas):
        # Assuming arena is a Ray actor
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        process.append(arena.set_id.remote(arena_id))
        process.append(arena.set_train.remote())
    
    # Wait for initialization
    ray.get(process)

    # Get IDs for mapping
    arena_ids = ray.get([e.get_id.remote() for e in arenas])
    
    # 2. Setup Agents
    # We rebuild the agent for every process to ensure they are independent
    if cfg.single_agent:
        print('--- Single Agent Mode ---')
        # Build one agent and share it (not recommended for stateful agents in parallel)
        agent = build_single_agent(cfg)
        agents = [agent for _ in range(len(arenas))]
        agent.reset(arena_ids)
    else:
        print('--- Multi Agent Mode (Independent Agents) ---')
        agents = [build_single_agent(cfg) for _ in range(len(arenas))]
        for i, agent in enumerate(agents):
            agent.reset([arena_ids[i]])
    
    # 3. Initialize Buffers
    observations = [{k: [] for k in obs_config.keys()} for _ in range(len(arenas))]
    actions = [{'norm-pixel-pick-and-place': []} for _ in range(len(arenas))]

    # 4. Dataset Setup
    os.makedirs(cfg.save_root, exist_ok=True)
    dataset = TrajectoryDataset(
        data_path=cfg.data_path,
        data_dir=cfg.save_root,
        io_mode='a',
        obs_config=obs_config,
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
    
    # Progress bar tracks NEW trajectories
    pbar = tqdm(total=cfg.trials, desc='Collecting Trajectories')
    # If you want to include existing count in bar: pbar.update(existing_trajs)
    
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
                
                # Reset Buffers
                observations[idx] = {k: [] for k in obs_config.keys()}
                actions[idx] = {'norm-pixel-pick-and-place': []}
                
                # Remove from ready list so it doesn't get stepped immediately
                ready_arenas.remove(arenas[idx])
                
                # Reset Agent Logic
                if cfg.single_agent:
                    agents[idx].reset([arena_id])
                else:
                    # Rebuild/Reset agent to clear internal state
                    agents[idx] = build_single_agent(cfg)
                    agents[idx].reset([arena_id])
                
                # Queue Environment Reset
                waiting_infos.append(arenas[idx].reset.remote())
            else:
                new_ready_infos.append(info)
        
        ready_infos = new_ready_infos

if __name__ == '__main__':
    main()