import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import ray
import numpy as np
import cv2
from dotmap import DotMap
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import agent_arena as ag_ar
from agent_arena.utilities.perform_parallel import setup_arenas, step_arenas
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
from lagarnet.utils import obs_config, action_config, reward_names, evaluation_names

def get_actions(agents, arena_ids, ready_infos, actions):
    ready_actions = []
    for i, info in enumerate(ready_infos):
        idx = arena_ids.index(info['arena_id'])
        ready_actions.append(agents[idx].act([info], update=True)[0]) 
    
    for info, a in zip(ready_infos, ready_actions):
        arena_id = info['arena_id']
        idx = arena_ids.index(arena_id)
        if 'norm-pixel-pick-and-place' in a:
            action = a['norm-pixel-pick-and-place']
        else:
            action = a
        action = np.stack([action['pick_0'], action['place_0']])
        actions[idx]['norm-pixel-pick-and-place'].append(action)

    return ready_actions

def update_observations(observations, idx, info):
    # Resize and Format Observations
    rgb = cv2.resize(info['observation']['rgb'], (128, 128))
    depth = cv2.resize(info['observation']['depth'], (128, 128))
    mask = cv2.resize(info['observation']['mask'].astype(np.float32), (128, 128))
    mask = mask > 0.9

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
        observations[idx][key].append(info['reward'][key])
    for key in evaluation_names:
        observations[idx][key].append(info['evaluation'][key])

def build_single_agent(cfg):
    """
    Builds the agent using the merged configuration from Hydra.
    """
    # 1. Determine Agent Name
    # We try to get the name from the config, or fall back to the experiment name
    if 'name' in cfg.agent:
        agent_name = cfg.agent.name
    else:
        # Fallback if your agent yaml doesn't explicitly have a 'name' field
        agent_name = cfg.exp_name 

    print(f"Building Agent: {agent_name}...")

    # 2. Retrieve/Construct Config
    if 'oracle' not in agent_name:
        # Assuming cfg.agent contains the params needed, or we load by name
        # If your agent yaml is fully self-contained in cfg.agent, pass that.
        # Otherwise, retrieve from file:
        agent_config_name = cfg.agent.get('config_name', 'default')
        agent_trained_arena = cfg.agent.get('trained_arena_name', 'default')
        
        agent_config = ag_ar.retrieve_config(
            agent_name, 
            agent_trained_arena, 
            agent_config_name,
            config_dir='../configuration'
        )
    else:
        agent_config = DotMap({'oracle': True})
    
    # 3. Build
    agent = ag_ar.build_agent(agent_name, config=agent_config)
    
    # 4. Set Log Dir
    # We use the save_root defined in the data_collection yaml
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    agent.set_log_dir(save_dir)
    
    # 5. Load Checkpoint
    if 'oracle' not in agent_name:
        # Check if 'checkpoint' is in cfg (it might not be in run_exp, but can be added to data_collection)
        ckpt = cfg.get('checkpoint', -1)
        if ckpt != -1:
            agent.load_checkpoint(ckpt)
        else:
            print('Loading the latest checkpoint')
            agent.load()
            
    return agent

# Point config_path to the folder containing 'data_collection'
@hydra.main(config_path="../conf", config_name="data_collection/collect_dataset_01", version_base=None)
def main(cfg: DictConfig):
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("---------------------")

    ray.init()

    # 1. Setup Arenas
    # The 'arena' config is now merged into cfg.arena.name due to defaults list
    arena_name = cfg.arena.name
    arenas = setup_arenas(arena_name, num_processes=cfg.parallel_processes)
    
    process = []
    for i, arena in enumerate(arenas):
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        process.append(arena.set_id.remote(arena_id))
        process.append(arena.set_train.remote())
    ray.get(process)

    arena_ids = ray.get([e.get_id.remote() for e in arenas])
    
    # 2. Setup Agents
    if cfg.single_agent:
        print('--- Single Agent Mode ---')
        agent = build_single_agent(cfg)
        agents = [agent for _ in range(len(arenas))]
        agent.reset(arena_ids)
    else:
        print('--- Multi Agent Mode ---')
        agents = [build_single_agent(cfg) for _ in range(len(arenas))]
        for i, agent in enumerate(agents):
            agent.reset([arena_ids[i]])
    
    # 3. Init Data Structures
    observations = [{k: [] for k in obs_config.keys()} for _ in range(len(arenas))]
    actions = [{'norm-pixel-pick-and-place': []} for _ in range(len(arenas))]

    # 4. Dataset Init
    os.makedirs(cfg.save_root, exist_ok=True)
    dataset = TrajectoryDataset(
        data_path=cfg.data_path,
        data_dir=cfg.save_root,
        io_mode='a',
        obs_config=obs_config,
        act_config=action_config,
        whole_trajectory=True
    )
    
    # 5. Collection Loop
    waiting_infos = [e.reset.remote() for e in arenas]
    ready_arenas = []
    ready_actions = []
    ready_infos = []
    
    print(f'Saved data count: {dataset.num_trajectories()}')
    pbar = tqdm(total=cfg.trials, desc='Collecting Trajectories')
    pbar.update(dataset.num_trajectories())
    
    while dataset.num_trajectories() < cfg.trials:
        
        ready_actions = get_actions(agents, arena_ids, ready_infos, actions)
        
        ready_arenas, ready_infos, waiting_infos = step_arenas(
            arenas, arena_ids, 
            ready_arenas, ready_actions, 
            waiting_infos)
        
        new_ready_infos = []
        for info in ready_infos:
            arena_id = info['arena_id']
            idx = arena_ids.index(arena_id)

            update_observations(observations, idx, info)
            
            if info['done']:
                dataset.add_trajectory(observations[idx], actions[idx])
                pbar.update(1)
                
                # Reset buffers
                observations[idx] = {k: [] for k in obs_config.keys()}
                actions[idx] = {'norm-pixel-pick-and-place': []}
                
                ready_arenas.remove(arenas[idx])
                
                # Reset Agent
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