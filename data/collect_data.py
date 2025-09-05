import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
import numpy as np
import argparse
import cv2
import os
from dotmap import DotMap
import agent_arena as ag_ar
from agent_arena.utilities.perform_parallel \
    import setup_arenas
from agent_arena.utilities.perform_parallel \
    import step_arenas

from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
from lagarnet.utils import obs_config, action_config, reward_names, evaluation_names
from tqdm import tqdm

def get_actions(agents, arena_ids, ready_infos, actions):
    ready_actions = []
    for i, info in enumerate(ready_infos):
        idx = arena_ids.index(info['arena_id'])
        ready_actions.append(agents[idx].act([info], update=True)[0]) 
        # update the internal state with last info and last action first #
        # regarding the corresponding arena_id, then act to give action.
    
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
    #   print('rgb shape', info['observation']['rgb'].shape)
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

def get_agent(
        agent_name, agent_config_name, 
        arena_name, agent_save_dir, checkpoint=-1):
    if 'oracle' not in agent_name:
        config = ag_ar.retrieve_config(
            agent_name, 
            arena_name, 
            agent_config_name,
            config_dir='../configuration'
            )
    else:
        config = DotMap({
            'oracle': True
        })
    
    agent = ag_ar.build_agent(agent_name, config=config)
    save_dir = os.path.join(agent_save_dir, arena_name, agent_name, agent_config_name)
    agent.set_log_dir(save_dir)
    print('Finished building agent {}'.format(agent_name))
    if 'oracle' not in agent_name:
        if checkpoint != -1:
            agent.load_checkpoint(checkpoint)
        else:
            print('Loading the latest checkpoint')
            agent.load()
    return agent


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--parallel_processes', type=int, default=4)
    parser.add_argument('--agent_name', type=str, default='oracle-garment|mask-biased-pick-and-place')
    parser.add_argument('--agent_config_name', type=str, default='human-50')
    parser.add_argument('--agent_trained_arena_name', type=str, default='softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:30')
    parser.add_argument('--arena_name', type=str, default='softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:20')
    parser.add_argument('--agent_save_dir', type=str, default='/data/planet-clothpick++')
    parser.add_argument('--single_agent', action='store_true')
    parser.add_argument('--checkpoint', type=int, default='-1')
    parser.add_argument('--data_path', type=str, default='gc_longsleeve_flattening')
    return parser.parse_args()


def main():
    ray.init()

    arg_parser = argument_parser()

    arenas = setup_arenas(arg_parser.arena_name, num_processes=arg_parser.parallel_processes)
    process = []
    for i, arena in enumerate(arenas):
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        process.append(arena.set_id.remote(arena_id))
        process.append(arena.set_train.remote())
    ray.get(process)
        # ray.get(arena.set_id.remote(arena_id))
        # ray.get(arena.set_train.remote())

    arena_ids = ray.get([e.get_id.remote() for e in arenas])
    
    if arg_parser.single_agent:
        print('Single Agent !!!')
        agent = get_agent(
            arg_parser.agent_name, 
            arg_parser.agent_config_name, 
            arg_parser.agent_trained_arena_name,
            arg_parser.agent_save_dir,
            arg_parser.checkpoint)
        agents = [agent for _ in range(len(arenas))]
        agent.reset(arena_ids)
    else:
        agents = [get_agent(arg_parser.agent_name, 
                            arg_parser.agent_config_name, 
                            arg_parser.agent_trained_arena_name,
                            arg_parser.agent_save_dir,
                            arg_parser.checkpoint) for _ in range(len(arenas))]
        for i, agent in enumerate(agents):
            agent.reset([arena_ids[i]])
    
    observations = [{
        k: [] for k in obs_config.keys()
    } for _ in range(len(arenas))]
    actions = [{'norm-pixel-pick-and-place': []} for _ in range(len(arenas))]


    dataset = TrajectoryDataset(
        data_path=arg_parser.data_path,
        data_dir='./datasets',
        io_mode='a',
        obs_config=obs_config,
        act_config=action_config,
        whole_trajectory=True
    )
    
    waiting_infos = [e.reset.remote() for e in arenas]
    ready_arenas = []
    ready_actions = []
    ready_infos = []
    
    print('saved data', dataset.num_trajectories())
    ## intialise tdqm
    pbar = tqdm(total=arg_parser.trials, desc='Collecting Trajectories')
    pbar.update(dataset.num_trajectories())
    while dataset.num_trajectories() < arg_parser.trials:
        
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
                ## reset the agent and arena
                
                dataset.add_trajectory(
                    observations[idx],
                    actions[idx])
                pbar.update(1)
                # clean observations and actions
                observations[idx] = {
                    k: [] for k in obs_config.keys()
                }
                actions[idx] = {'norm-pixel-pick-and-place': []}
                
                ready_arenas.remove(arenas[idx])
                if arg_parser.single_agent:
                    agents[idx].reset([arena_id])
                else:
                    agents[idx] = get_agent(arg_parser.agent_name, 
                            arg_parser.agent_config_name, 
                            arg_parser.agent_trained_arena_name,
                            arg_parser.agent_save_dir,
                            arg_parser.checkpoint)
                    agents[idx].reset([arena_id])
                waiting_infos.append(arenas[idx].reset.remote())
            else:
                new_ready_infos.append(info)
        ready_infos = new_ready_infos

if __name__ == '__main__':
    main()