import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb 
import itertools
from torch.amp import GradScaler
import time

LOSS_NORMALIZATIONS = {
            'rigid':{'fling':{'manipulation':5, 'nocs':3}, 'place':{'manipulation':1.2, 'nocs':3}},
            'deformable':{'fling':{'manipulation':0.8, 'nocs':3}, 'place':{'manipulation':0.1, 'nocs':3}}
        }

# @profile
def optimize_fb(
             policy, 
             optimizer, 
             loader,
             num_updates=None,
             deformable_weight=None, 
             action_primitives=None, 
             unfactorized_networks=False, 
             coverage_reward=False,
             verbose=True, 
             unfactorized_rewards=False,
             mode='train',
             **kwargs):

    scaler = GradScaler('cuda')

    value_net = policy.value_net
 
    print("[Network] >> Optimizing value network, with reward factorization:", not unfactorized_rewards)
    if coverage_reward:
        print("[Network] Using coverage reward")

    distances = ['rigid', 'deformable']
   
    if loader is None or optimizer is None:
        print(">> No loader or optimizer provided, skipping training")
        return

    device = value_net.device
    mean_update_stats = {}

    if num_updates == None: num_updates = len(loader)

    end = time.time()
    for update_id, in_dict in enumerate(loader):
        start = time.time()
        if update_id >= num_updates: break

        losses = {distance:
                {'fling':{'manipulation':0}, 
                'place':{'manipulation':0}} 
                for distance in distances}
        
        losses = {distance: {primitive :{'manipulation':0} for primitive in action_primitives} for distance in distances}
        l2_error = {distance: {primitive :{'manipulation':0} for primitive in action_primitives} for distance in distances}
        unfactorized_losses = {primitive:0 for primitive in action_primitives}
        visualizations = {distance:{primitive:{'manipulation':None, 'nocs':None, 'obs':None, 'distribution':None} for primitive in action_primitives} for distance in distances}
        
        value_net.train() if mode == 'train' else value_net.eval()
        stats = dict()

        for primitive_id in range(len(action_primitives)):

            action_primitive = action_primitives[primitive_id]
            
            weighted_reward = in_dict['weighted_reward'].to(device)
            deformable_reward = in_dict['deformable_reward'].to(device)
            rigid_reward = in_dict['rigid_reward'].to(device)
            l2_reward = in_dict['l2_reward'].to(device)
            cov_reward = in_dict['coverage_reward'].to(device)
            obs = in_dict['obs'].to(device)
            action = in_dict['action'].to(device)
            
            if len(deformable_reward.shape) == 3:
                deformable_reward = torch.masked_select(deformable_reward.to(device), action.to(device))
                rigid_reward = torch.masked_select(rigid_reward.to(device), action.to(device))
                weighted_reward = deformable_reward
                l2_reward = deformable_reward
                cov_reward = deformable_reward
                
            rewards = {'rigid': rigid_reward, 'deformable': deformable_reward}

            if unfactorized_rewards:
                if coverage_reward:
                    print("[Network] Using coverage reward")
                    unfactorized_reward = cov_reward.to(device)
                else:
                    print("[Network] Using unfactorized reward")
                    unfactorized_reward = l2_reward.to(device)
            else:
                # print("[Network] Using factorized reward")
                unfactorized_reward = weighted_reward.to(device)
            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                obs = value_net.preprocess_obs(obs.to(device, non_blocking=True))
                out = value_net.forward_for_optimize(obs, action_primitive, preprocess=False)
                unfactorized_value_pred_dense = (1-deformable_weight) * out['rigid'][action_primitive] + deformable_weight * out['deformable'][action_primitive]
                unfactorized_value_pred = torch.masked_select(unfactorized_value_pred_dense.squeeze(), action.to(device))
                unfactorized_losses[action_primitive] = torch.nn.functional.smooth_l1_loss(unfactorized_value_pred, unfactorized_reward.to(device, non_blocking=True))

            for distance in distances:
                with torch.amp.autocast('cuda'):
                    value_pred_dense = out[distance][action_primitive]
                    value_pred = torch.masked_select(
                        value_pred_dense.squeeze(),
                        action.to(device, non_blocking=True))
                    reward = rewards[distance].to(device)
                    manipulation_loss = torch.nn.functional.smooth_l1_loss(value_pred, reward)
                    losses[distance][action_primitive]['manipulation'] = manipulation_loss / LOSS_NORMALIZATIONS[distance][action_primitive]['manipulation']
                    l2_error[distance][action_primitive]['manipulation'] = manipulation_loss

                log_idx = 0
                visualizations[distance][action_primitive]['manipulation'] = value_pred_dense[log_idx].detach().cpu().numpy().astype(np.float32)
                visualizations[distance][action_primitive]['obs'] = obs[log_idx].detach().cpu().numpy().astype(np.float32)

        #OPTIMIZE
        loss = 0

        for distance in distances:
            for primitive in action_primitives:
                stats[f'loss/{primitive}/unfactorized']= unfactorized_losses[primitive] / len(action_primitives)
                stats[f'loss/{primitive}/{distance}/factorized'] = losses[distance][primitive]['manipulation'] / len(action_primitives)
                stats[f'l2_error/{primitive}/{distance}/factorized'] = l2_error[distance][primitive]['manipulation'] / len(action_primitives)

        if unfactorized_networks:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/unfactorized' in k)
        else:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/factorized' in k)

        if mode == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if verbose:
            print(f"{mode} Update {update_id+1}/{num_updates} - Loss: {loss.item():.4f}")

        for k,v in stats.items():
            if k not in mean_update_stats:
                mean_update_stats[k] = []
            mean_update_stats[k].append(float(v))

        print(f"[Optimize] Total: {time.time()-end}, Sample: {start-end}, Opt_fb: {time.time()-start}")
        end = time.time()

    if mode=='train':
        policy.train_steps += 1
 
    #VISUALIZATIONS
    pairings = itertools.product(action_primitives, distances)
    sample_obs = visualizations[distance][action_primitive]['obs']
    num_channels = sample_obs.shape[0]

    num_with_rgb = num_channels - 2
    fig, axs = plt.subplots(num_with_rgb, 2, figsize=(4, num_with_rgb*2))

    for i, action_primitive in enumerate(action_primitives):
        chosen_obs = visualizations[distance][action_primitive]['obs']
        axs[0, i].set_title(f"({chosen_obs[:3].min():.2f}, {chosen_obs[:3].max():.2f})", fontsize=8)
        axs[0, i].imshow((chosen_obs[:3].transpose(1, 2, 0) * 0.5) + 0.5)
        for j in range(3, num_channels):
            axs[j-2, i].set_title(f"({chosen_obs[j].min():.2f}, {chosen_obs[j].max():.2f})", fontsize=8)
            axs[j-2, i].imshow(chosen_obs[j])
    for ax in axs.flat:
        ax.set_axis_off()
    fig.tight_layout()
    wandb.log({"network_input": wandb.Image(fig)}, step=policy.train_steps.item())
    # plt.savefig("logs/log_images/network_input.png")
    fig.clear()
    plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(2, 8))
    for i, (primitive, distance) in enumerate(list(pairings)):
        axs[i].set_title(f"{primitive}_{distance} {visualizations[distance][primitive]['manipulation'][0].min():.2f},{visualizations[distance][primitive]['manipulation'][0].max():.2f}", fontsize=8)
        axs[i].imshow(visualizations[distance][primitive]['manipulation'][0], cmap='jet')
        axs[i].set_axis_off()
    fig.tight_layout()
    wandb.log({"network_output": wandb.Image(fig)}, step=policy.train_steps.item())
    # plt.savefig("logs/log_images/network_output.png")
    fig.clear()
    plt.close()

    for k,v in mean_update_stats.items():
        wandb.log({f'{mode}_{k}':np.mean(v)}, step=int(policy.train_steps.item()))

    value_net.eval()
    print("[Network] << Optimized value network")