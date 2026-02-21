import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
import numpy as np
import torch

DELTA_WEIGHTED_REWARDS_MEAN = -0.0018245290312917787
DELTA_WEIGHTED_REWARDS_STD = 0.072
DELTA_POINTWISE_REWARDS_STD = 0.12881897698788683

def rewards_from_group(group):

    deformable_weight = group.attrs["deformable_weight"]

    deformable_reward = -group.attrs['postaction_l2_distance']/DELTA_WEIGHTED_REWARDS_STD

    rigid_reward = -group.attrs['postaction_icp_distance']/DELTA_WEIGHTED_REWARDS_STD

    delta_pointwise_distance = -group.attrs['postaction_pointwise_distance']
    delta_pointwise_distance /= DELTA_POINTWISE_REWARDS_STD
    l2_reward = delta_pointwise_distance

    weighted_reward = deformable_weight * deformable_reward + (1-deformable_weight) * rigid_reward

    preaction_coverage = group.attrs['postaction_coverage'] - group.attrs['preaction_coverage']

    return {'weighted':torch.tensor(weighted_reward).float(), \
            'deformable': torch.tensor(deformable_reward).float(), \
            'rigid': torch.tensor(rigid_reward).float(),        \
            'l2':torch.tensor(l2_reward).float(),
            'coverage': torch.tensor(preaction_coverage).float()}


# @profile
class GraspDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_path: str,
                 scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
                 check_validity=False,
                 filter_fn=None,
                 obs_color_jitter=True,
                 use_normalized_coverage=True,
                 replay_buffer_size=2000,
                 fixed_replay_buffer=False,
                 positional_encoding=None,
                 reward_type=None,
                 action_primitives=None,
                 episode_length=None,
                 gamma=0.0,
                 max_pos_num=10000,
                 pix_grasp_dist=16,
                 sample_size=None,
                 category='all',
                 network=None,
                 supervised_training=False,
                 **kwargs):
        
        self.supervised_training = supervised_training
        self.seed = kwargs['seed']
        self.category = category
        self.hdf5_path = hdf5_path
        self.filter_fn = filter_fn
        self.use_normalized_coverage = use_normalized_coverage
        self.rgb_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.2, hue=0.5),
                transforms.RandomAdjustSharpness(1.1, p=0.25),
                ])\
            if obs_color_jitter else lambda x: x
        self.replay_buffer_size = replay_buffer_size
        self.action_primitives = action_primitives
        self.episode_length = episode_length
        self.gamma = gamma
        self.sample_size = sample_size
        self.max_pos_num = max_pos_num  
        self.pix_grasp_dist = pix_grasp_dist
        self.network = network

        if check_validity:
            for k in tqdm(self.keys, desc='Checking validity'):
                self.check_validity(k)

        if self.network != 'online':
            dataset = h5py.File(self.hdf5_path, "r")

        self.keys = self.get_keys()
        print("Number of keys:", len(self.keys))
        self.size = len(self.keys)

        self.scale_factors = np.array(scale_factors)
        self.positional_encoding = positional_encoding

        self.reward_type = reward_type

        if not fixed_replay_buffer:
            self.replay_buffer_size = 100000

        
    def get_keys(self):
        with h5py.File(self.hdf5_path, "r") as dataset:
            if not self.supervised_training:
                min_index = len(dataset) - self.replay_buffer_size
            else:
                min_index = 0

            keys = []
            for key, item in dataset.items():
                category = key.split('_')[0]
                if self.category == 'all' or category == self.category:
                    if self.network == 'prior':
                        count = item.attrs['count']
                        keys.extend([f"{key}/{i+1:05d}" for i in range(count)])
                        self.network = 'prior'
                        self.getitem = self.getitem_prior

                    elif self.network == 'fling':
                        count = dataset['obs'].shape[0]
                        keys = [i for i in range(count)]
                        self.network = 'fling'
                        self.getitem = self.getitem_fling
                        break

                    elif self.network == 'offline':
                        keys.extend([f'{key}/{k}' for k in item.keys()])
                        self.getitem = self.getitem_offline

                    elif self.network == 'online':
                        if int(key.split('_')[-1]) < min_index: continue
                        attrs = item.attrs
                        if self.filter_fn is None or self.filter_fn(attrs) and \
                            ('postaction_weighted_distance' in attrs):
                            keys.append(key)
                        self.getitem = self.getitem_offline
        return keys
    
    def check_validity(self, key):
        with h5py.File(self.hdf5_path, "a") as dataset:
            group = dataset.get(key)
            if 'actions' not in group or 'observations' not in group \
                or 'postaction_coverage' not in group.attrs:
                del dataset[key]
                return

    def __len__(self):
        return len(self.keys)

    def getitem_offline(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            sub_group = dataset.get(self.keys[index])
            rewards_dict = rewards_from_group(sub_group)
            obs_dset = sub_group['obs']
            obs = torch.empty(obs_dset.shape, dtype=torch.float32)
            obs_dset.read_direct(obs.numpy()) 
            weighted_reward = rewards_dict['weighted']
            deformable_reward = rewards_dict['deformable']
            rigid_reward = rewards_dict['rigid']
            l2_reward = rewards_dict['l2']
            coverage_reward = rewards_dict['coverage']
            action = torch.zeros_like(obs[0], dtype=torch.bool)
            action[sub_group.attrs['local_y'], sub_group.attrs['local_z']] = True
            
            return action, weighted_reward, deformable_reward, rigid_reward, l2_reward, coverage_reward, obs
    

    def getitem_prior(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            sub_group = dataset.get(self.keys[index])
            group_key = str(self.keys[index].split('/')[0])
            rewards_dict = rewards_from_group(sub_group)
            obs_dset = dataset[group_key]['obs']
            obs = torch.empty(obs_dset.shape, dtype=torch.float32)
            obs_dset.read_direct(obs.numpy())
            weighted_reward = rewards_dict['weighted']
            deformable_reward = rewards_dict['deformable']
            rigid_reward = rewards_dict['rigid']
            l2_reward = rewards_dict['l2']
            coverage_reward = rewards_dict['coverage']
            action = torch.zeros_like(obs[0], dtype=torch.bool)
            action[sub_group.attrs['y'], sub_group.attrs['z']] = True

        return action, weighted_reward, deformable_reward, rigid_reward, l2_reward, coverage_reward, obs
    
    def getitem_fling(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            obs = torch.from_numpy(dataset['obs'][index])
            deformable_reward = torch.tensor(dataset['deformable_vmap'][index])
            rigid_reward = torch.tensor(dataset['rigid_vmap'][index])
            l2_reward = deformable_reward
            weighted_reward = deformable_reward
            coverage_reward = deformable_reward
            action = (deformable_reward != -torch.inf)
        
        return action, weighted_reward, deformable_reward, rigid_reward, l2_reward, coverage_reward, obs

    
    def __getitem__(self, index):

        action, weighted_reward, deformable_reward, rigid_reward, l2_reward, coverage_reward, obs = self.getitem(index)

        retval = {}
        retval['action'] = action
        retval['weighted_reward'] = weighted_reward
        retval['deformable_reward'] = deformable_reward
        retval['rigid_reward'] = rigid_reward
        retval['l2_reward'] = l2_reward
        retval['coverage_reward'] = coverage_reward
        retval['obs'] = obs

        if self.network == 'prior':
            for key, value in retval.items():
                if np.isnan(value).any() or np.isinf(value).any():
                    print("NaN or Inf detected in sample: ", key)
                    print(f"Retrying index {index}")
                    new_index = np.random.randint(0, len(self.keys))
                    return self.__getitem__(new_index)

        return retval