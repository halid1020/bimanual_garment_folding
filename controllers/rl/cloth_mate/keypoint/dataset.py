import cv2
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import visualize

class KeypointDataset(Dataset):
    def __init__(self, dataset_path, heatmap_sigma=5, is_train=True, img_size=128):
        self.dataset_path = dataset_path
        self.sigma = heatmap_sigma
        self.is_train = is_train
        self.img_size = img_size

        self.affine_params = {
            "max_rotate": 5,       
            "max_translate": 0.1,   
            "min_scale": 0.9,       
            "max_scale": 1.1,    
            "max_shear": 10       
        }

        self.color_transform = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ) if is_train else None

        self.dataset = h5py.File(dataset_path, 'r')
        
        self.index = [key for key in self.dataset.keys() if 'image' in self.dataset[key]]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key = self.index[idx]
        group = self.dataset[key]

        image = group['image'][()][:,:,:3]          # [H, W, C]
        if key.split('_')[0] == 'neg':
            label = False
            keypoints = -np.ones((4, 2), dtype=np.float32)
            visibility = np.zeros((4,), dtype=np.float32)
        else:
            label =True
            keypoints = group['keypoints'][()]  # [N, 2] (x, y)
            visibility = group['visibility'][()]  # [N,]
            
        image = (image*255).astype(np.uint8)
        keypoints = keypoints[:, ::-1].astype(np.float32)

        if self.is_train:
            image, keypoints, visibility = self.apply_geometric_augmentation(image, keypoints, visibility)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # [C, H, W]

        if self.is_train and self.color_transform:
            image_tensor = self.color_transform(image_tensor)

        heatmap = self._generate_heatmap(
            image_shape=(self.img_size, self.img_size),
            keypoints=np.concatenate([keypoints, visibility[:, None]], axis=-1)
        )
        heatmap = torch.from_numpy(heatmap).float().permute(2, 0, 1)  # [C, H, W]

        return {
            "image": image_tensor,
            "heatmap": heatmap,
            "keypoints": torch.from_numpy(keypoints).float(),
            "visibility": torch.from_numpy(visibility).float(),
            "label": torch.tensor(label).float(),
        }

    def apply_geometric_augmentation(self, image, keypoints=None, visibility=None):
        H, W = image.shape[:2]
        
        angle = np.random.uniform(-self.affine_params["max_rotate"], self.affine_params["max_rotate"])
        translate = np.random.uniform(-self.affine_params['max_translate'], self.affine_params['max_translate'], 2) * [W, H]
        scale = np.random.uniform(self.affine_params["min_scale"], self.affine_params["max_scale"])
        shear = np.random.uniform(-self.affine_params["max_shear"], self.affine_params["max_shear"])

        M = self._get_affine_matrix(angle, translate, scale, shear, H, W)
        
        image = cv2.warpAffine(
            image, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        if keypoints is None or visibility is None:
            return image

        ones = np.ones((len(keypoints), 1))
        homogenous_coords = np.concatenate([keypoints, ones], axis=1)  # [N, 3]
        trans_coords = (M @ homogenous_coords.T).T  # [N, 2]
        
        valid_x = (trans_coords[:, 0] >= 0) & (trans_coords[:, 0] < W)
        valid_y = (trans_coords[:, 1] >= 0) & (trans_coords[:, 1] < H)
        valid_mask = (valid_x & valid_y).astype(visibility.dtype)

        visibility = visibility * valid_mask
        
        trans_coords[:, 0] = np.clip(trans_coords[:, 0], 0, W-1)
        trans_coords[:, 1] = np.clip(trans_coords[:, 1], 0, H-1)
        
        return image, trans_coords, visibility

    def _get_affine_matrix(self, angle, translate, scale, shear, H, W):

        center = (W/2, H/2)
        M_rotate = cv2.getRotationMatrix2D(center, angle, scale)
        
        shear_rad = np.deg2rad(shear)
        shear_mat = np.array([
            [1, np.tan(shear_rad), 0],
            [np.tan(shear_rad), 1, 0]
        ])
        
        M_rotate_h = np.vstack([M_rotate, [0, 0, 1]])  # 3x3
        shear_mat_h = np.vstack([shear_mat, [0, 0, 1]])  # 3x3

        M = (shear_mat_h @ M_rotate_h)[:2]  # 2x3
        
        M[:, 2] += translate
        
        return M

    def _generate_heatmap(self, image_shape, keypoints):

        h, w = image_shape
        num_kps = len(keypoints)
        heatmap = np.zeros((h, w, num_kps), dtype=np.float32)
        
        for i in range(num_kps):
            x, y, vis = keypoints[i]
            if vis < 0.5:
                continue
                
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            dist = (xx - x)**2 + (yy - y)**2
            gaussian = np.exp(-dist / (2 * self.sigma**2))
            
            heatmap[..., i] = gaussian / gaussian.max()
            
        return heatmap

if __name__ == "__main__":
    dataset = KeypointDataset(dataset_path='data/keypoints_balanced.hdf5', heatmap_sigma=5, is_train=True)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for sample in dataloader:
        print(sample.keys())
        print(sample['label'])
        exit()