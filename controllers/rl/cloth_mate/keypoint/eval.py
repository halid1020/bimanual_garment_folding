from model import UNet
from dataset import KeypointDataset
from utils import get_keypoints, visualize, calculate_pck, calculate_acc
from torch.utils.data import DataLoader, random_split
import torch, cv2
import torch.nn.functional as F
import numpy as np 


def heatmap_loss(pred, target):
    return F.mse_loss(pred, target)

num_keypoints = 4
train_ratio = 0.8
batch_size = 4
lr = 1e-3
model_save_path = "best_model.pth"
num_epochs = 100
datasest_path = 'data/keypoints_balanced.hdf5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_ch=3, out_ch=num_keypoints).to(device)
checkpoint = torch.load("best_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

dataset = KeypointDataset(dataset_path=datasest_path, heatmap_sigma=5, is_train=False)

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42) 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
val_dataset.is_train = False

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model.eval()

eval_pck = 0
eval_cls = 0

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        visibility = batch['visibility'].bool()
        pred_heatmap, pred_cls = model(images)
        label = batch['label'].to(device).cpu().long().numpy()
        true_coords = batch['keypoints'].cpu().numpy()
        pred_coords = get_keypoints(pred_heatmap.cpu().numpy(), threshold=0.2)
        pck = calculate_pck(pred_coords[visibility], true_coords[visibility], threshold=5)
        
        pred_cls = pred_cls.cpu().numpy()
        acc = calculate_acc(pred_cls, label, 0.475)

        eval_cls += acc
        eval_pck += pck

        # visual_image = []
        # for i in range(images.shape[0]):
        #     img = images[i].cpu().numpy().transpose(1, 2, 0)
        #     img = np.clip(img * 255, 0, 255).astype(np.uint8)
        #     img = np.ascontiguousarray(img)
        #     cv2.putText(img, f"{pred_cls[i].item():.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #     kpt = pred_coords[i]
        #     heatmap = pred_heatmap[i].cpu().numpy()
        #     visual_image.append(visualize(visualize(img, kpt),true_coords[i], color=(0, 255, 0)))
        # cv2.imshow("Visualize", np.concatenate(visual_image, axis=1))
        # cv2.waitKey(0)

    print(f"Eval Accuracy: {eval_cls / len(val_loader):.4f}")
    print(f"Eval PCK: {eval_pck / len(val_loader):.4f}")



