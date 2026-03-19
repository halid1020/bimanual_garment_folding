from model import UNet
from dataset import KeypointDataset
from utils import get_keypoints, calculate_pck, calculate_acc
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from tqdm import tqdm

def heatmap_loss_fn(pred, target):
    return F.mse_loss(pred, target)

def cls_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target)

num_keypoints = 4
train_ratio = 0.8
batch_size = 32
lr = 1e-3
model_save_path = "best_model.pth"
num_epochs = 20
datasest_path = 'data/keypoints_balanced.hdf5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_ch=3, out_ch=num_keypoints).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = KeypointDataset(dataset_path=datasest_path, heatmap_sigma=5, is_train=True)
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
val_dataset.is_train = False

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_val_loss = float("inf")

def compute_loss_and_metrics(batch, model, device):
    images = batch['image'].to(device)
    label = batch['label'].to(device)
    heatmaps = batch['heatmap'].to(device)
    
    pred_heatmaps, pred_cls = model(images)
    
    loss_1 = heatmap_loss_fn(pred_heatmaps * label[:, None, None, None], heatmaps * label[:, None, None, None])
    loss_2 = 0.01*cls_loss_fn(pred_cls, label[:, None])
    loss = loss_1 + loss_2

    pred_heatmaps = (pred_heatmaps.cpu().detach().numpy())[label.cpu().long().numpy()]
    pred_coords = get_keypoints(pred_heatmaps)
    true_keypoints = (batch['keypoints'].cpu().detach().numpy())[label.cpu().long().numpy()]

    pck = calculate_pck(pred_coords, true_keypoints)
    acc = calculate_acc(pred_cls.cpu().detach().numpy(), label.cpu().detach().numpy())

    return loss, loss_1, loss_2, pck, acc

if __name__ == "__main__":
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, heatmap_loss, cls_loss, train_pck, train_acc = 0, 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", unit="batch")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            loss, loss_1, loss_2, pck, acc = compute_loss_and_metrics(batch, model, device)
            
            loss.backward()
            optimizer.step()

            train_loss += loss
            heatmap_loss += loss_1
            cls_loss += loss_2
            train_acc += acc
            train_pck += pck

            pbar.set_postfix(loss=loss.item(), heatmap_loss=loss_1.item(), cls_loss=loss_2.item(), pck=pck, acc=acc)

        train_loss /= len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} Heatmap Loss: {heatmap_loss/len(train_loader):.4f} Class Loss: {cls_loss/len(train_loader):.4f} Train PCK: {train_pck/len(train_loader):.4f} Train ACC: {train_acc/len(train_loader):.4f}")

        model.eval()
        val_loss, heatmap_loss, cls_loss, val_pck, val_acc = 0, 0, 0, 0, 0
        pbar = tqdm(val_loader, desc=f"[Epoch {epoch}] Validation", unit="batch")
        
        with torch.no_grad():
            for batch in pbar:
                loss, loss_1, loss_2, pck, acc = compute_loss_and_metrics(batch, model, device)

                val_loss += loss
                heatmap_loss += loss_1
                cls_loss += loss_2
                val_pck += pck
                val_acc += acc

                pbar.set_postfix(loss=loss.item(), heatmap_loss=loss_1.item(), cls_loss=loss_2.item(), pck=pck, acc=acc)

        val_loss /= len(val_loader)
        val_pck /= len(val_loader)
        val_acc /= len(val_loader)
        heatmap_loss /= len(val_loader)
        cls_loss /= len(val_loader)
        
        print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f} Heatmap Loss: {heatmap_loss:.4f} Class Loss: {cls_loss:.4f} Validation PCK: {val_pck:.4f} Validation ACC: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_save_path)
            print(f"Saved best model at epoch {epoch} with val_loss {val_loss:.4f}")
