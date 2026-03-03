import cv2, os
import numpy as np
from skimage.draw import line

def calculate_acc(pred_cls, true_cls, threshold=0.5):
    # pred_cls: [N, 1]
    # true_cls: [N, 1]
    pred_cls = pred_cls.squeeze()>threshold
    correct = np.sum(pred_cls == true_cls.squeeze())
    total = len(pred_cls)
    return correct / total

def is_line_all_ones(image, start, end):

    x1, y1 = start[1], start[0]
    x2, y2 = end[1], end[0]

    rr, cc = line(int(y1), int(x1), int(y2), int(x2))
    rr = np.clip(rr, 0, image.shape[0] - 1)
    cc = np.clip(cc, 0, image.shape[1] - 1)
    return np.any(image[rr, cc] == 0)

def calculate_pck(pred_coords, true_coords, threshold=5):
    # pred_coords: [N, C, 2]
    # true_coords: [N, C, 2]
    mask = np.all(pred_coords != -1, axis=-1) & np.all(true_coords != -1, axis=-1)
    distances = np.sqrt(np.sum((pred_coords - true_coords)**2, axis=-1))
    correct = (distances <= threshold).astype(float)

    if mask.sum() == 0:
        return 0
    return np.mean(correct[mask])

def get_keypoints(heatmap, threshold=0):
    if heatmap.ndim == 4: # N, C, H, W
        heatmap = heatmap.reshape(-1, *heatmap.shape[2:])
    keypoints = []
    for c in range(heatmap.shape[0]):
        hm = heatmap[c]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        if hm[y, x] > threshold:
            keypoints.append((x, y))
        else:
            keypoints.append((-1, -1))
    keypoints = np.array(keypoints)
    if len(keypoints)>4:
        keypoints = keypoints.reshape(-1, 4, 2)

    return keypoints

def visualize(image_np, keypoints, heatmap=None, wait=True, save_path=None, gui=False, color=(0, 0, 255)):

    image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
    image_np = image_np.copy() 
    keypoints_np = keypoints.astype(int)
    for x, y in keypoints_np:
        cv2.circle(image_np, (x, y), radius=3, color=color, thickness=-1)

    blender = image_np.copy()

    if heatmap is not None:
        for idx in range(len(keypoints)):
            # if torch.all(heatmap[idx] == 0): continue
            heatmap_np = heatmap[idx].cpu().numpy()
            heatmap_img = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_img = heatmap_img.astype(np.uint8)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            blender = np.concatenate([blender, cv2.addWeighted(image_np, 0.7, heatmap_img, 0.3, 0)], axis=1)
    else:
        blender = image_np

    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, f"image.png"), blender)
    elif gui:
        cv2.imshow('Image + Keypoints + Heatmap', blender)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
    else:
        return blender