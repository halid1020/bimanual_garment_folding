import h5py
import numpy as np
import cv2
import argparse

def visualize_sample_opencv(sample_group: h5py.Group):

    image = (sample_group['image'][()] * 255).astype(np.uint8)[:,:,:3]

    if 'keypoints' not in sample_group:
        return image

    keypoints = sample_group['keypoints'][()]
    visibility = sample_group['visibility'][()]
    
    display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    h, w = display_img.shape[:2]
    
    for idx, (y, x) in enumerate(keypoints):
        if visibility[idx]:
            color = (0, 255, 0) 
            radius = 3
            cv2.circle(display_img, (int(x), int(y)), radius, color, -1)
            
            label = ['TL','TR','BL','BR'][idx]
            cv2.putText(display_img, label, 
                       (int(x)+10, int(y)+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1, cv2.LINE_AA)

    text = f"Distance: {sample_group.attrs['preaction_weighted_distance']:.4f}"
    cv2.putText(display_img, text, (10, h-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    
    return display_img

def auto_visualize(hdf5_path: str, sample_count=500):
    with h5py.File(hdf5_path, 'r') as f:
        all_samples = [s.decode('utf-8') for s in f['samples']]
        selected = np.random.choice(all_samples, min(sample_count, len(all_samples)), False)
        
        for idx, sample_id in enumerate(selected):
            group = f[sample_id]
            display_img = visualize_sample_opencv(group)
            
            winname = f"Sample {idx+1}/{len(selected)}"
            cv2.imshow(winname, display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def dataset_statistics(hdf5_path: str):
    with h5py.File(hdf5_path, 'r') as f:
        print("\n=== Dataset Statistics ===")
        print(f"Total Samples: {len(f['samples'])}")
        
        visibility_stats = np.zeros(4, dtype=int)
        valid_samples = 0
        
        for sample_id in f['samples']:
            group = f[sample_id.decode('utf-8')]
            visibility = group['visibility_mask'][()]
            visibility_stats += visibility
            if np.sum(visibility) >= 2:
                valid_samples += 1
        
        print("\nKeypoint Visibility:")
        labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
        for name, count in zip(labels, visibility_stats):
            print(f"{name}: {count} ({count/len(f['samples'])*100:.1f}%)")
        
        print(f"\nValid Samples (≥2 visible): {valid_samples} ({valid_samples/len(f['samples'])*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenCV Dataset Visualization Tool')
    parser.add_argument('filepath', help='Path to the HDF5 file')
    parser.add_argument('--stats', action='store_true', help='Display statistics')
    args = parser.parse_args()

    if args.stats:
        dataset_statistics(args.filepath)
    else:
        auto_visualize(args.filepath)
        print("Visualization completed. Press any key to switch images")