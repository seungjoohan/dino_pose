import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
from transformers import AutoImageProcessor
from src.utils import KeyPoints, com_weights, KeyPointConnections, read_annotation

def argmax_ind(heatmap):
    """
    Compute the maximum location of a heatmap
    Returns [x, y, confidencescore]
    """
    ind = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return ind[0], ind[1], heatmap[ind[0], ind[1]]

def weighted_max_loc(heatmap, target_size=(224,224)):
    """
    Compute the weighted maximum location of a heatmap
    """
    heatmap = np.squeeze(heatmap)
    center_x, center_y, _ = argmax_ind(heatmap)
    padding = 2
    min_x = max(0, center_x - padding)
    max_x = min(heatmap.shape[1], center_x + padding + 1)  # +1 for inclusive range
    min_y = max(0, center_y - padding)
    max_y = min(heatmap.shape[0], center_y + padding + 1)  # +1 for inclusive range
    cropped_heatmap = heatmap[min_y:max_y, min_x:max_x]
    
    loc_x = np.sum((0.5 + np.arange(min_x, max_x)) * np.sum(cropped_heatmap, axis=0)) / np.sum(cropped_heatmap)
    loc_y = np.sum((0.5 + np.arange(min_y, max_y)) * np.sum(cropped_heatmap, axis=1)) / np.sum(cropped_heatmap)
    loc_x = loc_x / heatmap.shape[1] * target_size[0]
    loc_y = loc_y / heatmap.shape[0] * target_size[1]

    return loc_x, loc_y

def get_keypoints_from_heatmaps(heatmaps, target_size=(224, 224)):
    """
    Extract keypoints from heatmaps and scale them to target image size
    """
    heatmaps = heatmaps.squeeze()
    num_kp, _, _ = heatmaps.shape
    return [weighted_max_loc(heatmaps[idx, :, :], target_size) for idx in range(num_kp)]
    
def get_keypoints_from_heatmaps_batch(heatmaps_batch, target_size=(224, 224)):
    """
    Process a batch of heatmaps to extract keypoint coordinates
    """
    batch_size, _, _, _ = heatmaps_batch.shape
    return np.array([get_keypoints_from_heatmaps(heatmaps_batch[idx], target_size) for idx in range(batch_size)])

def compute_pckh(pred_keypoints, target_keypoints, threshold_ratio=0.5):
    """
    Compute PCKh metric
    if head and neck are not visible use hip distances - most likely to be visible when head and neck are not visible
    """
    correct_count = 0
    total_count = 0
    num_kp = pred_keypoints.shape[0]
    top_gt = target_keypoints[KeyPoints.TOP.value]
    neck_gt = target_keypoints[KeyPoints.NECK.value]
    if top_gt[2] == 0 or neck_gt[2] == 0:
        top_gt = target_keypoints[KeyPoints.RIGHT_HIP.value]
        neck_gt = target_keypoints[KeyPoints.LEFT_HIP.value]
        if top_gt[2] == 0 or neck_gt[2] == 0:
            print("TOP, NECK, RIGHT_HIP, LEFT_HIP are not visible. Check cropping")
            import pdb;pdb.set_trace()

    threshold_dist = np.sqrt((top_gt[0] - neck_gt[0])**2 + (top_gt[1] - neck_gt[1])**2) * threshold_ratio

    for k in range(num_kp):
        if target_keypoints[k, 2] == 0:
            continue
        dist = np.sqrt(np.sum((pred_keypoints[k] - target_keypoints[k][:2])**2))
        if dist < threshold_dist:
            correct_count += 1
        total_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


def compute_pckh_dataset(model, image_dir, annotation_path, model_name, device, threshold_ratio=0.5):
    img_info, anns = read_annotation(annotation_path)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pckh_2d = []
    pckh_3d = []
    for i, idx in enumerate(img_info):
        img_path = os.path.join(image_dir, f"{idx['file_name']}")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        inputs = image_processor(images=img, return_tensors="pt")
        input = inputs.pixel_values.to(device)
        heatmaps, z_coords = model(input)
        pred_kps = get_keypoints_from_heatmaps(heatmaps.cpu().detach().numpy(), (width, height))
        target_kps = np.array(anns[i]['keypoints']).reshape(-1, 3)
        target_z_coords = np.array(anns[i]['keypoints_z'])
        pckh_2d.append(compute_pckh(np.array(pred_kps), target_kps, threshold_ratio))
        pckh_3d.append(compute_pckh_z(z_coords.cpu().detach().numpy(), target_z_coords, target_kps, threshold_ratio))
    return np.mean(pckh_2d), np.mean(pckh_3d)

def compute_pckh_z_batch(pred_z_coords_batch, target_z_coords_batch, target_keypoints_batch, threshold=0.5):
    """
    Compute PCKh metric for a batch of z-coordinates
    """
    batch_size = pred_z_coords_batch.shape[0]
    entry_pckh = []
    for b in range(batch_size):
        pred_z_coords = pred_z_coords_batch[b]
        target_z_coords = target_z_coords_batch[b]
        target_keypoints = target_keypoints_batch[b]
        entry_pckh.append(compute_pckh_z(pred_z_coords, target_z_coords, target_keypoints, threshold))
    
    return np.mean(entry_pckh)

def compute_pckh_z(pred_z_coords, target_z_coords, target_keypoints, threshold=0.5):
    """
    Compute PCKh metric for z-coordinates
    """
    # scale 2d visible keypoints - very unlikely to have non visible keypoints in 3d (dataset characterstic)
    vis_mask = np.array([1 if point[2] != 0 else 0 for point in target_keypoints])
    x_coords = target_keypoints[:, 0] * vis_mask
    y_coords = target_keypoints[:, 1] * vis_mask
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    scale = (x_std + y_std) / 2
    
    x_coords_scaled = x_coords / scale
    y_coords_scaled = y_coords / scale

    x_coords_centered = x_coords_scaled - (x_coords_scaled * com_weights)
    y_coords_centered = y_coords_scaled - (y_coords_scaled * com_weights)

    # approximating distiance to camera
    d_to_camera = np.max((5, np.max(np.abs(target_z_coords[target_z_coords < 0] * 1.1))))

    center = [0, 0]
    true_x = (x_coords_centered - center[0]) * (target_z_coords + d_to_camera) / d_to_camera
    true_y = (y_coords_centered - center[1]) * (target_z_coords + d_to_camera) / d_to_camera

    threshold_dist = threshold * np.sqrt((true_x[1] - true_x[0])**2 + (true_y[1] - true_y[0])**2 + (target_z_coords[1] - target_z_coords[0])**2)
    pred_distances = np.abs(pred_z_coords - target_z_coords)
    
    return np.nanmean(pred_distances < threshold_dist)
    
def plot_keypoints(image, pred_heatmaps, keypoint_label=True, figsize=(12, 8)):
    """
    Plot keypoints on an image
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import numpy as np
    
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            # Denormalize the image if needed
            img_np = image.cpu().detach().numpy()
            img_np = img_np.transpose(1, 2, 0)  # Convert to HWC
            
            # Denormalize if image values are in [0,1]
            if img_np.max() <= 1.0:
                img_np = img_np * 255
                
            img_np = img_np.astype(np.uint8)
        else:
            raise ValueError("Image tensor should be in [C, H, W] format with C=3")
    else:
        img_np = np.array(image)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)
    
    # Extract keypoints from heatmaps
    width, height = image.size
    image_size = (width, height)

    pred_keypoints = get_keypoints_from_heatmaps(pred_heatmaps, image_size)
    
    # Get max values from heatmaps for confidence visualization
    confidences = []
    for k in range(pred_heatmaps.shape[2]):
        max_val = np.max(pred_heatmaps[:, :, k])
        confidences.append(max_val)
    
    # Plot predicted keypoints
    for i, (x, y) in enumerate(pred_keypoints):
        confidence = confidences[i]
        
        # Scale circle size based on confidence
        circle_size = max(4, min(10, confidence * 12))
        
        # Add keypoint
        circle = Circle((x, y), circle_size, color='red', alpha=0.7)
        ax.add_patch(circle)
        
        if keypoint_label:
            # Add keypoint index label
            ax.text(x + 5, y + 5, KeyPoints(i).name, fontsize=8, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))
    
    # Draw skeleton
    for link in KeyPointConnections.links:
        from_idx = link['from'].value
        to_idx = link['to'].value
        
        # Check if both keypoints are within image boundaries
        from_pt = pred_keypoints[from_idx]
        to_pt = pred_keypoints[to_idx]
        
        if (0 <= from_pt[0] <= width and 0 <= from_pt[1] <= height and
            0 <= to_pt[0] <= width and 0 <= to_pt[1] <= height):
            ax.plot([from_pt[0], to_pt[0]], [from_pt[1], to_pt[1]], 
                   color=link['color'], linewidth=2, alpha=0.7)
    
    ax.set_title("Predicted Keypoints")
    plt.axis('off')
    plt.show()
    return fig

def __main__():
    import os
    import sys
    # Add the project root directory to Python's path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from model.dinov2_pose import Dinov2PoseModel
    from config.config import get_default_configs
    from data_loader.data_loader import create_dataloaders

    model = Dinov2PoseModel()

    config_dataset, config_training, config_preproc, config_model = get_default_configs()
    
    # Create dataloader
    print(f"Creating dataloader for {config_dataset['val_images_dir']}...")
    train_dataloader = create_dataloaders(
        config_preproc=config_preproc,
        config_model=config_model,
        images_dir_path=config_dataset['val_images_dir'],
        annotation_json_path=config_dataset['val_annotation_json'],
        batch_size=config_training['batch_size'],
        num_workers=config_training['multiprocessing_num']
    )

    for batch in train_dataloader:
        pred_heatmaps, pred_z_coords = model(batch['image'])
        np_pred_heatmaps = pred_heatmaps.cpu().detach().numpy()
        np_pred_z_coords = pred_z_coords.cpu().detach().numpy()
        
        image_height = batch['image'].shape[2]
        image_width = batch['image'].shape[3]
        image_size = (image_height, image_width)
        
        # Extract keypoints, scaling to the image size
        pred_keypoints_batch = get_keypoints_from_heatmaps_batch(np_pred_heatmaps, image_size)
        target_keypoints_batch = batch['2d_keypoints'].cpu().detach().numpy()
        target_z_coords = batch['z_coords'].cpu().detach().numpy()

        pckh_2d = compute_pckh_batch(pred_keypoints_batch, target_keypoints_batch)
        print(f"PCKh (2D): {pckh_2d:.4f}")
        pckh_3d = compute_pckh_z_batch(np_pred_z_coords, target_z_coords, target_keypoints_batch)
        print(f"PCKh (3D): {pckh_3d:.4f}")
        
    
if __name__ == "__main__":
    __main__()