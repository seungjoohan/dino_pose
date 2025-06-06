import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
from transformers import AutoImageProcessor
from src.utils import KeyPoints, com_weights, KeyPointConnections, read_annotation
from model.model_utils import resolve_model_name

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
    center_y, center_x, _ = argmax_ind(heatmap)
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
            # No reference points available - return 0.0
            print("Warning: TOP, NECK, RIGHT_HIP, LEFT_HIP are not visible. Cannot compute PCKh.")
            return 0.0  

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
    print(f"Loading dataset to evaluate model performance...")
    img_info, anns = read_annotation(annotation_path)
    # Resolve model name (family name -> actual HuggingFace model name)
    actual_model_name = resolve_model_name(model_name)
    image_processor = AutoImageProcessor.from_pretrained(actual_model_name)
    pckh_2d = []
    pckh_3d = []
    print(f"Computing PCKh for {len(img_info)} images...")
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

def convert_z_to_annotation_space(pred_z_coords, target_z_coords, target_keypoints):
    """
    Convert predicted z_coords from normalized space back to annotation coordinate space
    """
    # Calculate the same scale and CoM adjustment as used in data preprocessing
    vis_mask = np.array([1 if point[2] != 0 else 0 for point in target_keypoints])
    x_coords = target_keypoints[:, 0] * vis_mask
    y_coords = target_keypoints[:, 1] * vis_mask
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    scale = (x_std + y_std) / 2
    if scale == 0:
        scale = np.finfo(np.float32).eps
    
    # Calculate CoM adjustment for target_z_coords
    com_adjustment = np.sum(target_z_coords * com_weights)
    
    # Convert predicted z_coords back to annotation coordinate system
    # Reverse the preprocessing: pred_z_coords * scale + com_adjustment
    pred_z_coords_annotation = pred_z_coords * scale + com_adjustment
    
    return pred_z_coords_annotation, scale, com_adjustment

def compute_pckh_z(pred_z_coords, target_z_coords, target_keypoints, threshold=0.5):
    """
    Compute PCKh metric for z-coordinates
    """
    # Convert predicted z_coords back to annotation coordinate system
    pred_z_coords_annotation, scale, com_adjustment = convert_z_to_annotation_space(
        pred_z_coords, target_z_coords, target_keypoints
    )
    
    # Compute pckh
    pred_distances = np.abs(pred_z_coords_annotation - target_z_coords)
    top_idx = 0
    neck_idx = 1
    
    if (target_keypoints[top_idx, 2] > 0 and target_keypoints[neck_idx, 2] > 0):
        # 3D distance between top and neck
        x_dist = target_keypoints[top_idx, 0] - target_keypoints[neck_idx, 0]
        y_dist = target_keypoints[top_idx, 1] - target_keypoints[neck_idx, 1]
        z_dist = target_z_coords[top_idx] - target_z_coords[neck_idx]
        reference_dist_3d = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    else:
        # Fallback to using standard deviation of z coordinates
        reference_dist_3d = np.std(target_z_coords[target_z_coords != 0])
    
    threshold_dist = threshold * reference_dist_3d
    
    correct_predictions = pred_distances < threshold_dist
    
    return np.nanmean(correct_predictions)
    
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
        circle_size = 4 #max(4, min(10, confidence * 12))
        
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
    return fig

def plot_3d_keypoints(image, pred_heatmaps, pred_z_coords, keypoint_label=True, figsize=(10, 8)):
    """
    Plot 3D keypoints using predicted z_coords (converted to annotation space)
    Note: Since we don't have ground truth z coordinates in demo mode,
    we'll use a simplified conversion based on 2D keypoint spread
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            img_np = image.cpu().detach().numpy()
            img_np = img_np.transpose(1, 2, 0)  # Convert to HWC
            if img_np.max() <= 1.0:
                img_np = img_np * 255
            img_np = img_np.astype(np.uint8)
        else:
            raise ValueError("Image tensor should be in [C, H, W] format with C=3")
    else:
        img_np = np.array(image)
    
    # Extract 2D keypoints from heatmaps
    width, height = image.size
    image_size = (width, height)
    pred_keypoints = get_keypoints_from_heatmaps(pred_heatmaps, image_size)
    
    # Convert predicted z_coords to approximate annotation space
    # Since we don't have ground truth for scaling, use reasonable estimates
    x_coords = [kp[0] for kp in pred_keypoints]
    y_coords = [kp[1] for kp in pred_keypoints]
    x_std = np.std(x_coords) if len(x_coords) > 1 else 50.0
    y_std = np.std(y_coords) if len(y_coords) > 1 else 50.0
    scale = (x_std + y_std) / 2
    
    # Convert z_coords to approximate annotation space (scale up from normalized)
    z_coords_annotation = pred_z_coords * scale
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get confidences from heatmaps
    confidences = []
    for k in range(pred_heatmaps.shape[2]):
        max_val = np.max(pred_heatmaps[:, :, k])
        confidences.append(max_val)
    
    # Plot keypoints in 3D
    for i, ((x, y), z) in enumerate(zip(pred_keypoints, z_coords_annotation)):
        confidence = confidences[i]
        
        # Scale point size based on confidence
        point_size = max(20, min(100, confidence * 150))
        
        # Color by confidence
        color = plt.cm.viridis(confidence)
        
        ax.scatter(x, y, z, s=point_size, c=[color], alpha=0.8)
        
        if keypoint_label:
            ax.text(x, y, z, f'{i}:{KeyPoints(i).name[:4]}', fontsize=8)
    
    # Draw skeleton connections in 3D
    for link in KeyPointConnections.links:
        from_idx = link['from'].value
        to_idx = link['to'].value
        
        if from_idx < len(pred_keypoints) and to_idx < len(pred_keypoints):
            from_pt = pred_keypoints[from_idx]
            to_pt = pred_keypoints[to_idx]
            from_z = z_coords_annotation[from_idx]
            to_z = z_coords_annotation[to_idx]
            
            # Check if both points are within reasonable bounds
            if (0 <= from_pt[0] <= width and 0 <= from_pt[1] <= height and
                0 <= to_pt[0] <= width and 0 <= to_pt[1] <= height):
                ax.plot([from_pt[0], to_pt[0]], 
                       [from_pt[1], to_pt[1]], 
                       [from_z, to_z], 
                       color=link['color'], linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (depth)')
    ax.set_title('Predicted 3D Keypoints')
    
    # Set equal aspect ratio for better visualization
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords_annotation) - min(z_coords_annotation)
    max_range = max(x_range, y_range, z_range) / 2.0
    
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords_annotation) + min(z_coords_annotation)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    return fig

def plot_keypoints_combined(image, pred_heatmaps, pred_z_coords, keypoint_label=True, figsize=(20, 8)):
    """
    Plot both 2D and 3D keypoints side by side
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Circle
    import numpy as np
    
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            img_np = image.cpu().detach().numpy()
            img_np = img_np.transpose(1, 2, 0)  # Convert to HWC
            if img_np.max() <= 1.0:
                img_np = img_np * 255
            img_np = img_np.astype(np.uint8)
        else:
            raise ValueError("Image tensor should be in [C, H, W] format with C=3")
    else:
        img_np = np.array(image)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Extract keypoints
    width, height = image.size
    image_size = (width, height)
    pred_keypoints = get_keypoints_from_heatmaps(pred_heatmaps, image_size)
    
    # Get confidences
    confidences = []
    for k in range(pred_heatmaps.shape[2]):
        max_val = np.max(pred_heatmaps[:, :, k])
        confidences.append(max_val)
    
    # Subplot 1: 2D keypoints
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_np)
    
    # Plot 2D keypoints
    for i, (x, y) in enumerate(pred_keypoints):
        confidence = confidences[i]
        circle_size = max(4, min(10, confidence * 12))
        circle = Circle((x, y), circle_size, color='red', alpha=0.7)
        ax1.add_patch(circle)
        
        if keypoint_label:
            ax1.text(x + 5, y + 5, KeyPoints(i).name, fontsize=8, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))
    
    # Draw 2D skeleton
    for link in KeyPointConnections.links:
        from_idx = link['from'].value
        to_idx = link['to'].value
        
        if from_idx < len(pred_keypoints) and to_idx < len(pred_keypoints):
            from_pt = pred_keypoints[from_idx]
            to_pt = pred_keypoints[to_idx]
            
            if (0 <= from_pt[0] <= width and 0 <= from_pt[1] <= height and
                0 <= to_pt[0] <= width and 0 <= to_pt[1] <= height):
                ax1.plot([from_pt[0], to_pt[0]], [from_pt[1], to_pt[1]], 
                        color=link['color'], linewidth=2, alpha=0.7)
    
    ax1.set_title("2D Keypoints")
    ax1.axis('off')
    
    # Subplot 2: 3D keypoints
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Convert z_coords to approximate annotation space
    x_coords = [kp[0] for kp in pred_keypoints]
    y_coords = [kp[1] for kp in pred_keypoints]
    x_std = np.std(x_coords) if len(x_coords) > 1 else 50.0
    y_std = np.std(y_coords) if len(y_coords) > 1 else 50.0
    scale = (x_std + y_std) / 2
    z_coords_annotation = pred_z_coords * scale
    
    # Plot 3D keypoints
    for i, ((x, y), z) in enumerate(zip(pred_keypoints, z_coords_annotation)):
        confidence = confidences[i]
        point_size = max(20, min(100, confidence * 150))
        color = plt.cm.viridis(confidence)
        
        ax2.scatter(x, y, z, s=point_size, c=[color], alpha=0.8)
        
        if keypoint_label:
            ax2.text(x, y, z, f'{i}:{KeyPoints(i).name[:4]}', fontsize=8)
    
    # Draw 3D skeleton
    for link in KeyPointConnections.links:
        from_idx = link['from'].value
        to_idx = link['to'].value
        
        if from_idx < len(pred_keypoints) and to_idx < len(pred_keypoints):
            from_pt = pred_keypoints[from_idx]
            to_pt = pred_keypoints[to_idx]
            from_z = z_coords_annotation[from_idx]
            to_z = z_coords_annotation[to_idx]
            
            if (0 <= from_pt[0] <= width and 0 <= from_pt[1] <= height and
                0 <= to_pt[0] <= width and 0 <= to_pt[1] <= height):
                ax2.plot([from_pt[0], to_pt[0]], 
                        [from_pt[1], to_pt[1]], 
                        [from_z, to_z], 
                        color=link['color'], linewidth=2, alpha=0.7)
    
    # Set 3D plot properties
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_zlabel('Z (depth)')
    ax2.set_title('3D Keypoints')
    
    # Set equal aspect ratio
    x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 100
    y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 100
    z_range = max(z_coords_annotation) - min(z_coords_annotation) if len(z_coords_annotation) > 1 else 100
    max_range = max(x_range, y_range, z_range) / 2.0
    
    mid_x = (max(x_coords) + min(x_coords)) * 0.5 if len(x_coords) > 1 else x_coords[0]
    mid_y = (max(y_coords) + min(y_coords)) * 0.5 if len(y_coords) > 1 else y_coords[0]
    mid_z = (max(z_coords_annotation) + min(z_coords_annotation)) * 0.5 if len(z_coords_annotation) > 1 else z_coords_annotation[0]
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def __main__():
    pass
        
    
if __name__ == "__main__":
    __main__()