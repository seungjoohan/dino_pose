import os
import torch
import numpy as np
import math
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from pycocotools.coco import COCO
from PIL import Image
from data_loader.data_augmentation import *
from src.utils import com_weights



class PoseDataset(Dataset):
    """
    PyTorch Dataset for human pose estimation using COCO format annotations
    """
    def __init__(self, images_dir_path, annotation_json_path, image_processor, num_model_keypoints, config_preproc, config_model,transform=True):
        self.images_dir_path = images_dir_path
        self.annotation_json_path = annotation_json_path
        self.image_processor = image_processor
        self.num_model_keypoints = num_model_keypoints
        self.config_preproc = config_preproc
        self.transform = transform
        self.config_model = config_model
        
        # Load COCO annotations
        self.coco = COCO(annotation_json_path)
        
        # Get image IDs
        self.img_ids = self.coco.getImgIds()
            
        # Calculate number of keypoints
        first_ann = list(self.coco.anns.values())[0]
        self.num_keypoints = len(first_ann["keypoints"]) // 3

        if self.num_model_keypoints != self.num_keypoints:
            raise ValueError(f"Number of model keypoints ({self.num_model_keypoints}) does not match number of keypoints in annotations ({self.num_keypoints})")
        
        print(f"Loaded dataset with {len(self.img_ids)} images and {self.num_keypoints} keypoints")
        
    def __len__(self):
        return len(self.img_ids)
    
    def preprocess_image(self, image):
        """Preprocess image for DINOv2 model input"""
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs.pixel_values.squeeze(0)  
    
    def keypoints_to_heatmaps(self, image, keypoints, target_size, num_keypoints=24):
        """
        Convert keypoints to heatmaps. Apply gaussian smoothing
        """
        heatmap = np.zeros((image.size[0], image.size[1], num_keypoints))

        # Gaussian heatmap parameters
        width, height = image.size
        sigma = 15.0
        th = 1.6052
        delta = math.sqrt(th * 2)

        for i, point in enumerate(keypoints):
            # skip if point is invalid or invisible
            if point[0] < 0 or point[1] < 0 or point[2] == 0:
                continue

            # Center of heatmap from annotated keypoint
            center_x, center_y = point[0], point[1]

            x_min = int(max(0, center_x - delta * sigma))
            y_min = int(max(0, center_y - delta * sigma))
            x_max = int(min(width, center_x + delta * sigma))
            y_max = int(min(height, center_y + delta * sigma))

            # skip if bounding box is invalid
            if x_min >= x_max or y_min >= y_max:
                continue

            # Create Gaussian heatmap
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            d2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
            exponent = np.exp(-d2 / (2 * sigma ** 2))
            heatmap[y_min:y_max, x_min:x_max, i] = np.maximum(heatmap[y_min:y_max, x_min:x_max, i], exponent)

        # Resize heatmap to target size
        heatmap = cv2.resize(heatmap, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)

        return heatmap
    
    def get_z_coords(self, keypoints, keypoints_3d):
        """
        Get z coordinates from 3D keypoints
        z_scale calculated based on variance of visible keypoints - could be improved
        """
        scale = 1.0
        vis_mask = np.array([1 if point[2] != 0 else 0 for point in keypoints])
        x_coords = keypoints[:, 0] * vis_mask
        y_coords = keypoints[:, 1] * vis_mask
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        scale = (x_std + y_std) / 2

        z_coords = keypoints_3d - np.sum(keypoints_3d * com_weights)
        scaled_z = z_coords / scale

        return scaled_z
    
    def apply_augmentations(self, image, keypoints, keypoints_3d):
        """Apply data augmentations based on config"""
        
        if self.config_preproc["pre_crop"]:
            image, keypoints = pre_crop_image(image, keypoints)
        
        if self.config_preproc["is_scale"]:
            image, keypoints, keypoints_3d = pose_random_scale(image, keypoints, keypoints_3d, self.config_preproc)
        
        if self.config_preproc["is_rotate"]:
            image, keypoints = pose_rotation(image, keypoints, self.config_preproc)
        
        if self.config_preproc["is_flipping"]:
            image, keypoints, keypoints_3d = pose_flip(image, keypoints, keypoints_3d)

        if self.config_preproc["is_resize_shortest_edge"]:
            image, keypoints, keypoints_3d = pose_resize_shortestedge(image, keypoints, keypoints_3d, self.image_processor.crop_size['width'], self.image_processor)
        
        if self.config_preproc["is_crop"]:
            image, keypoints = pose_crop(image, keypoints, 0, 0, self.image_processor.crop_size['width'], self.image_processor.crop_size['height'])
        
        if self.config_preproc["is_occultation"]:
            image = random_occultation(image)

        return image, keypoints, keypoints_3d
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image info
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.images_dir_path, img_info["file_name"])
        
        # Load keypoint annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter out annotations with no keypoints
        anns = [ann for ann in anns if ann.get("num_keypoints", 0) > 0]
        
        # Sanity check - if no annotations with keypoints, raise error
        if not anns:
            raise ValueError(f"No annotations found for image {img_path}")
        
        # Use first person's keypoints
        ann = anns[0]
        
        # Extract keypoints [x, y, visibility] and 3d keypoints [z]
        kps = np.array(ann["keypoints"]).reshape(-1, 3)
        if "keypoints_z" in ann:
            keypoints_z = np.array(ann["keypoints_z"])
        else:
            raise ValueError(f"No 3d keypoints found for image {img_path}")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply augmentations
        if self.transform and self.config_preproc:
            img, kps, keypoints_z = self.apply_augmentations(img, kps.copy(), keypoints_z.copy())

        # Convert keypoints to heatmaps, and 3d keypoints to z_coords
        heatmap = self.keypoints_to_heatmaps(img, kps, (self.config_model['output_heatmap_size'], self.config_model['output_heatmap_size']))
        z_coords = self.get_z_coords(kps, keypoints_z)

        # Preprocess image for model
        image = self.preprocess_image(img)

        return {
            'image': image,
            '2d_heatmaps': torch.tensor(heatmap, dtype=torch.float32),
            '2d_keypoints': torch.tensor(kps, dtype=torch.float32),
            'z_coords': torch.tensor(z_coords, dtype=torch.float32)
        }


def create_dataloaders(config_preproc, config_model,images_dir_path, annotation_json_path,
                       batch_size=8, num_workers=4):
    """
    Create data loaders for training and validation
    
    Args:
        config_training: Training configuration
        config_model: Model configuration
        config_preproc: Preprocessing configuration
        images_dir_path: Path to images directory
        annotation_json_path: Path to annotation JSON
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        dataloader: DataLoader for training
    """
    # Create image processor
    image_processor = AutoImageProcessor.from_pretrained(config_model["model_name"])
    
    # Create dataset
    dataset = PoseDataset(
        images_dir_path=images_dir_path,
        annotation_json_path=annotation_json_path,
        image_processor=image_processor,
        num_model_keypoints=config_model["num_keypoints"],
        config_preproc=config_preproc,
        config_model=config_model,
        transform=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 
