import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.model_utils import create_model_from_config, save_model_checkpoint, load_model_smart
from data_loader.data_loader import create_dataloaders
from config.config import get_default_configs
from src.model_utils import compute_pckh_dataset

class DynamicLossWeighting:
    def __init__(self, initial_weight=0.1, target_ratio=1.0, adjustment_rate=0.1):
        self.weight = initial_weight
        self.target_ratio = target_ratio
        self.adjustment_rate = adjustment_rate
        self.best_weight = initial_weight
        self.best_val_loss = float('inf')
        
    def update(self, kp_loss, z_loss, is_validation=False):
        if is_validation:
            return self.weight
            
        # make weight * z_loss ≈ kp_loss
        target_weight = (kp_loss + 1e-8) / (z_loss + 1e-8)
        
        # Exponential moving average for smooth update
        self.weight = (1 - self.adjustment_rate) * self.weight + self.adjustment_rate * target_weight
        
        # weight boundary
        min_weight = 1e-3
        max_weight = 10.0
        self.weight = max(min_weight, min(max_weight, self.weight))
        
        return self.weight
    
    def update_best_weight(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weight = self.weight

def keypoint_loss(pred_heatmaps, target_heatmaps, confidence_mask):
    """
    Compute MSE loss only on visible keypoints
    """
    # adjust heatmaps to visible keypoints
    confidence_mask = (confidence_mask > 1).float()
    expanded_mask = confidence_mask.unsqueeze(2).unsqueeze(2).expand_as(pred_heatmaps)

    diff = (pred_heatmaps - target_heatmaps) ** 2
    weight = torch.exp(-diff.detach())
    weighted_diff = weight * diff

    masked_diff = weighted_diff * expanded_mask
    return masked_diff.mean()
    # masked_pred = pred_heatmaps * expanded_mask
    # masked_target = target_heatmaps * expanded_mask
    
    # mse = torch.nn.MSELoss()
    # return mse(masked_pred, masked_target)

def z_loss(pred_z, target_z, confidence_mask):
    """
    Compute MSE loss for z only on visible keypoints
    """
    # adjust z to visible keypoints
    confidence_mask = (confidence_mask > 1).float()
    z_pred = pred_z * confidence_mask
    z_target = target_z * confidence_mask
    
    # mse = torch.nn.MSELoss()
    # return mse(z_pred, z_target)
    return torch.abs(z_pred - z_target).mean() # trying L1 loss

def train_one_epoch(model, dataloader, device, optimizer, loss_weighting, epoch, is_validation=False):
    start_time = time.time()
    model.train() if not is_validation else model.eval()
    running_loss = 0.0
    running_keypoint_loss = 0.0
    running_z_coords_loss = 0.0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                      desc=f"{'Validation' if is_validation else f'Epoch {epoch+1} Training'}", 
                      leave=False)
    
    with torch.no_grad() if is_validation else torch.enable_grad():
        for i, batch in progress_bar:
            # Move data to device
            pixel_values = batch['image'].to(device)
            heatmaps = batch['2d_heatmaps'].to(device)
            kps = batch['2d_keypoints'].to(device)
            z_coords = batch['z_coords'].to(device)
            
            if not is_validation:
                optimizer.zero_grad()
            
            pred_heatmaps, pred_z_coords = model(pixel_values)
            
            # Get the confidence mask
            confidence_mask = kps[..., 2]
            
            # Compute losses
            kp_loss = keypoint_loss(pred_heatmaps, heatmaps, confidence_mask)
            z_coords_loss = z_loss(pred_z_coords, z_coords, confidence_mask)
            
            # Get current 2d & 3d lossweight
            current_weight = loss_weighting.update(
                kp_loss.item(), 
                z_coords_loss.item(),
                is_validation=is_validation
            )
            
            # Compute weighted loss
            loss = kp_loss + current_weight * z_coords_loss
            
            if not is_validation:
                loss.backward()
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            running_keypoint_loss += kp_loss.item()
            running_z_coords_loss += z_coords_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{running_loss / (i + 1):.6f}",
                'kp_loss': f"{running_keypoint_loss / (i + 1):.6f}",
                'z_coords_loss': f"{running_z_coords_loss / (i + 1):.6f}",
                'weight': f"{current_weight:.4f}"
            })
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_keypoint_loss = running_keypoint_loss / len(dataloader)
    avg_z_coords_loss = running_z_coords_loss / len(dataloader)

    end_time = time.time()

    if is_validation:
        print(f"Validation - Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, 3D Loss: {avg_z_coords_loss:.4f}")
    else:
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, 3D Loss: {avg_z_coords_loss:.4f}, Elapsed Time: {end_time - start_time:.2f}s")
    
    return avg_loss, avg_keypoint_loss, avg_z_coords_loss

def main(args):
    # Get configurations
    config_dataset, config_training, config_preproc, config_model = get_default_configs()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config_training['checkpoint_dir'], exist_ok=True)
    
    # Create dataloader
    print(f"Creating dataloader for {config_dataset['train_images_dir']}...")
    train_dataloader = create_dataloaders(
        config_preproc=config_preproc,
        config_model=config_model,
        images_dir_path=config_dataset['train_images_dir'],
        annotation_json_path=config_dataset['train_annotation_json'],
        batch_size=config_training['batch_size'],
        num_workers=config_training['multiprocessing_num']
    )
    
    if config_dataset['val_images_dir'] and config_dataset['val_annotation_json']:
        print(f"Creating validation dataloader for {config_dataset['val_images_dir']}...")
        val_dataloader = create_dataloaders(
            config_preproc=config_preproc,
            config_model=config_model,
            images_dir_path=config_dataset['val_images_dir'],
            annotation_json_path=config_dataset['val_annotation_json'],
            batch_size=config_training['batch_size'],
            num_workers=config_training['multiprocessing_num']
        )
    else:
        val_dataloader = None
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model {config_model['model_name']}...")
    
    # Check if we need to load from checkpoint or create new model
    if config_model['load_model'].endswith('.pth'):
        print(f"Loading model from {config_model['load_model']}")
        model = load_model_smart(config_model['load_model'], eval_mode=False)
    else:
        model = create_model_from_config(config_model)
    model.to(device)
    
    # Compile the model
    try:
        # compile model in optimized mode except for MPS device
        if device.type == 'mps':
            compiled_model = model
            print("Using uncompiled model for MPS device")
        else:
            compiled_model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=True,
                dynamic=True
            )
            print("Model compiled successfully with max-autotune mode")
    except Exception as e:
        print(f"Failed to compile model: {e}")
        compiled_model = model
        print("Using uncompiled model")
    
    # Print model parameters    
    print(f"Trainable parameters: {model.count_parameters():,}")
    # model.print_trainable_parameters()
    
    # Create optimizer
    optimizer = optim.AdamW(
        compiled_model.parameters(),
        lr=config_training['learning_rate'],
        weight_decay=config_training['weight_decay']
    )
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Initialize loss weighting
    loss_weighting = DynamicLossWeighting(
        initial_weight=0.1,
        target_ratio=1.0,
        adjustment_rate=0.1
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_pckh_2d, best_pckh_3d = compute_pckh_dataset(model, config_dataset['val_images_dir'], config_dataset['val_annotation_json'], config_model['model_name'], device)
    print(f"Starting training with PCKh (2D): {best_pckh_2d:.4f}, PCKh (3D): {best_pckh_3d:.4f}")
    
    for epoch in range(config_training['num_epochs']):
        # Train
        train_loss, train_kp_loss, train_z_loss = train_one_epoch(
            model=compiled_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_weighting=loss_weighting,
            is_validation=False
        )
        train_losses.append(train_loss)
        
        # Validation
        if val_dataloader:
            val_loss, val_kp_loss, val_z_loss = train_one_epoch(
                model=compiled_model,
                dataloader=val_dataloader,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                loss_weighting=loss_weighting,
                is_validation=True
            )
            val_losses.append(val_loss)
            
            # Update scheduler
            scheduler.step(val_loss)
            # Update best weight based on validation loss
            loss_weighting.update_best_weight(val_loss)
            
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(config_training['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_model_checkpoint(
                model=compiled_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=val_loss,
                loss_weight=loss_weighting.best_weight,
                config_model=config_model,
                config_training=config_training,
                config_preproc=config_preproc,
                save_path=checkpoint_path
            )
        
        # Save best model
        if (epoch + 1) % config_training['save_freq'] == 0:
            # save model only when pckh improved
            pckh_2d, pckh_3d = compute_pckh_dataset(model, config_dataset['val_images_dir'], config_dataset['val_annotation_json'], config_model['model_name'], device)
            print(f"Epoch {epoch+1} - PCKh (2D): {pckh_2d:.4f}, PCKh (3D): {pckh_3d:.4f}")

            # save model only when either 2d or 3d pckh improved
            if np.mean(pckh_2d) > best_pckh_2d or np.mean(pckh_3d) > best_pckh_3d:
                checkpoint_path = os.path.join(config_training['checkpoint_dir'], f'best_model_{epoch+1}.pth')
                save_model_checkpoint(
                    model=compiled_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_loss=val_loss,
                    loss_weight=loss_weighting.best_weight,
                    config_model=config_model,
                    config_training=config_training,
                    config_preproc=config_preproc,
                    save_path=checkpoint_path
                )
            
            # update best pckh scores
            if pckh_2d > best_pckh_2d:
                best_pckh_2d = pckh_2d
            if pckh_3d > best_pckh_3d:
                best_pckh_3d = pckh_3d
    
    # Save final model
    checkpoint_path = os.path.join(config_training['checkpoint_dir'], 'final_model.pth')
    save_model_checkpoint(
        model=compiled_model,
        optimizer=optimizer,
        epoch=config_training['num_epochs'],
        train_loss=train_loss,
        valid_loss=val_loss,
        loss_weight=loss_weighting.best_weight,
        config_model=config_model,
        config_training=config_training,
        config_preproc=config_preproc,
        save_path=checkpoint_path
    )
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_dataloader:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(config_training['checkpoint_dir'], 'loss_plot.png'))
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv2 pose model")
    parser.add_argument("--config_file", type=str, default="config/config.py", help="model training config file")
    
    args = parser.parse_args()
    main(args) 