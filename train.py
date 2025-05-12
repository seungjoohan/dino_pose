import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.dinov2_pose import Dinov2PoseModel
from data_loader.data_loader import create_dataloaders
from config.config import get_default_configs
from src.model_utils import compute_pckh_dataset


def keypoint_loss(pred_heatmaps, target_heatmaps, confidence_mask):
    """
    Compute MSE loss only on visible keypoints
    """
    # adjust heatmaps to visible keypoints
    confidence_mask = (confidence_mask > 1).float()
    expanded_mask = confidence_mask.unsqueeze(2).unsqueeze(2).expand_as(pred_heatmaps)
    masked_pred = pred_heatmaps * expanded_mask
    masked_target = target_heatmaps * expanded_mask
    
    mse = torch.nn.MSELoss()
    return mse(masked_pred, masked_target)

def z_loss(pred_z, target_z, confidence_mask):
    """
    Compute MSE loss for z only on visible keypoints
    """
    # adjust z to visible keypoints
    confidence_mask = (confidence_mask > 1).float()
    z_pred = pred_z * confidence_mask
    z_target = target_z * confidence_mask
    
    mse = torch.nn.MSELoss()
    return mse(z_pred, z_target)

def train_one_epoch(model, dataloader, device, optimizer, epoch, print_freq=10):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        device: Device to train on
        optimizer: Optimizer
        epoch: Current epoch
        print_freq: Frequency of printing training stats
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    running_keypoint_loss = 0.0
    running_z_coords_loss = 0.0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                      desc=f"Epoch {epoch+1} Training", leave=False)
    
    for i, batch in progress_bar:
        # Move data to device
        pixel_values = batch['image'].to(device)
        heatmaps = batch['2d_heatmaps'].to(device)
        kps = batch['2d_keypoints'].to(device)
        z_coords = batch['z_coords'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred_heatmaps, pred_z_coords = model(pixel_values)
        
        # Get the confidence mask
        confidence_mask = kps[..., 2]
        
        # Compute losses
        kp_loss = keypoint_loss(pred_heatmaps, heatmaps, confidence_mask)
        z_coords_loss = z_loss(pred_z_coords, z_coords, confidence_mask)
        
        # Weighted combined loss
        loss = kp_loss + 0.1 * z_coords_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        running_keypoint_loss += kp_loss.item()
        running_z_coords_loss += z_coords_loss.item() * 0.1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{running_loss / (i + 1):.6f}",
            'kp_loss': f"{running_keypoint_loss / (i + 1):.6f}",
            'z_coords_loss': f"{running_z_coords_loss / (i + 1):.6f}"
        })
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_keypoint_loss = running_keypoint_loss / len(dataloader)
    avg_z_coords_loss = running_z_coords_loss / len(dataloader)
    
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, 3D Loss: {avg_z_coords_loss:.4f}")
    
    return avg_loss, avg_keypoint_loss, avg_z_coords_loss

def validate(model, dataloader, device):
    """
    Validate the model
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        device: Device to validate on
    
    Returns:
        avg_loss: Average loss for the validation set
    """
    model.eval()
    running_loss = 0.0
    running_keypoint_loss = 0.0
    running_z_coords_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                          desc=f"Validation", leave=False)
        
        for i, batch in progress_bar:
            # Move data to device
            pixel_values = batch['image'].to(device)
            heatmaps = batch['2d_heatmaps'].to(device)
            kps = batch['2d_keypoints'].to(device)
            z_coords = batch['z_coords'].to(device)
            
            # Forward pass
            pred_heatmaps, pred_z_coords = model(pixel_values)
            
            # Get the confidence mask
            confidence_mask = kps[..., 2]
            
            # Compute losses
            kp_loss = keypoint_loss(pred_heatmaps, heatmaps, confidence_mask)
            z_coords_loss = z_loss(pred_z_coords, z_coords, confidence_mask)
            
            # Weighted combined loss
            loss = kp_loss + 0.1 * z_coords_loss
            
            # Update statistics
            running_loss += loss.item()
            running_keypoint_loss += kp_loss.item()
            running_z_coords_loss += z_coords_loss.item() * 0.1
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': f"{running_loss / (i + 1):.6f}",
                'val_kp_loss': f"{running_keypoint_loss / (i + 1):.6f}",
                'val_z_coords_loss': f"{running_z_coords_loss / (i + 1):.6f}"
            })
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_keypoint_loss = running_keypoint_loss / len(dataloader)
    avg_z_coords_loss = running_z_coords_loss / len(dataloader)
    
    print(f"Validation - Loss: {avg_loss:.4f}, Keypoint Loss: {avg_keypoint_loss:.4f}, 3D Loss: {avg_z_coords_loss:.4f}")
    
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
    model = Dinov2PoseModel(
        num_keypoints=config_model['num_keypoints'],
        backbone=config_model['model_name'],
        unfreeze_last_n_layers=config_model['unfreeze_last_n_layers'],
        heatmap_size=config_model['output_heatmap_size']
    )

    # load model from checkpoint if specified
    if config_model['load_model'].endswith('.pth'):
        # load model from checkpoint
        model_dict = torch.load(config_model['load_model'])
        model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)
    
    # Print model parameters    
    print(f"Trainable parameters: {model.count_parameters():,}")
    model.print_trainable_parameters()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
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
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_pckh_2d, best_pckh_3d = compute_pckh_dataset(model, config_dataset['val_images_dir'], config_dataset['val_annotation_json'], config_model['model_name'], device)
    print(f"Starting training with PCKh (2D): {best_pckh_2d:.4f}, PCKh (3D): {best_pckh_3d:.4f}")
    
    for epoch in range(config_training['num_epochs']):
        # Train
        train_loss, train_kp_loss, train_z_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            print_freq=config_training['print_freq']
        )
        train_losses.append(train_loss)
        
        # Validate
        if val_dataloader:
            val_loss, val_kp_loss, val_z_loss = validate(
                model=model,
                dataloader=val_dataloader,
                device=device
            )
            val_losses.append(val_loss)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save checkpoint 
        if (epoch + 1) % 20 == 0:
            # save model
            checkpoint_path = os.path.join(config_training['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config_model': config_model,
                'config_training': config_training,
                'config_preproc': config_preproc
                }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if (epoch + 1) % config_training['save_freq'] == 0:
            # save model only when pckh improved
            pckh_2d, pckh_3d = compute_pckh_dataset(model, config_dataset['val_images_dir'], config_dataset['val_annotation_json'], config_model['model_name'], device)
            print(f"Epoch {epoch+1} - PCKh (2D): {pckh_2d:.4f}, PCKh (3D): {pckh_3d:.4f}")

            # save model only when either 2d or 3d pckh improved
            if np.mean(pckh_2d) > best_pckh_2d or np.mean(pckh_3d) > best_pckh_3d:
                checkpoint_path = os.path.join(config_training['checkpoint_dir'], f'best_model_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'config_model': config_model,
                    'config_training': config_training,
                    'config_preproc': config_preproc
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
            # update best pckh scores
            if pckh_2d > best_pckh_2d:
                best_pckh_2d = pckh_2d
            if pckh_3d > best_pckh_3d:
                best_pckh_3d = pckh_3d
    
    # Save final model
    checkpoint_path = os.path.join(config_training['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'epoch': config_training['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'config_model': config_model,
        'config_training': config_training,
        'config_preproc': config_preproc
    }, checkpoint_path)
    print(f"Saved final model to {checkpoint_path}")
    
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