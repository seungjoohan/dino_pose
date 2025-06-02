import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model
from model.lora import LoRAAttention

class Dinov2PoseModel(nn.Module):
    def __init__(self, num_keypoints=24, backbone="facebook/dinov2-base", unfreeze_last_n_layers=0, heatmap_size=48):
        super(Dinov2PoseModel, self).__init__()
        self.dinov2backbone = Dinov2Model.from_pretrained(backbone)
        self.image_processor = AutoImageProcessor.from_pretrained(backbone)
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # Freeze backbone
        for param in self.dinov2backbone.parameters():
            param.requires_grad = False
            
        # Optionally unfreeze some layers for fine-tuning
        if unfreeze_last_n_layers > 0:
            # Unfreeze the last n transformer layers
            for i in range(1, unfreeze_last_n_layers + 1):
                layer_idx = len(self.dinov2backbone.encoder.layer) - i
                for param in self.dinov2backbone.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            
            # Also unfreeze layer normalization layers in the last n layers
            for i in range(1, unfreeze_last_n_layers + 1):
                layer_idx = len(self.dinov2backbone.encoder.layer) - i
                # Unfreeze both layer norms in each transformer block
                for param in self.dinov2backbone.encoder.layer[layer_idx].norm1.parameters():
                    param.requires_grad = True
                for param in self.dinov2backbone.encoder.layer[layer_idx].norm2.parameters():
                    param.requires_grad = True

        # Get backbone output features dimension
        self.feat_dim = self.dinov2backbone.config.hidden_size
        
        # Project features to a 3D volume that can be upsampled to heatmaps
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6*6*256),  # Project to 6x6 spatial features with 256 channels
            nn.ReLU()
        )
        
        # Heatmap generation using transposed convolutions
        self.heatmap_head = nn.Sequential(
            # Input: [B, 256, 6, 6]
            
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 64, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # [B, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, num_keypoints, kernel_size=1)  # [B, num_keypoints, 48, 48]
        )
        
        # Z-coordinate head
        self.z_head = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints)
        )
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Input images (B, C, W, H)
            
        Returns:
            heatmaps: Predicted 2D heatmaps (B, num_keypoints, width, height)
            z_coords: Predicted z-coordinates (B, num_keypoints)
        """
        # Extract features using the backbone
        outputs = self.dinov2backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        
        # Project features to 3D volume for heatmap generation
        projected_features = self.feature_projection(features)  # [B, 6*6*256]
        batch_size = projected_features.size(0)
        
        # Reshape to [B, 256, 6, 6]
        reshaped_features = projected_features.view(batch_size, 256, 6, 6)
        
        # Generate heatmaps using transposed convolutions
        heatmaps = self.heatmap_head(reshaped_features)  # [B, num_keypoints, 48, 48]
        
        # Apply Z-coordinate head
        z_coords = self.z_head(features)  # [B, num_keypoints]
        
        return heatmaps, z_coords
        
    def count_parameters(self, trainable_only=True):
        """
        Count the number of parameters in the model
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
            
    def print_trainable_parameters(self):
        """
        Print the names of trainable parameters
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")
        
class Dinov2PoseModelLoRA(nn.Module):
    def __init__(self, num_keypoints=24, backbone="facebook/dinov2-base", heatmap_size=48,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super(Dinov2PoseModelLoRA, self).__init__()
        self.dinov2backbone = Dinov2Model.from_pretrained(backbone)
        self.image_processor = AutoImageProcessor.from_pretrained(backbone)
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # Apply LoRA to attention layers
        for i, layer in enumerate(self.dinov2backbone.encoder.layer):
            if i >= len(self.dinov2backbone.encoder.layer) - 1:
                layer.attention = LoRAAttention(
                    layer.attention, 
                    r=lora_rank, 
                    alpha=lora_alpha, 
                    dropout=lora_dropout
                )

        # Get backbone output features dimension
        self.feat_dim = self.dinov2backbone.config.hidden_size
        
        # Project features to a 3D volume that can be upsampled to heatmaps
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6*6*256),  # Project to 6x6 spatial features with 256 channels
            nn.ReLU()
        )
        
        # Heatmap generation using transposed convolutions
        self.heatmap_head = nn.Sequential(
            # Input: [B, 256, 6, 6]
            
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 64, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # [B, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, num_keypoints, kernel_size=1)  # [B, num_keypoints, 48, 48]
        )
        
        # Z-coordinate head
        self.z_head = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints)
        )
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Input images (B, C, W, H)
            
        Returns:
            heatmaps: Predicted 2D heatmaps (B, num_keypoints, width, height)
            z_coords: Predicted z-coordinates (B, num_keypoints)
        """
        # Extract features using the backbone
        outputs = self.dinov2backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        
        # Project features to 3D volume for heatmap generation
        projected_features = self.feature_projection(features)  # [B, 6*6*256]
        batch_size = projected_features.size(0)
        
        # Reshape to [B, 256, 6, 6]
        reshaped_features = projected_features.view(batch_size, 256, 6, 6)
        
        # Generate heatmaps using transposed convolutions
        heatmaps = self.heatmap_head(reshaped_features)  # [B, num_keypoints, 48, 48]
        
        # Apply Z-coordinate head
        z_coords = self.z_head(features)  # [B, num_keypoints]
        
        return heatmaps, z_coords
        
    def count_parameters(self, trainable_only=True):
        """
        Count the number of parameters in the model
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
            
    def print_trainable_parameters(self):
        """
        Print the names of trainable parameters
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")
        
    
def __main__():
    model = Dinov2PoseModelLoRA()
    model.print_trainable_parameters()
    
if __name__ == "__main__":
    __main__()