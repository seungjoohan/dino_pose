import torch
import torch.nn as nn
import timm
from transformers import AutoImageProcessor
from .lora import LoRAAttention
from .base_pose import BasePoseModel
from typing import Dict, Any

class FastVitPoseModel(BasePoseModel):
    def __init__(self, num_keypoints=24, backbone="fastvit_t8.apple_in1k", heatmap_size=48):
        super(FastVitPoseModel, self).__init__()
        # Remove 'timm/' prefix if present
        if backbone.startswith('timm/'):
            backbone = backbone[5:]
        
        self.backbone_name = f"timm/{backbone}"  # Keep original name for compatibility
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        
        # Load FastViT backbone directly from timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)  # num_classes=0 for feature extraction
        
        # Get model configuration for image preprocessing
        self.backbone_config = self.backbone.default_cfg
        self.input_size = self.backbone_config.get('input_size', (3, 224, 224))
        self.image_size = self.input_size[1]  # Assuming square images
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get backbone output features dimension
        self.feat_dim = self._get_feature_dim()
        
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
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 256, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # [B, 64, 48, 48]
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
    
    @classmethod
    def from_config(cls, model_name: str, config: Dict[str, Any]):
        """Factory method to create model from configuration"""
        return cls(
            num_keypoints=config['num_keypoints'],
            backbone=model_name,
            heatmap_size=config['output_heatmap_size']
        )
    
    def _get_feature_dim(self):
        """Get the feature dimension from backbone output"""
        with torch.no_grad():
            dummy_input = torch.randn(1, *self.input_size)
            try:
                features = self.backbone(dummy_input)
                return features.shape[-1]
            except Exception as e:
                print(f"Warning: Could not auto-detect feature dimension: {e}")
                print("Using default feature dimension of 768")
                return 768
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Input images (B, C, H, W)
            
        Returns:
            heatmaps: Predicted 2D heatmaps (B, num_keypoints, height, width)
            z_coords: Predicted z-coordinates (B, num_keypoints)
        """
        # Extract features using the timm backbone (returns [B, feat_dim])
        features = self.backbone(pixel_values)
        
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
        """Count the number of parameters in the model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
            
    def print_trainable_parameters(self):
        """Print the names of trainable parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")

class FastVitPoseModelLoRA(BasePoseModel):
    """
    TODO: FASTViT LoRA - check if LoRAAttention is compatible with timm model
    """
    def __init__(self, num_keypoints=24, backbone="fastvit_t8.apple_in1k", heatmap_size=48,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super(FastVitPoseModelLoRA, self).__init__()
        # Remove 'timm/' prefix if present
        if backbone.startswith('timm/'):
            backbone = backbone[5:]
        
        self.backbone_name = f"timm/{backbone}"  # Keep original name for compatibility
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        
        # Load FastViT backbone directly from timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)  # num_classes=0 for feature extraction
        
        # Get model configuration for image preprocessing
        self.backbone_config = self.backbone.default_cfg
        self.input_size = self.backbone_config.get('input_size', (3, 224, 224))
        self.image_size = self.input_size[1]  # Assuming square images
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # TODO: Implement FastViT-specific LoRA for RepMixer blocks
        # FastViT uses RepMixer blocks instead of traditional attention
        # Current LoRA implementation is designed for attention layers
        
        # Get backbone output features dimension
        self.feat_dim = self._get_feature_dim()
        
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
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 256, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),    # [B, 64, 48, 48]
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
    
    @classmethod
    def from_config(cls, model_name: str, config: Dict[str, Any]):
        """Factory method to create model from configuration"""
        return cls(
            num_keypoints=config['num_keypoints'],
            backbone=model_name,
            heatmap_size=config['output_heatmap_size'],
            lora_rank=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.1)
        )
    
    def _get_feature_dim(self):
        """Get the feature dimension from backbone output"""
        with torch.no_grad():
            dummy_input = torch.randn(1, *self.input_size)
            try:
                features = self.backbone(dummy_input)
                return features.shape[-1]
            except Exception as e:
                print(f"Warning: Could not auto-detect feature dimension: {e}")
                print("Using default feature dimension of 768")
                return 768
    
    def forward(self, pixel_values):
        # Extract features using the timm backbone (returns [B, feat_dim])
        features = self.backbone(pixel_values)
        
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
        """Count the number of parameters in the model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
            
    def print_trainable_parameters(self):
        """Print the names of trainable parameters"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")

if __name__ == "__main__":
    print("Testing FastViT models...")
    
    # Test standard model
    print("=== Testing FastVitPoseModel ===")
    model = FastVitPoseModel()
    print(f"Standard model created successfully")
    print(f"Trainable parameters: {model.count_parameters():,}")
    
    # Test LoRA model (will show warning)
    print("\n=== Testing FastVitPoseModelLoRA ===")
    model_lora = FastVitPoseModelLoRA()
    print(f"LoRA model created successfully")
    print(f"Trainable parameters: {model_lora.count_parameters():,}")