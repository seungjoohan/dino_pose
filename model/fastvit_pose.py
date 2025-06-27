import torch
import torch.nn as nn
import timm
from .lora import FastViTLoRA
from .base_pose import BasePoseModel
from .pose_heads import SpatialAwarePoseHeads
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
        
        # Load FastViT backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        
        self.backbone.head = SpatialAwarePoseHeads(
            feat_channels=768,  # FastViT final_conv output
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            spatial_input_size=14,
            z_coord_config={
                'hidden_dims': (1024, 512, 256),
                'dropout_rate': 0.1
            }
        )
        # Get model configuration for image preprocessing
        self.backbone_config = self.backbone.default_cfg
        self.input_size = self.backbone_config.get('input_size', (3, 224, 224))
        self.image_size = self.input_size[1]  # Assuming square images
        
        # Freeze backbone except head
        for name, param in self.backbone.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
    
    def forward(self, pixel_values):
        return self.backbone(pixel_values)
   
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
    
    
    def _extract_both_features(self, pixel_values):
        """Extract both spatial feature map and global features from FastViT backbone"""
        x = pixel_values
        
        # Forward through stem
        x = self.backbone.stem(x)
        
        # Forward through all stages
        for stage in self.backbone.stages:
            x = stage(x)
        
        # Apply final conv (384 -> 768 channels)
        feature_map = self.backbone.final_conv(x)  # [B, 768, H', W']
        
        # Global average pooling for Z head
        global_features = feature_map.mean(dim=(2, 3))  # [B, 768]
        
        return feature_map, global_features

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
    def __init__(self, num_keypoints=24, backbone="fastvit_t8.apple_in1k", heatmap_size=48,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super(FastVitPoseModelLoRA, self).__init__()
        # Remove 'timm/' prefix if present
        if backbone.startswith('timm/'):
            backbone = backbone[5:]
        
        self.backbone_name = f"timm/{backbone}"  # Keep original name for compatibility
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        
        # Store LoRA configuration for loading fixes
        self.lora_config = {
            'rank': lora_rank,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        
        # Load FastViT backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to MLP layers (fc1, fc2 in ConvMlp)
        self.lora_modules = FastViTLoRA.apply_lora_to_model(
            self.backbone, 
            target_layers=['mlp.fc1', 'mlp.fc2'],
            r=lora_rank, 
            alpha=lora_alpha, 
            dropout=lora_dropout
        )
        
        self.backbone.head = SpatialAwarePoseHeads(
            feat_channels=768,  # FastViT final_conv output
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            spatial_input_size=14,
            z_coord_config={
                'hidden_dims': (1024, 512, 256),
                'dropout_rate': 0.1
            }
        )
        # Get model configuration for image preprocessing
        self.backbone_config = self.backbone.default_cfg
        self.input_size = self.backbone_config.get('input_size', (3, 224, 224))
        self.image_size = self.input_size[1]  # Assuming square images
    
    def apply_loading_fixes(self):
        """fix loading issues for LoRA model"""
        print("Applying FastVit LoRA loading fixes...")
        
        # 1. Update LoRA alpha and rank values to match config
        for name, module in self.named_modules():
            if hasattr(module, 'alpha') and hasattr(module, 'rank'):
                if module.alpha != self.lora_config['alpha'] or module.rank != self.lora_config['rank']:
                    module.alpha = self.lora_config['alpha']
                    module.rank = self.lora_config['rank']
        
        # 2. fix dropout rate
        for name, module in self.named_modules():
            if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
                if abs(module.dropout.p - self.lora_config['dropout']) > 1e-6:
                    module.dropout.p = self.lora_config['dropout']
        
        # 3. set evaluation mode
        self.eval()
        for module in self.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
                module.eval()
        
        print("FastVit LoRA loading fixes applied successfully!")
    
    def forward(self, pixel_values):
        return self.backbone(pixel_values)
    
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