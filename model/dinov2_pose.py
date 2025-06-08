import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model
from .lora import LoRAAttention
from .base_pose import BasePoseModel
from .pose_heads import PoseHeads
from typing import Dict, Any

class Dinov2PoseModel(BasePoseModel):
    def __init__(self, num_keypoints=24, backbone="facebook/dinov2-base", unfreeze_last_n_layers=0, heatmap_size=48):
        super(Dinov2PoseModel, self).__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone)
        self.backbone_name = backbone
        self.image_processor = AutoImageProcessor.from_pretrained(backbone)
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Optionally unfreeze some layers for fine-tuning
        if unfreeze_last_n_layers > 0:
            # Unfreeze the last n transformer layers
            for i in range(1, unfreeze_last_n_layers + 1):
                layer_idx = len(self.backbone.encoder.layer) - i
                for param in self.backbone.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
            
            # Also unfreeze layer normalization layers in the last n layers
            for i in range(1, unfreeze_last_n_layers + 1):
                layer_idx = len(self.backbone.encoder.layer) - i
                # Unfreeze both layer norms in each transformer block
                for param in self.backbone.encoder.layer[layer_idx].norm1.parameters():
                    param.requires_grad = True
                for param in self.backbone.encoder.layer[layer_idx].norm2.parameters():
                    param.requires_grad = True

        # Get backbone output features dimension
        self.feat_dim = self.backbone.config.hidden_size
        
        # Initialize pose heads
        self.pose_heads = PoseHeads(
            feat_dim=self.feat_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size
        )
    
    @classmethod
    def from_config(cls, model_name: str, config: Dict[str, Any]):
        """Factory method to create model from configuration"""
        return cls(
            num_keypoints=config['num_keypoints'],
            backbone=model_name,
            unfreeze_last_n_layers=config.get('unfreeze_last_n_layers', 0),
            heatmap_size=config['output_heatmap_size']
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
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        
        # pose heads
        heatmaps, z_coords = self.pose_heads(features)
        
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
        
class Dinov2PoseModelLoRA(BasePoseModel):
    def __init__(self, num_keypoints=24, backbone="facebook/dinov2-base", heatmap_size=48,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super(Dinov2PoseModelLoRA, self).__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone)
        self.backbone_name = backbone
        self.image_processor = AutoImageProcessor.from_pretrained(backbone)
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # Freeze backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i >= len(self.backbone.encoder.layer) - 1:
                layer.attention = LoRAAttention(
                    layer.attention, 
                    r=lora_rank, 
                    alpha=lora_alpha, 
                    dropout=lora_dropout
                )

        # Get backbone output features dimension
        self.feat_dim = self.backbone.config.hidden_size
        
        # Initialize pose heads
        self.pose_heads = PoseHeads(
            feat_dim=self.feat_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size
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
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Input images (B, C, W, H)
            
        Returns:
            heatmaps: Predicted 2D heatmaps (B, num_keypoints, width, height)
            z_coords: Predicted z-coordinates (B, num_keypoints)
        """
        # Extract features using the backbone
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        
        # pose heads
        heatmaps, z_coords = self.pose_heads(features)
        
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