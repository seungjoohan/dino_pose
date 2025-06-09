import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model
from .lora import LoRAAttention
from .base_pose import BasePoseModel
from .pose_heads import SpatialAwarePoseHeads
from typing import Dict, Any

class Dinov2PoseModel(BasePoseModel):
    def __init__(self, num_keypoints=24, backbone="facebook/dinov2-base", unfreeze_last_n_layers=0, heatmap_size=48):
        super(Dinov2PoseModel, self).__init__()
        self.backbone = Dinov2Model.from_pretrained(backbone)
        self.backbone_name = backbone
        self.image_processor = AutoImageProcessor.from_pretrained(backbone)
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self._coreml_patch_applied = False

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
        self.pose_heads = SpatialAwarePoseHeads(
            feat_channels=self.feat_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            spatial_input_size=16,
            z_coord_config={
                'hidden_dims': (1024, 512, 256),
                'dropout_rate': 0.1
            }
        )
    
    def apply_coreml_compatibility_patch(self):
        """
        Apply Core ML compatibility patch to fix bicubic interpolation issue.
        Changes bicubic -> nearest interpolation in position embeddings.
        Call this method before converting to Core ML.
        """
        if self._coreml_patch_applied:
            print("⚠️  Core ML patch already applied")
            return
        
        print("🔧 Applying Core ML compatibility patch...")
        
        # Store original interpolate function
        original_interpolate = self.backbone.embeddings.interpolate_pos_encoding
        
        def patched_interpolate_pos_encoding(embeddings, width, height):
            """
            Core ML compatible position embedding interpolation
            Uses nearest instead of bicubic interpolation
            """
            num_patches = embeddings.shape[1] - 1
            num_positions = self.backbone.embeddings.position_embeddings.shape[1] - 1
            
            if num_patches == num_positions and height == width:
                return self.backbone.embeddings.position_embeddings
            
            # Separate class token and patch embeddings
            class_pos_embedding = self.backbone.embeddings.position_embeddings[:, 0]
            patch_pos_embedding = self.backbone.embeddings.position_embeddings[:, 1:]
            
            dim = embeddings.shape[-1]
            h0 = height // self.backbone.embeddings.patch_size
            w0 = width // self.backbone.embeddings.patch_size
            
            # Add small offset to avoid floating point errors
            h0, w0 = h0 + 0.1, w0 + 0.1
            
            sqrt_num_positions = int(num_positions**0.5)
            patch_pos_embedding = patch_pos_embedding.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
            patch_pos_embedding = patch_pos_embedding.permute(0, 3, 1, 2)
            
            patch_pos_embedding = F.interpolate(
                patch_pos_embedding,
                size=(int(h0), int(w0)),
                mode="nearest", # bicubic -> nearest interpolation
            )
            
            patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
            
            return torch.cat((class_pos_embedding.unsqueeze(0), patch_pos_embedding), dim=1)
        
        # Apply the patch
        self.backbone.embeddings.interpolate_pos_encoding = patched_interpolate_pos_encoding
        self._coreml_patch_applied = True
        
        print("   ✅ Core ML compatibility patch applied successfully!")
        print("   📝 bicubic interpolation -> nearest interpolation")
        print("   🎯 Model is now ready for Core ML conversion")
    
    def remove_coreml_compatibility_patch(self):
        """ Remove Core ML compatibility patch and restore original behavior. """
        if not self._coreml_patch_applied:
            print("⚠️  No Core ML patch to remove")
            return
        
        print("🔄 Removing Core ML compatibility patch...")
        
        # Recreate the backbone to restore original behavior
        original_backbone = Dinov2Model.from_pretrained(self.backbone_name)
        
        # Copy weights back
        self.backbone.load_state_dict(original_backbone.state_dict())
        self._coreml_patch_applied = False
        
        print("   ✅ Core ML patch removed successfully!")
        print("   📝 Restored original bicubic interpolation")
    
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
        # Extract features using the backbone
        outputs = self.backbone(pixel_values)
            
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, 256, 384]
        B, N, D = patch_tokens.shape
            
        # Reshape to 2D feature map: 256 tokens -> 16x16 spatial
        H = W = int(N ** 0.5)  # 16 = sqrt(256)
        # Use contiguous() to ensure memory layout compatibility
        spatial_features = patch_tokens.contiguous().view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, 384, 16, 16]
        
        heatmaps, z_coords = self.pose_heads(spatial_features)
        
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
        self._coreml_patch_applied = False

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
        self.pose_heads = SpatialAwarePoseHeads(
            feat_channels=self.feat_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            spatial_input_size=16,
            z_coord_config={
                'hidden_dims': (1024, 512, 256),
                'dropout_rate': 0.1
            }
        )
    
    def apply_coreml_compatibility_patch(self):
        """
        Apply Core ML compatibility patch to fix bicubic interpolation issue.
        Changes bicubic -> nearest interpolation in position embeddings.
        Call this method before converting to Core ML.
        """
        if self._coreml_patch_applied:
            print("⚠️  Core ML patch already applied")
            return
        
        print("🔧 Applying Core ML compatibility patch (LoRA model)...")
        
        # Store original interpolate function
        original_interpolate = self.backbone.embeddings.interpolate_pos_encoding
        
        def patched_interpolate_pos_encoding(embeddings, width, height):
            """
            Core ML compatible position embedding interpolation
            Uses nearest instead of bicubic interpolation
            """
            num_patches = embeddings.shape[1] - 1
            num_positions = self.backbone.embeddings.position_embeddings.shape[1] - 1
            
            if num_patches == num_positions and height == width:
                return self.backbone.embeddings.position_embeddings
            
            # Separate class token and patch embeddings
            class_pos_embedding = self.backbone.embeddings.position_embeddings[:, 0]
            patch_pos_embedding = self.backbone.embeddings.position_embeddings[:, 1:]
            
            dim = embeddings.shape[-1]
            h0 = height // self.backbone.embeddings.patch_size
            w0 = width // self.backbone.embeddings.patch_size
            
            # Add small offset to avoid floating point errors
            h0, w0 = h0 + 0.1, w0 + 0.1
            
            sqrt_num_positions = int(num_positions**0.5)
            patch_pos_embedding = patch_pos_embedding.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
            patch_pos_embedding = patch_pos_embedding.permute(0, 3, 1, 2)
            
            patch_pos_embedding = F.interpolate(
                patch_pos_embedding,
                size=(int(h0), int(w0)),
                mode="nearest",  # bicubic -> nearest interpolation
            )
            
            patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
            
            return torch.cat((class_pos_embedding.unsqueeze(0), patch_pos_embedding), dim=1)
        
        # Apply the patch
        self.backbone.embeddings.interpolate_pos_encoding = patched_interpolate_pos_encoding
        self._coreml_patch_applied = True
        
        print("   ✅ Core ML compatibility patch applied successfully!")
        print("   📝 bicubic interpolation -> nearest interpolation")
        print("   🎯 LoRA model is now ready for Core ML conversion")
    
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
        # Extract features using the backbone
        outputs = self.backbone(pixel_values)
            
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, 256, 384]
        B, N, D = patch_tokens.shape
            
        # Reshape to 2D feature map: 256 tokens -> 16x16 spatial
        H = W = int(N ** 0.5)  # 16 = sqrt(256)
        # Use contiguous() to ensure memory layout compatibility
        spatial_features = patch_tokens.contiguous().view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, 384, 16, 16]
        
        heatmaps, z_coords = self.pose_heads(spatial_features)
        
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