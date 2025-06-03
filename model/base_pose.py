import torch
import torch.nn as nn
from typing import Dict, Any
from abc import ABC, abstractmethod

class BasePoseModel(nn.Module, ABC):
    """
    Base class for all pose estimation models
    Provides common interface and factory method pattern
    """
    
    def __init__(self):
        super(BasePoseModel, self).__init__()
        self.num_keypoints = None
        self.heatmap_size = None
        self.backbone_name = None
        self.backbone = None
    
    @classmethod
    @abstractmethod
    def from_config(cls, model_name: str, config: Dict[str, Any]):
        """
        Factory method to create model from configuration
        Each subclass must implement this method
        
        Args:
            model_name: Backbone model name (e.g., 'facebook/dinov2-base')
            config: Configuration dictionary with model parameters
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def forward(self, pixel_values):
        """
        Forward pass of the model
        
        Args:
            pixel_values: Input images (B, C, W, H)
            
        Returns:
            heatmaps: Predicted 2D heatmaps (B, num_keypoints, width, height)  
            z_coords: Predicted z-coordinates (B, num_keypoints)
        """
        pass
    
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