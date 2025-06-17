import torch
import torch.nn as nn
from typing import Tuple, Optional


class HeatmapHead(nn.Module):
    """
    Heatmap head for 2D keypoint detection
    Converts feature vectors to spatial heatmaps using transposed convolutions
    """
    
    def __init__(self, 
                 feat_dim: int, 
                 num_keypoints: int, 
                 heatmap_size: int = 48,
                 intermediate_features: int = 512,
                 spatial_size: int = 6):

        super().__init__()
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.spatial_size = spatial_size
        self.intermediate_features = intermediate_features
        
        self.feature_projection = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, spatial_size * spatial_size * intermediate_features),
            nn.ReLU()
        )
        
        # Calculate required upsampling stages
        # Current: 6x6 -> target heatmap_size
        self.num_stages = self._calculate_stages(spatial_size, heatmap_size)
        
        # Create upsampling layers
        self.upsampling_layers = self._create_upsampling_layers()
        
        # Final prediction layer
        self.prediction_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)
    
    def _calculate_stages(self, start_size: int, target_size: int) -> int:
        """Calculate number of upsampling stages needed"""
        stages = 0
        current_size = start_size
        while current_size < target_size:
            current_size *= 2
            stages += 1
        return stages
    
    def _create_upsampling_layers(self) -> nn.ModuleList:
        """Create upsampling layers based on target size"""
        layers = nn.ModuleList()
        
        # Calculate required stages dynamically
        current_size = self.spatial_size
        target_size = self.heatmap_size
        
        # First stage: intermediate_features -> 256
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(self.intermediate_features, 256, 
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ))
        current_size *= 2
        
        # Add more stages until we get close to target size
        in_channels = 256
        out_channels = 128
        
        while current_size < target_size:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            current_size *= 2
            in_channels = out_channels
            out_channels = max(64, out_channels // 2)  # Reduce channels but don't go below 64
        
        # If we overshot, add a final layer to adjust size
        if current_size > target_size:
            # Add an adaptive pooling layer or convolution to get exact size
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(target_size)
            ))
        elif in_channels != 64:
            # Ensure final layer has 64 channels
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ))
        
        return layers
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        # Project to spatial features
        projected = self.feature_projection(features)  # [B, spatial_size^2 * intermediate_features]
        
        # Reshape to spatial format
        spatial_features = projected.view(
            batch_size, self.intermediate_features, self.spatial_size, self.spatial_size
        )  # [B, intermediate_features, spatial_size, spatial_size]
        
        # Apply upsampling layers
        x = spatial_features
        for layer in self.upsampling_layers:
            x = layer(x)
        
        heatmaps = self.prediction_layer(x)  # [B, num_keypoints, heatmap_size, heatmap_size]
        
        return heatmaps


class ZCoordinateHead(nn.Module):
    """
    Z-coordinate head for depth estimation
    Predicts z-coordinates directly from feature vectors
    """
    
    def __init__(self, 
                 feat_dim: int, 
                 num_keypoints: int,
                 hidden_dims: Tuple[int, ...] = (1024, 512),
                 dropout_rate: float = 0.2):
 
        super().__init__()
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints
        
        # Build MLP layers
        layers = []
        in_dim = feat_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(in_dim, num_keypoints))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class PoseHeads(nn.Module):
    """
    Combin heatmap and z-coordinate prediction in a single module
    """
    
    def __init__(self,
                 feat_dim: int,
                 num_keypoints: int,
                 heatmap_size: int = 48,
                 heatmap_config: Optional[dict] = None,
                 z_coord_config: Optional[dict] = None):

        super().__init__()
        
        # Default configurations
        heatmap_config = heatmap_config or {}
        z_coord_config = z_coord_config or {}
        
        # Initialize heads
        self.heatmap_head = HeatmapHead(
            feat_dim=feat_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            **heatmap_config
        )
        
        self.z_head = ZCoordinateHead(
            feat_dim=feat_dim,
            num_keypoints=num_keypoints,
            **z_coord_config
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heatmaps = self.heatmap_head(features)
        z_coords = self.z_head(features)
        
        return heatmaps, z_coords
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters in the heads"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class HourglassModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, in_channels, 2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.down(x)
        x = self.up(x)
        return x + skip

class SpatialAwareHeatmapHead(nn.Module):
    """
    Spatial-aware heatmap head that preserves spatial information
    Designed specifically for FastViT with spatial feature maps
    """

    def __init__(self, 
                 feat_channels: int = 768,  # From FastViT final_conv
                 num_keypoints: int = 24, 
                 heatmap_size: int = 48,
                 spatial_input_size: int = 14):
        
        super().__init__()
        self.feat_channels = feat_channels
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.spatial_input_size = spatial_input_size
        
        # Feature refinement layers
        self.feature_refine = nn.Sequential(
            nn.Conv2d(feat_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            HourglassModule(256, 256)
        )
        
        upsampling_stages = []
        current_size = spatial_input_size
        in_channels = 256
        
        while current_size < heatmap_size:
            out_channels = max(128, in_channels // 2)
            stride = heatmap_size // current_size
            upsampling_stages.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 
                                 kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            current_size *= 2
            in_channels = out_channels
        
        self.upsampling = nn.Sequential(*upsampling_stages)
        
        final_channels = in_channels if upsampling_stages else 256
        self.prediction = nn.Sequential(
            nn.Conv2d(final_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        
        self.target_size = heatmap_size
        self.use_interpolation = (current_size != heatmap_size)
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # Refine features
        x = self.feature_refine(feature_map)  # [B, 256, H, W]
        x = self.upsampling(x)
        
        heatmaps = self.prediction(x)  # [B, num_keypoints, ?, ?]
        
        # Ensure exact target size using interpolation
        if self.use_interpolation:
            heatmaps = torch.nn.functional.interpolate(
                heatmaps, 
                size=(self.target_size, self.target_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return heatmaps


class SpatialAwarePoseHeads(nn.Module):
    """ Combine spatial-aware heatmap head with standard z-coordinate head """

    def __init__(self,
                 feat_channels: int = 768,
                 num_keypoints: int = 24,
                 heatmap_size: int = 48,
                 spatial_input_size: int = 14,
                 z_coord_config: Optional[dict] = None):

        super().__init__()
        
        # Default Z head configuration
        z_coord_config = z_coord_config or {}
        
        # Spatial-aware heatmap head
        self.heatmap_head = SpatialAwareHeatmapHead(
            feat_channels=feat_channels,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size,
            spatial_input_size=spatial_input_size
        )
        
        # Standard Z head (uses global pooled features)
        feat_dim = feat_channels 
        self.z_head = ZCoordinateHead(
            feat_dim=feat_dim,
            num_keypoints=num_keypoints,
            **z_coord_config
        )
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heatmaps = self.heatmap_head(feature_map)
        features = feature_map.mean(dim=(2, 3))
        z_coords = self.z_head(features)
        
        return heatmaps, z_coords 