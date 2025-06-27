import torch
import os
from typing import Union, Dict, Any, Optional
from transformers import AutoModel, AutoImageProcessor
from .base_pose import BasePoseModel
from .dinov2_pose import Dinov2PoseModel, Dinov2PoseModelLoRA
from .fastvit_pose import FastVitPoseModel, FastVitPoseModelLoRA

# Backbone registry for supported models
BACKBONE_REGISTRY = {
    # DINOv2 models
    'facebook/dinov2-small': {
        'model_class': Dinov2PoseModel,
        'lora_class': Dinov2PoseModelLoRA,
        'family': 'dinov2',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False,
            'unfreeze_last_n_layers': 0
        }
    },
    'facebook/dinov2-base': {
        'model_class': Dinov2PoseModel,
        'lora_class': Dinov2PoseModelLoRA,
        'family': 'dinov2',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False,
            'unfreeze_last_n_layers': 0
        }
    },
    'facebook/dinov2-large': {
        'model_class': Dinov2PoseModel,
        'lora_class': Dinov2PoseModelLoRA,
        'family': 'dinov2',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False,
            'unfreeze_last_n_layers': 0
        }
    },
    # FastViT models
    'timm/fastvit_t8.apple_in1k': {
        'model_class': FastVitPoseModel,
        'lora_class': FastVitPoseModelLoRA,
        'family': 'fastvit',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False
        }
    },
    'timm/fastvit_ma36.apple_in1k': {
        'model_class': FastVitPoseModel,
        'lora_class': FastVitPoseModelLoRA,
        'family': 'fastvit',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False
        }
    },
    'timm/fastvit_sa12.apple_in1k': {
        'model_class': FastVitPoseModel,
        'lora_class': FastVitPoseModelLoRA,
        'family': 'fastvit',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False
        }
    },
    'timm/fastvit_sa24.apple_in1k': {
        'model_class': FastVitPoseModel,
        'lora_class': FastVitPoseModelLoRA,
        'family': 'fastvit',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False
        }
    },
    'timm/fastvit_sa36.apple_in1k': {
        'model_class': FastVitPoseModel,
        'lora_class': FastVitPoseModelLoRA,
        'family': 'fastvit',
        'default_config': {
            'num_keypoints': 24,
            'output_heatmap_size': 48,
            'use_lora': False
        }
    }
}

# Family to default model mapping
FAMILY_DEFAULTS = {
    'dinov2': 'facebook/dinov2-small',
    'fastvit': 'timm/fastvit_t8.apple_in1k'
}

# Family information
FAMILY_INFO = {
    'dinov2': {
        'description': 'Vision Transformer with self-supervised learning',
        'features': ['Self-supervised pre-training', 'Strong feature representations', 'LoRA support'],
        'available_sizes': ['small (21M)', 'base (86M)', 'large (300M)'],
        'default': 'facebook/dinov2-small'
    },
    'fastvit': {
        'description': 'FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization',
        'features': ['Fast inference', 'Hybrid architecture', 'Mobile-optimized'],
        'available_sizes': ['t8 (4M)','sa12 (9M)', 'sa24 (22M)', 'sa36 (31M)', 'ma36 (44M)'],
        'default': 'timm/fastvit_t8.apple_in1k'
    }
}

def register_backbone(model_name: str, model_class, lora_class=None, family=None, default_config=None):
    """Register a new backbone model"""
    BACKBONE_REGISTRY[model_name] = {
        'model_class': model_class,
        'lora_class': lora_class,
        'family': family,
        'default_config': default_config or {}
    }

def register_family_default(family_name: str, default_model_name: str):
    """Register a default model for a family"""
    FAMILY_DEFAULTS[family_name] = default_model_name

def resolve_model_name(model_name_or_family: str) -> str:
    """
    Resolve family name to actual model name
    If it's already a full model name, return as-is
    If it's a family name, return the default model for that family
    """
    # Check if it's already a registered model name
    if model_name_or_family in BACKBONE_REGISTRY:
        return model_name_or_family
    
    # Check if it's a family name
    if model_name_or_family in FAMILY_DEFAULTS:
        return FAMILY_DEFAULTS[model_name_or_family]
    
    # Return as-is (might be a HuggingFace model)
    return model_name_or_family

def get_family_models(family_name: str) -> Dict[str, Dict]:
    """Get all models in a family"""
    family_models = {}
    for model_name, info in BACKBONE_REGISTRY.items():
        if info.get('family') == family_name:
            family_models[model_name] = info
    return family_models

def list_families() -> Dict[str, Dict]:
    """List all available families with their models"""
    families = {}
    for family_name in FAMILY_DEFAULTS.keys():
        families[family_name] = {
            'info': FAMILY_INFO.get(family_name, {}),
            'default': FAMILY_DEFAULTS[family_name],
            'models': list(get_family_models(family_name).keys())
        }
    return families

def is_supported_backbone(model_name: str) -> bool:
    """Check if a backbone is supported"""
    return model_name in BACKBONE_REGISTRY

def is_family_name(name: str) -> bool:
    """Check if name is a family name"""
    return name in FAMILY_DEFAULTS

def is_huggingface_model(model_name: str) -> bool:
    """Check if model name looks like a Hugging Face model and try to load it"""
    if '/' not in model_name:
        return False
    
    try:
        # Try to load model info without downloading the full model
        AutoImageProcessor.from_pretrained(model_name)
        return True
    except Exception:
        return False

def create_model_from_config(config_model: Dict[str, Any]) -> BasePoseModel:
    """
    Create model from configuration dictionary using factory method pattern
    Supports both full model names and family names
    """
    model_name_or_family = config_model['model_name']
    
    # Resolve family name to actual model name
    model_name = resolve_model_name(model_name_or_family)
    
    # Update config with resolved model name
    resolved_config = config_model.copy()
    resolved_config['model_name'] = model_name
    
    # Check if it's a registered backbone
    if model_name in BACKBONE_REGISTRY:
        registry_entry = BACKBONE_REGISTRY[model_name]
        
        if resolved_config.get('use_lora', False):
            if registry_entry['lora_class'] is None:
                family = registry_entry.get('family', 'this backbone')
                raise ValueError(f"LoRA not supported for {family} family (model: {model_name})")
            
            model_class = registry_entry['lora_class']
            print(f"Created LoRA model with {model_name} ({registry_entry.get('family', 'unknown')} family)")
        else:
            model_class = registry_entry['model_class']
            print(f"Created standard model with {model_name} ({registry_entry.get('family', 'unknown')} family)")
            
        # Use registry defaults and override with config
        model_config = registry_entry['default_config'].copy()
        model_config.update(resolved_config)
        
        # Use factory method - each model class handles its own creation logic
        return model_class.from_config(model_name, model_config)
    else:
        raise ValueError(f"Unsupported backbone: {model_name}. Use 'model_info.py --backbones' to see supported models and families.")

def save_model_checkpoint(model: BasePoseModel, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         train_loss: float,
                         valid_loss: float,
                         loss_weight: float,
                         config_model: Dict[str, Any],
                         config_training: Dict[str, Any],
                         config_preproc: Dict[str, Any],
                         save_path: str,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
    """
    Save model checkpoint with all necessary metadata
    """
    # Add model type information to config
    enhanced_config_model = config_model.copy()
    enhanced_config_model['model_class'] = model.__class__.__name__
    
    # Determine model type based on class name
    is_lora_model = 'LoRA' in model.__class__.__name__
    enhanced_config_model['model_type'] = 'lora' if is_lora_model else 'standard'
    
    # Get backbone name and family info
    if hasattr(model, 'backbone_name'):
        backbone_name = model.backbone_name
        # Add family info if available
        if backbone_name in BACKBONE_REGISTRY:
            enhanced_config_model['model_family'] = BACKBONE_REGISTRY[backbone_name].get('family', 'unknown')
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'config'):
        backbone_name = model.backbone.config.name_or_path
    else:
        backbone_name = 'unknown'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'loss_weight': loss_weight,
        'config_model': enhanced_config_model,
        'config_training': config_training,
        'config_preproc': config_preproc,
        'model_architecture': {
            'class_name': model.__class__.__name__,
            'num_keypoints': model.num_keypoints,
            'backbone': backbone_name,
            'heatmap_size': model.heatmap_size,
        }
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

def load_model_smart(model_path: str, 
                    device: Optional[torch.device] = None,
                    eval_mode: bool = True) -> BasePoseModel:
    """
    Smart model loading that automatically detects model type and creates appropriate architecture
    Supports both full model names and family names
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # Check if it's a checkpoint file
    if os.path.isfile(model_path) and model_path.endswith('.pth'):
        print(f"Loading model from checkpoint: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        if 'config_model' in checkpoint:
            config_model = checkpoint['config_model']
        elif 'model_architecture' in checkpoint:
            # Fallback: reconstruct config from architecture info
            arch = checkpoint['model_architecture']
            config_model = {
                'model_name': arch['backbone'],
                'num_keypoints': arch['num_keypoints'],
                'output_heatmap_size': arch['heatmap_size'],
                'use_lora': 'LoRA' in arch['class_name']
            }
        else:
            raise ValueError(f"Checkpoint {model_path} missing model configuration")
        
        # Create model from config (now with correct LoRA settings)
        model = create_model_from_config(config_model)
        
        # Load state dict with strict=False to handle potential missing keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        print(f"Loaded weights from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # fix loading issues
        if hasattr(model, 'apply_loading_fixes'):
            model.apply_loading_fixes()
        
        # LoRA model specific handling
        if 'LoRA' in model.__class__.__name__:
            print("Applying LoRA-specific loading fixes...")
            # check LoRA alpha and rank values
            for name, module in model.named_modules():
                if hasattr(module, 'alpha') and hasattr(module, 'rank'):
                    print(f"LoRA module {name}: alpha = {module.alpha}, rank = {module.rank}, scaling = {module.scaling}")
        
    elif is_supported_backbone(model_path) or is_family_name(model_path):
        # Resolve family name to actual model name
        actual_model_name = resolve_model_name(model_path)
        
        if is_family_name(model_path):
            print(f"Using family '{model_path}' -> default model: {actual_model_name}")
        else:
            print(f"Creating new model with registered backbone: {actual_model_name}")
        
        # Get default configuration from registry
        registry_entry = BACKBONE_REGISTRY[actual_model_name]
        config_model = registry_entry['default_config'].copy()
        config_model['model_name'] = actual_model_name
        
        model = create_model_from_config(config_model)
        
    elif is_huggingface_model(model_path):
        print(f"Attempting to create model with Hugging Face backbone: {model_path}")
        print("Warning: This backbone is not officially supported. Using FastViT model as fallback.")
        
        # Try to create a FastViT model with the given backbone
        try:
            model = FastVitPoseModel.from_config(model_path, {
                'num_keypoints': 24,
                'output_heatmap_size': 48
            })
            print(f"Successfully created FastViT model with backbone: {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to create model with backbone {model_path}: {e}")
        
    else:
        available_families = list(FAMILY_DEFAULTS.keys())
        available_models = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Invalid model path: {model_path}. "
                        f"Expected .pth file, family name {available_families}, "
                        f"or supported backbone model name. Use 'model_info.py --backbones' to see all options.")
    
    # Move to device and set eval mode
    model.to(device)
    if eval_mode:
        model.eval()
        # set evaluation mode for dropout and batch norm
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()
    
    print(f"Model loaded on device: {device}")
    return model

def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get model information without loading the full model
    """
    if not os.path.isfile(model_path) or not model_path.endswith('.pth'):
        raise ValueError(f"Invalid checkpoint path: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', 'unknown'),
        'valid_loss': checkpoint.get('valid_loss', 'unknown'),
        'loss_weight': checkpoint.get('loss_weight', 'unknown'),
    }
    
    if 'config_model' in checkpoint:
        config = checkpoint['config_model']
        info.update({
            'model_type': config.get('model_type', 'unknown'),
            'backbone': config.get('model_name', 'unknown'),
            'family': config.get('model_family', 'unknown'),
            'num_keypoints': config.get('num_keypoints', 'unknown'),
            'use_lora': config.get('use_lora', False),
            'heatmap_size': config.get('output_heatmap_size', 'unknown')
        })
    
    if 'model_architecture' in checkpoint:
        arch = checkpoint['model_architecture']
        info.update({
            'class_name': arch.get('class_name', 'unknown'),
            'architecture': arch
        })
    
    return info

def list_supported_backbones() -> Dict[str, Dict]:
    """List all supported backbone models"""
    return BACKBONE_REGISTRY.copy() 