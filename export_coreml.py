#!/usr/bin/env python3
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from model.dinov2_pose import Dinov2PoseModel, Dinov2PoseModelLoRA
from model.fastvit_pose import FastVitPoseModel, FastVitPoseModelLoRA


def detect_model_family(state_dict: Dict[str, torch.Tensor], checkpoint: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve model family and config from checkpoint.
    """
    print("🔍 Analyzing checkpoint to detect model family...")
    
    all_keys = list(state_dict.keys())
    key_str = " ".join(all_keys)
    
    # default config
    config = {
        'num_keypoints': 24,
        'heatmap_size': 48
    }
    
    # try to extract config from checkpoint
    if 'model_config' in checkpoint:
        config.update(checkpoint['model_config'])
        print(f"   📋 Found model_config in checkpoint: {checkpoint['model_config']}")
    
    # 1. detect LoRA model
    lora_indicators = ['lora_A', 'lora_B', '.lora.', 'lora_dropout']
    if any(indicator in key_str for indicator in lora_indicators):
        print("   🎯 Detected: DINOv2 LoRA Pose Model")
        
        # check DINOv2 backbone - default is facebook/dinov2-small
        if 'backbone.embeddings' in key_str or 'backbone.encoder' in key_str:
            # estimate backbone size from embedding dimension
            embedding_keys = [k for k in all_keys if 'backbone.embeddings.cls_token' in k]
            if embedding_keys:
                # extract actual embedding dimension from state_dict
                cls_token_key = embedding_keys[0]
                embedding_dim = state_dict[cls_token_key].shape[-1]
                
                if embedding_dim == 384:
                    backbone = 'facebook/dinov2-small'
                elif embedding_dim == 768:
                    backbone = 'facebook/dinov2-base'
                elif embedding_dim == 1024:
                    backbone = 'facebook/dinov2-large'
                else:
                    backbone = 'facebook/dinov2-small'
            else:
                backbone = 'facebook/dinov2-small'
            
            # Extract LoRA configuration from state_dict
            lora_rank = 8 
            lora_alpha = 16
            lora_dropout = 0.1
            
            # Find first LoRA A weight to determine rank
            lora_a_keys = [k for k in all_keys if 'lora_A.weight' in k]
            if lora_a_keys:
                first_lora_a = lora_a_keys[0]
                lora_rank = state_dict[first_lora_a].shape[0]
                print(f"   📊 Detected LoRA rank: {lora_rank}")
            
            config.update({
                'backbone': backbone,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout
            })
            
            return 'dinov2_lora', config
    
    # 2. detect standard DINOv2 model
    dinov2_indicators = ['backbone.embeddings', 'backbone.encoder', 'backbone.layernorm']
    if any(indicator in key_str for indicator in dinov2_indicators):
        print("   🎯 Detected: DINOv2 Pose Model")
        
        # estimate backbone size from embedding dimension
        embedding_keys = [k for k in all_keys if 'backbone.embeddings.cls_token' in k]
        if embedding_keys:
            # extract actual embedding dimension from state_dict
            cls_token_key = embedding_keys[0]
            embedding_dim = state_dict[cls_token_key].shape[-1]
            
            if embedding_dim == 384:
                backbone = 'facebook/dinov2-small'
            elif embedding_dim == 768:
                backbone = 'facebook/dinov2-base'
            elif embedding_dim == 1024:
                backbone = 'facebook/dinov2-large'
            else:
                backbone = 'facebook/dinov2-small'
        else:
            backbone = 'facebook/dinov2-small'
        
        # default config
        config.update({
            'backbone': backbone,
            'unfreeze_last_n_layers': 0
        })
        
        return 'dinov2', config
    
    # 3. detect FastViT model
    fastvit_indicators = ['backbone.patch_embed', 'backbone.stages', 'backbone.norm']
    if any(indicator in key_str for indicator in fastvit_indicators):
        # Check if this is a LoRA version
        if any(indicator in key_str for indicator in lora_indicators):
            print("   🎯 Detected: FastViT LoRA Pose Model")
            
            # estimate backbone size from embedding dimension
            if 'stages.3.' in key_str and 'stages.2.' in key_str:
                backbone = 'fastvit_t8.apple_in1k'
            else:
                backbone = 'fastvit_t8.apple_in1k'
            
            # Extract LoRA configuration from state_dict
            lora_rank = 8
            lora_alpha = 16
            lora_dropout = 0.1
            
            # Find first LoRA A weight to determine rank
            lora_a_keys = [k for k in all_keys if 'lora_A.weight' in k]
            if lora_a_keys:
                first_lora_a = lora_a_keys[0]
                lora_rank = state_dict[first_lora_a].shape[0]
                print(f"   📊 Detected LoRA rank: {lora_rank}")
            
            config.update({
                'backbone': backbone,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout
            })
            
            return 'fastvit_lora', config
        else:
            print("   🎯 Detected: FastViT Pose Model")
            
            # estimate backbone size from embedding dimension
            if 'stages.3.' in key_str and 'stages.2.' in key_str:
                backbone = 'fastvit_t8.apple_in1k'
            else:
                backbone = 'fastvit_t8.apple_in1k'
            
            config.update({
                'backbone': backbone
            })
            
            return 'fastvit', config
    
    # 4. unknown model
    print("   ❓ Unknown model family detected")
    print(f"   📝 Available keys preview: {all_keys[:5]}...")
    
    return 'unknown', config


def create_model_from_family(family: str, config: Dict[str, Any]):
    """ Create model from detected family and config """
    print(f"🏗️  Creating {family} model with config: {config}")
    
    if family == 'dinov2':
        return Dinov2PoseModel(
            num_keypoints=config['num_keypoints'],
            backbone=config['backbone'],
            heatmap_size=config['heatmap_size'],
            unfreeze_last_n_layers=config.get('unfreeze_last_n_layers', 0)
        )
    
    elif family == 'dinov2_lora':
        return Dinov2PoseModelLoRA(
            num_keypoints=config['num_keypoints'],
            backbone=config['backbone'],
            heatmap_size=config['heatmap_size'],
            lora_rank=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.1)
        )
    
    elif family == 'fastvit':
        return FastVitPoseModel(
            num_keypoints=config['num_keypoints'],
            backbone=config['backbone'],
            heatmap_size=config['heatmap_size']
        )
    
    elif family == 'fastvit_lora':
        return FastVitPoseModelLoRA(
            num_keypoints=config['num_keypoints'],
            backbone=config['backbone'],
            heatmap_size=config['heatmap_size'],
            lora_rank=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unsupported model family: {family}")


class CoreMLWrapper(nn.Module):
    """
    Core ML 변환을 위한 범용 래퍼
    """
    
    def __init__(self, pose_model, family: str):
        super().__init__()
        self.model = pose_model
        self.family = family
        
        # 정규화 파라미터 설정
        if family.startswith('dinov2'):
            # DINOv2는 ImageNet 정규화 사용
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        elif family == 'fastvit':
            # FastViT도 ImageNet 정규화 사용 (동일)
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            # 기본 정규화
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, image: torch.Tensor):
        """
        Core ML 입력 처리 (0-1 범위 이미지를 받아서 정규화)
        """
        normalized = (image - self.mean) / self.std
        return self.model(normalized)


def export_dinov2_to_coreml(model, output_path: str) -> bool:
    """ Convert DINOv2 model to Core ML """
    print("🔧 Applying Core ML compatibility patch for DINOv2...")
    model.apply_coreml_compatibility_patch()
    
    return _export_to_coreml(model, 'dinov2', output_path)


def export_dinov2_lora_to_coreml(model, output_path: str) -> bool:
    """ Convert DINOv2 LoRA model to Core ML """
    print("🔧 Applying Core ML compatibility patch for DINOv2 LoRA...")
    model.apply_coreml_compatibility_patch()
    
    return _export_to_coreml(model, 'dinov2_lora', output_path)


def export_fastvit_to_coreml(model, output_path: str) -> bool:
    """ Convert FastViT model to Core ML """
    print("📦 FastViT model - no special patches needed")
    
    return _export_to_coreml(model, 'fastvit', output_path)

def export_fastvit_lora_to_coreml(model, output_path: str) -> bool:
    """ Convert FastViT LoRA model to Core ML """
    return _export_to_coreml(model, 'fastvit_lora', output_path)

def _export_to_coreml(model, family: str, output_path: str) -> bool:
    """ Common Core ML conversion logic """
    try:
        # 1. create wrapper
        print("📦 Creating Core ML wrapper...")
        wrapper = CoreMLWrapper(model, family)
        wrapper.eval()
        
        # 2. test model
        print("🧪 Testing model...")
        test_input = torch.randn(1, 3, 224, 224) * 0.5 + 0.5  # 0-1 range
        
        with torch.no_grad():
            heatmaps, depths = wrapper(test_input)
            print(f"   ✅ Forward pass successful!")
            print(f"      - Heatmaps: {heatmaps.shape}")
            print(f"      - Depths: {depths.shape}")
        
        # 3. convert to TorchScript
        print("📊 Converting to TorchScript...")
        traced_model = torch.jit.trace(wrapper, test_input)
        
        # 4. convert to Core ML
        print("🔄 Converting to Core ML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=(1, 3, 224, 224),
                    scale=1.0/255.0,
                    bias=[0, 0, 0],
                    color_layout=ct.colorlayout.RGB
                )
            ],
            outputs=[
                ct.TensorType(name="heatmaps"),
                ct.TensorType(name="depths")
            ],
            minimum_deployment_target=ct.target.iOS15,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL  # Use CPU + GPU + Neural Engine
        )
        
        # 5. set metadata
        print("💾 Setting metadata and saving...")
        mlmodel.short_description = f"{family.upper()} pose estimation model (24 keypoints)"
        mlmodel.input_description["image"] = "RGB image (224x224)"
        mlmodel.output_description["heatmaps"] = "Keypoint heatmaps (24, 48, 48)"
        mlmodel.output_description["depths"] = "Keypoint depths (24,)"
        
        # 6. 저장
        mlmodel.save(output_path)
        
        # 파일 크기 확인
        if os.path.isdir(output_path):
            size_mb = sum(
                os.path.getsize(os.path.join(output_path, f))
                for f in os.listdir(output_path)
                if os.path.isfile(os.path.join(output_path, f))
            ) / (1024 * 1024)
        else:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"   ✅ Model saved: {output_path}")
        print(f"   📊 Size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return False


def load_checkpoint_and_export(checkpoint_path: str, output_path: str) -> bool:
    """ Load checkpoint and perform appropriate conversion """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    # extract state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("   📝 Found model_state_dict in checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("   📝 Found state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("   📝 Using checkpoint as state_dict directly")
    
    # detect model family
    family, config = detect_model_family(state_dict, checkpoint)
    
    if family == 'unknown':
        print("❌ Could not determine model family")
        return False
    
    # create model
    try:
        model = create_model_from_family(family, config)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"   ✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to create/load model: {e}")
        return False
    
    # select appropriate export function
    export_functions = {
        'dinov2': export_dinov2_to_coreml,
        'dinov2_lora': export_dinov2_lora_to_coreml,
        'fastvit': export_fastvit_to_coreml,
        'fastvit_lora': export_fastvit_lora_to_coreml
    }
    
    export_func = export_functions[family]
    
    print(f"\n🎯 Starting {family.upper()} → Core ML conversion")
    print("=" * 60)
    
    success = export_func(model, output_path)
    
    if success:
        print(f"\n✅ Export completed successfully!")
        print(f"📱 Core ML model ready: {output_path}")
    else:
        print(f"\n❌ Export failed!")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Smart Core ML Exporter - Automatically detects model type and exports to Core ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python export_coreml.py --checkpoint best_model.pth --output MyModel.mlpackage
python export_coreml.py -c checkpoint.pth -o ./models/converted_model.mlpackage

Supported Models:
- DINOv2 Pose Model
- DINOv2 LoRA Pose Model  
- FastViT Pose Model
"""
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for the Core ML model (.mlpackage)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    if not args.output.endswith('.mlpackage'):
        print("⚠️  Adding .mlpackage extension to output path")
        args.output += '.mlpackage'
    
    # create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🎯 Smart Core ML Exporter")
    print("=" * 50)
    print(f"📂 Input:  {args.checkpoint}")
    print(f"📱 Output: {args.output}")
    print()
    
    # run conversion
    success = load_checkpoint_and_export(args.checkpoint, args.output)
    
    if success:
        print("\n🎉 All done! Your model is ready for iOS deployment.")
        sys.exit(0)
    else:
        print("\n💥 Export failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 