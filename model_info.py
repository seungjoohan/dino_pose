#!/usr/bin/env python3
"""
Model Information Tool
Provides detailed information about supported models, checkpoints, and families.
"""

import argparse
import os
import sys
from pathlib import Path
from model.model_utils import (
    list_supported_backbones, 
    get_model_info, 
    list_families,
    get_family_models,
    FAMILY_DEFAULTS,
    FAMILY_INFO
)

def format_model_name(name: str, max_length: int = 25) -> str:
    """Format model name with consistent width"""
    if len(name) <= max_length:
        return name.ljust(max_length)
    else:
        return name[:max_length-3] + "..."

def print_families():
    """Print information about model families"""
    print("🏠 Model Families Overview:")
    print("=" * 60)
    
    families = list_families()
    
    for family_name, family_data in families.items():
        info = family_data['info']
        default_model = family_data['default']
        models = family_data['models']
        
        print(f"\n📦 {family_name.upper()} Family")
        print(f"   Description: {info.get('description', 'No description available')}")
        
        print(f"   Features: {', '.join(info.get('features', []))}")
        print(f"   Available sizes: {', '.join(info.get('available_sizes', []))}")
        print(f"   Default model: {default_model}")
        print(f"   Total models: {len(models)}")
        
        print(f"   📋 Models in this family:")
        for model in models:
            backbone_info = list_supported_backbones()[model]
            lora_support = "✅" if backbone_info['lora_class'] else "❌"
            keypoints = backbone_info['default_config']['num_keypoints']
            print(f"     • {model} | LoRA: {lora_support} | Keypoints: {keypoints}")

def print_backbones():
    """Print all supported backbone models grouped by family"""
    print("🔧 Supported Backbone Models:")
    print("=" * 60)
    
    backbones = list_supported_backbones()
    families = list_families()
    
    # Group by family
    for family_name in families.keys():
        family_models = get_family_models(family_name)
        
        if family_models:
            # Family header
            family_info = FAMILY_INFO.get(family_name, {})
            print(f"\n📦 {family_name.upper()} Family:")
            if family_info.get('description'):
                print(f"   {family_info['description']}")
            
            # Show default
            default_model = FAMILY_DEFAULTS.get(family_name)
            if default_model:
                print(f"   Default: '{family_name}' → {default_model}")
            
            print()
            
            # List models in family
            for model_name, info in family_models.items():
                class_name = info['model_class'].__name__
                lora_support = "✅" if info['lora_class'] else "❌"
                keypoints = info['default_config']['num_keypoints']
                
                formatted_name = format_model_name(model_name, 30)
                print(f"  🔹 {formatted_name} | Class: {class_name:<18} | LoRA: {lora_support} | Keypoints: {keypoints}")
    
    # Show models without families (if any)
    orphaned_models = {name: info for name, info in backbones.items() 
                      if not info.get('family') or info['family'] not in families}
    
    if orphaned_models:
        print(f"\n📦 Other Models:")
        for model_name, info in orphaned_models.items():
            class_name = info['model_class'].__name__
            lora_support = "✅" if info['lora_class'] else "❌"
            keypoints = info['default_config']['num_keypoints']
            
            formatted_name = format_model_name(model_name, 30)
            print(f"  🔹 {formatted_name} | Class: {class_name:<18} | LoRA: {lora_support} | Keypoints: {keypoints}")
    
    print(f"\n📊 Total: {len(backbones)} supported backbones in {len(families)} families")

def print_checkpoint_info(checkpoint_path: str):
    """Print detailed information about a checkpoint"""
    try:
        info = get_model_info(checkpoint_path)
        
        print(f"📄 Checkpoint Information: {checkpoint_path}")
        print("=" * 60)
        
        # Training info
        print("🎯 Training Information:")
        print(f"   Epoch: {info.get('epoch', 'unknown')}")
        print(f"   Training Loss: {info.get('train_loss', 'unknown')}")
        print(f"   Validation Loss: {info.get('valid_loss', 'unknown')}")
        print(f"   Loss Weight: {info.get('loss_weight', 'unknown')}")
        
        # Model info
        print("\n🏗️ Model Architecture:")
        print(f"   Model Type: {info.get('model_type', 'unknown')}")
        print(f"   Class Name: {info.get('class_name', 'unknown')}")
        print(f"   Backbone: {info.get('backbone', 'unknown')}")
        print(f"   Family: {info.get('family', 'unknown')}")
        print(f"   Keypoints: {info.get('num_keypoints', 'unknown')}")
        print(f"   Heatmap Size: {info.get('heatmap_size', 'unknown')}")
        print(f"   LoRA Enabled: {'Yes' if info.get('use_lora', False) else 'No'}")
        
        # File info
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"\n💾 File Information:")
        print(f"   File Size: {file_size:.2f} MB")
        print(f"   File Path: {os.path.abspath(checkpoint_path)}")
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return False
    
    return True

def list_checkpoints(directory: str = "checkpoints"):
    """List all checkpoint files in a directory"""
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' does not exist")
        return
    
    checkpoint_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                checkpoint_files.append(os.path.join(root, file))
    
    if not checkpoint_files:
        print(f"📂 No checkpoint files found in '{directory}'")
        return
    
    print(f"📂 Checkpoint files in '{directory}':")
    print("=" * 60)
    
    for i, checkpoint_path in enumerate(sorted(checkpoint_files), 1):
        rel_path = os.path.relpath(checkpoint_path)
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        
        try:
            info = get_model_info(checkpoint_path)
            model_type = info.get('model_type', 'unknown')
            epoch = info.get('epoch', '?')
            family = info.get('family', 'unknown')
            
            print(f"{i:2d}. {rel_path}")
            print(f"    📊 Epoch: {epoch} | Type: {model_type} | Family: {family} | Size: {file_size:.1f}MB")
        except:
            print(f"{i:2d}. {rel_path}")
            print(f"    ⚠️  Could not read checkpoint info | Size: {file_size:.1f}MB")
        print()

def main():
    parser = argparse.ArgumentParser(description='Model Information Tool')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', '-c', type=str,
                      help='Path to checkpoint file (.pth) to analyze')
    group.add_argument('--backbones', '-b', action='store_true',
                      help='List all supported backbone models')
    group.add_argument('--families', '-f', action='store_true',
                      help='Show information about model families')
    group.add_argument('--list-checkpoints', '-l', type=str, nargs='?', 
                      const='checkpoints', metavar='DIR',
                      help='List checkpoint files in directory (default: checkpoints)')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        
        if not print_checkpoint_info(args.checkpoint):
            sys.exit(1)
            
    elif args.backbones:
        print_backbones()
        
    elif args.families:
        print_families()
        
    elif args.list_checkpoints:
        list_checkpoints(args.list_checkpoints)

if __name__ == "__main__":
    main() 