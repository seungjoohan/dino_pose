import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.model_utils import plot_keypoints
from model.model_utils import load_model_smart, get_model_info
from transformers import AutoImageProcessor

def main():
    parser = argparse.ArgumentParser(description='DINOv2 Keypoint Detection Demo')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default="facebook/dinov2-base",
                        help='Path to model checkpoint or DINOv2 model name (e.g., facebook/dinov2-small)')
    parser.add_argument('--num_keypoints', type=int, default=24,
                        help='Number of keypoints to detect (only used for pretrained models)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for keypoint visualization')
    parser.add_argument('--keypoint_label', type=bool, default=True,
                        help='Whether to show keypoint labels in visualization')
    parser.add_argument('--show_info', action='store_true',
                        help='Show model information before inference')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Show model info if requested
    if args.show_info and args.model.endswith('.pth'):
        print("=== Model Information ===")
        try:
            info = get_model_info(args.model)
            for key, value in info.items():
                print(f"{key}: {value}")
            print("=" * 25)
        except Exception as e:
            print(f"Could not load model info: {e}")
    
    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = load_model_smart(args.model, device=device, eval_mode=True)
        
        # For pretrained models, override num_keypoints if specified
        if args.model.startswith("facebook/") and args.num_keypoints != 24:
            print(f"Note: Pretrained model uses default 24 keypoints, --num_keypoints={args.num_keypoints} ignored")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get model's image processor
    if hasattr(model, 'image_processor'):
        image_processor = model.image_processor
    elif hasattr(model, 'backbone_name'):
        # All models should have backbone_name now
        image_processor = AutoImageProcessor.from_pretrained(model.backbone_name)
    else:
        # Ultimate fallback
        print("Warning: Could not determine backbone name, using default DINOv2 processor")
        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    
    # Load and preprocess the image
    print(f"Processing image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    inputs = image_processor(image, return_tensors="pt").to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        heatmaps, depths = model(inputs.pixel_values)
    
    # Convert outputs to numpy
    heatmaps_np = heatmaps.squeeze().cpu().numpy()
    depths_np = depths.squeeze().cpu().numpy()

    # Print results
    print(f"Predicted 2D heatmaps shape: {heatmaps_np.shape}")
    print(f"Predicted 3D depths shape: {depths_np.shape}")
    print(f"Model info: {model.num_keypoints} keypoints, {model.heatmap_size}x{model.heatmap_size} heatmap size")
    
    # Visualize the results
    print("Visualizing results...")
    fig = plot_keypoints(image, heatmaps_np, keypoint_label=args.keypoint_label)
    
    # Save the visualization if requested
    if args.output:
        fig.savefig(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        plt.show()
    
    print("Done!")

if __name__ == '__main__':
    main() 