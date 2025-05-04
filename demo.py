import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.model_utils import plot_keypoints

from model.dinov2_pose import Dinov2PoseModel
from transformers import AutoImageProcessor

def main():
    parser = argparse.ArgumentParser(description='DINOv2 Keypoint Detection Demo')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--num_keypoints', type=int, default=24,
                        help='Number of keypoints to detect')
    parser.add_argument('--model', type=str, default="facebook/dinov2-base",
                        help='DINOv2 model variant to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for keypoint visualization')
    parser.add_argument('--keypoint_label', type=bool, default=True,
                        help='Whether to show keypoint labels in visualization')
    args = parser.parse_args()
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Check if image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Load and preprocess the image
    print(f"Processing image: {args.image}")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    image = Image.open(args.image)
    inputs = image_processor(image, return_tensors="pt").to(device)
    
    # Create the model
    if args.model.startswith("facebook"):
        print(f"Loading DINOv2 pose model ({args.model}) with {args.num_keypoints} keypoints...")
        model = Dinov2PoseModel(
            num_keypoints=args.num_keypoints,
            backbone=args.model,
            heatmap_size=image_processor.crop_size['height']
        )
    else:
        model_dict = torch.load(args.model)
        model = Dinov2PoseModel(
            num_keypoints=model_dict['config_model']['num_keypoints'],
            backbone=model_dict['config_model']['model_name'],
            heatmap_size=model_dict['config_model']['output_heatmap_size']
        )
        model.load_state_dict(model_dict['model_state_dict'])
        model.eval()
    model.to(device)
    
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
    
    # Visualize the results
    print("Visualizing results...")
    fig = plot_keypoints(image, heatmaps_np, keypoint_label=args.keypoint_label)
    
    # Save the visualization if requested
    if args.output:
        fig.savefig(args.output)
        print(f"Visualization saved to {args.output}")
    
    print("Done!")

if __name__ == "__main__":
    main() 