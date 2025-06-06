import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
from pathlib import Path
import imageio

from src.model_utils import plot_keypoints, plot_3d_keypoints, plot_keypoints_combined
from model.model_utils import load_model_smart, get_model_info
from transformers import AutoImageProcessor

def is_video_file(file_path):
    """Check if file is a video or gif"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    gif_extensions = ['.gif']
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in video_extensions + gif_extensions

def extract_frames_from_video(video_path, max_frames=None):
    """Extract frames from video or gif"""
    frames = []
    
    if video_path.lower().endswith('.gif'):
        # Handle GIF files
        gif = imageio.mimread(video_path)
        for i, frame in enumerate(gif):
            if max_frames and i >= max_frames:
                break
            # Convert numpy array to PIL Image
            pil_frame = Image.fromarray(frame).convert('RGB')
            frames.append(pil_frame)
    else:
        # Handle video files
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
                
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            frame_count += 1
        
        cap.release()
    
    return frames

def get_video_fps(video_path):
    """Get FPS of video file"""
    if video_path.lower().endswith('.gif'):
        # For GIF, return a reasonable default FPS
        return 10.0
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 25.0  # Default to 25 FPS if can't determine

def process_video_frames(frames, model, image_processor, device, plot_mode, keypoint_label):
    """Process each frame with pose estimation"""
    processed_frames = []
    
    print(f"Processing {len(frames)} frames...")
    
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}", end='\r')
        
        # Preprocess and run inference
        inputs = image_processor(frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            heatmaps, depths = model(inputs.pixel_values)
        
        # Convert outputs to numpy
        heatmaps_np = heatmaps.squeeze().cpu().numpy()
        depths_np = depths.squeeze().cpu().numpy()
        
        # Create visualization
        if plot_mode == '2d':
            fig = plot_keypoints(frame, heatmaps_np, keypoint_label=keypoint_label)
        elif plot_mode == '3d':
            fig = plot_3d_keypoints(frame, heatmaps_np, depths_np, keypoint_label=keypoint_label)
        else:  # combined
            fig = plot_keypoints_combined(frame, heatmaps_np, depths_np, keypoint_label=keypoint_label)
        
        # Convert matplotlib figure to image
        fig.canvas.draw()
        
        # Save to a temporary buffer and read back as PIL Image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_img = Image.open(buf)
        img_array = np.array(pil_img.convert('RGB'))
        buf.close()
        
        processed_frames.append(img_array)
        plt.close(fig)  # Close to free memory
    
    print(f"\nCompleted processing {len(processed_frames)} frames")
    return processed_frames

def save_video_output(frames, output_path, fps=25.0):
    """Save processed frames as video or gif"""
    output_ext = Path(output_path).suffix.lower()
    
    if len(frames) == 0:
        print("No frames to save!")
        return
    
    # Ensure all frames have the same size
    if len(frames) > 1:
        # Get the size of the first frame
        target_height, target_width = frames[0].shape[:2]
        print(f"Target frame size: {target_width}x{target_height}")
        
        # Resize all frames to match the first frame
        resized_frames = []
        for i, frame in enumerate(frames):
            if frame.shape[:2] != (target_height, target_width):
                print(f"Resizing frame {i+1} from {frame.shape[1]}x{frame.shape[0]} to {target_width}x{target_height}")
                # Use PIL for consistent resizing
                pil_frame = Image.fromarray(frame)
                pil_frame = pil_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
                resized_frame = np.array(pil_frame)
                resized_frames.append(resized_frame)
            else:
                resized_frames.append(frame)
        frames = resized_frames
    
    if output_ext == '.gif':
        # Save as GIF
        print(f"Saving as GIF: {output_path}")
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        # Save as video
        print(f"Saving as video: {output_path}")
        
        height, width, _ = frames[0].shape
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    print(f"Video saved successfully: {output_path}")

def process_single_image(image_path, model, image_processor, device, args):
    """Process single image (original logic)"""
    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
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
    print(f"Visualizing results in {args.plot_mode} mode...")
    
    if args.plot_mode == '2d':
        fig = plot_keypoints(image, heatmaps_np, keypoint_label=args.keypoint_label)
    elif args.plot_mode == '3d':
        fig = plot_3d_keypoints(image, heatmaps_np, depths_np, keypoint_label=args.keypoint_label)
    else:  # combined
        fig = plot_keypoints_combined(image, heatmaps_np, depths_np, keypoint_label=args.keypoint_label)
    
    # Save the visualization if requested
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {args.output}")
    else:
        plt.show()
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='DINOv2 Keypoint Detection Demo - Supports images, videos, and GIFs')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image, video, or gif')
    parser.add_argument('--model', type=str, default="facebook/dinov2-base",
                        help='Path to model checkpoint or DINOv2 model name (e.g., facebook/dinov2-small)')
    parser.add_argument('--num_keypoints', type=int, default=24,
                        help='Number of keypoints to detect (only used for pretrained models)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization output (image/video/gif)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for keypoint visualization')
    parser.add_argument('--keypoint_label', type=bool, default=False,
                        help='Whether to show keypoint labels in visualization')
    parser.add_argument('--show_info', action='store_true',
                        help='Show model information before inference')
    parser.add_argument('--plot_mode', type=str, choices=['2d', '3d', 'combined'], default='combined',
                        help='Visualization mode: 2d (original), 3d (3D only), or combined (side by side)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (for videos/gifs)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS for video/gif (if not specified, uses input FPS)')
    
    # Keep backward compatibility
    parser.add_argument('--image', type=str, default=None,
                        help='(Deprecated) Use --input instead')
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.image and not args.input:
        args.input = args.image
        print("Warning: --image is deprecated, use --input instead")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Determine if input is video/gif or image
    is_video = is_video_file(args.input)
    print(f"Input type: {'Video/GIF' if is_video else 'Image'}")
    
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
    
    # Process input based on type
    if is_video:
        # Process video or gif
        print(f"Extracting frames from: {args.input}")
        frames = extract_frames_from_video(args.input, max_frames=args.max_frames)
        
        if len(frames) == 0:
            print("No frames extracted from video/gif!")
            return
        
        print(f"Extracted {len(frames)} frames")
        
        # For video/gif, force 2D mode for better performance and stability
        video_plot_mode = '2d'
        if args.plot_mode != '2d':
            print(f"Note: For video/GIF processing, using 2D mode instead of {args.plot_mode} mode for better performance")
        
        # Get FPS
        input_fps = get_video_fps(args.input)
        output_fps = args.fps if args.fps else input_fps
        print(f"Input FPS: {input_fps:.2f}, Output FPS: {output_fps:.2f}")
        
        # Process frames
        processed_frames = process_video_frames(
            frames, model, image_processor, device, 
            video_plot_mode, args.keypoint_label
        )
        
        # Save output
        if args.output:
            save_video_output(processed_frames, args.output, fps=output_fps)
        else:
            print("No output path specified for video. Use --output to save the result.")
            
    else:
        # Process single image
        process_single_image(args.input, model, image_processor, device, args)
    
    print("Done!")

if __name__ == '__main__':
    main() 