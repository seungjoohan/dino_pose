# DINOv2 for Keypoint Estimation

This project uses DINOv2 (a Vision Transformer model) as a backbone for keypoint estimation, training on a custom dataset. The implementation includes custom heads to predict 2D heatmaps and 3D coordinate values for keypoints.

## Features

- Uses pre-trained DINOv2 as a backbone for feature extraction
- Custom heads for 2D heatmap prediction and 3D z-coordinate estimation
- Support for arbitrary number of keypoints (current version trained with 24 keypoints)
- Simple inference demo with visualization
- Trained using M2 Macbook pro

## Quick Start

### Run the Hugging Face Implementation

```bash
# Run with your own image
python src/demo.py --image path/to/your/image.jpg

# Choose a different DINOv2 model variant
python src/demo.py --image path/to/your/image.jpg --model facebook/dinov2-base
```

Fine-tuned model will soon be added as one of the model choices


## Model Details

The model architecture consists of:

1. **Backbone**: Frozen DINOv2 (Vision Transformer) for feature extraction
2. **Heatmap Head**: Multi-layer perceptron for 2D keypoint heatmap prediction
3. **Z-coords Head**: Multi-layer perceptron for z coordinate prediction

### Finetune DINOv2 keypoint estimation 
```bash
python train.py --config_file config/config.py
```

### Implementation using HuggingFace:
- Uses `Dinov2Model.from_pretrained("facebook/dinov2-small")`
- Automatic preprocessing with `AutoImageProcessor`
- Access to all DINOv2 variants through model hub

## References

- [DINOv2 GitHub Repository](https://github.com/facebookresearch/dinov2)
- [DINOv2 Paper: "DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193)
- [Hugging Face DINOv2 Models](https://huggingface.co/facebook/dinov2-small)
