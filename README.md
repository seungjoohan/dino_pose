# Finetuning Vision Transformers for Keypoint Estimation

This project is to train different **Vision Transformer** models as a backbone for keypoint estimation, training on a custom dataset. The implementation includes custom heads to predict 2D heatmaps and 3D coordinate values for keypoints. As pose models are primarily focused on on-device performances, I'll try to achieve more accurate, but optimized for on-device inference performance.

Currently supports FastViT and DinoV2 backbones!

```bash
# See what options are supported
python model_info.py --backbones
```

## Features

- Uses pre-trained Vision Transformers as a backbone for feature extraction
- Custom heads for 2D heatmap prediction and 3D z-coordinate estimation
- Support for arbitrary number of keypoints (current version trained with **24 keypoints**)
- Simple inference demo with visualization
- Trained Model CoreML conversion
- Swift UI app to apply finetuned models

## Quick Start

### Run the Hugging Face Implementation

```bash
# Run with your own input - supports both image and video
python src/demo.py --input path/to/your/image.jpg #--output path/to/video.mp4 (for video input)

# Choose a different DINOv2 model variant
python src/demo.py --input path/to/your/image.jpg --model facebook/dinov2-base
```

Fine-tuned model will soon be added as one of the model choices


## Model Details

The model architecture consists of:

1. **Backbone**: Frozen DINOv2 (Vision Transformer) for feature extraction
2. **Heatmap Head**: Multi-layer perceptron for 2D keypoint heatmap prediction
3. **Z-coords Head**: Multi-layer perceptron for z coordinate prediction

### Finetune ViT pose estimation 
```bash
python train.py --config_file config/config.py
```

### CoreML Conversion
```bash
python export_coreml.py -c /path/to/trained/model.pth -o /path/to/save/coreml.mlpackage
```

### Testing finetuned model on Swift UI
- Checkout README in `ios_test_app/` to run SwiftUI App

## References

- [DINOv2 GitHub Repository](https://github.com/facebookresearch/dinov2)
- [DINOv2 Paper: "DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193)
- [Hugging Face DINOv2 Models](https://huggingface.co/facebook/dinov2-small)
