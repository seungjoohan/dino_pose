# 🎯 Pose Estimation iOS Test App

A SwiftUI app for real-time pose estimation. Automatically discovers Core ML models in the `test_models` directory and performs pose detection from live camera feed.

## ✨ Key Features

- **Automatic Model Discovery**: Auto-scans `.mlpackage` models in `test_models/` directory
- **Model Selection**: Support for various model families including DINOv2, FastViT, etc.
- **Real-time Inference**: Real-time pose estimation using front camera
- **Skeleton Rendering**: Automatically displays skeleton when confidence ≥ 0.5
- **Performance Monitoring**: Real-time display of FPS, inference time, and average confidence

## 📁 Project Structure

```
ios_test_app/
├── PoseTestApp.swift      # Main app file
└── README.md             # This file

test_models/              # Models directory
├── dino_small_lora/
│   └── model.mlpackage
├── dinov2_base/
│   └── model.mlpackage
└── fastvit_test/
    └── model.mlpackage
```

## 🚀 Installation & Setup

### 1. Create Xcode Project

1. Create new iOS project in Xcode
2. Interface: Select SwiftUI
3. Language: Select Swift
4. Minimum iOS version: 15.0+

### 2. Add Files

1. Add `PoseTestApp.swift` file to project
2. Add entire `test_models` directory to Bundle Resources:
   - Select project root in project navigator
   - Build Phases → Copy Bundle Resources
   - Drag and drop `test_models` folder

### 3. Permission Setup

Add camera permission to `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app uses camera for real-time pose estimation</string>
```

### 4. Add Core ML Models

- Convert PyTorch models to Core ML using `export_coreml.py`
- Save generated `.mlpackage` files in `test_models/[model_name]/` directory

## 📱 Usage

### 1. Main Screen
- App automatically scans `test_models` directory on startup
- Available models displayed in list
- Select desired model

### 2. Real-time Pose Estimation
- Tap "Start Posing!" button
- Front camera activates
- Real-time pose detection performed
- Skeleton automatically displayed when confidence ≥ 0.5

### 3. UI Elements
- **Model Name**: Shows currently used model
- **FPS**: Frames per second (inference performance)
- **Conf**: Average keypoint confidence (0.0 ~ 1.0)
- **X Button**: Return to main screen

## 🎨 Skeleton Rendering

### Keypoint Colors
- 🟢 **Green**: Confidence > 0.7 (High confidence)
- 🟡 **Yellow**: Confidence 0.3 ~ 0.7 (Medium confidence)
- Transparent: Confidence < 0.3 (Not displayed)

### Skeleton Connections
- COCO-style skeleton based on 26 keypoints
- Connected with cyan lines
- Only displayed when both keypoints have confidence ≥ 0.3

## 🔧 Model Requirements

### Input
- **Image**: 224×224 RGB
- **Format**: CVPixelBuffer (kCVPixelFormatType_32BGRA)

### Output
- **heatmaps**: Shape (1, 24, 48, 48) - Heatmap for each keypoint
- **depths**: Shape (1, 24) - Depth information for each keypoint

### Supported Models
- DINOv2 (Standard/LoRA)
- FastViT
- Other models with same output format

## 🚨 Troubleshooting

### Models Not Detected
1. Verify `test_models` folder is added to Bundle Resources
2. Check `.mlpackage` files are in correct subdirectories
3. Perform Clean Build in Xcode

### Camera Not Working
1. Test on real device, not simulator
2. Verify camera permission added to `Info.plist`
3. Allow camera access for app in Settings

### Performance Issues
1. iPhone X or later recommended (A11 or later)
2. iOS 15.0+ required
3. Use latest devices for Neural Engine acceleration

## 📊 Performance Benchmarks

| Model | iPhone 13 Pro | iPhone 11 | iPhone XS |
|-------|---------------|-----------|-----------|
| DINOv2-small | ~100 FPS | ~60 FPS | ~40 FPS |
| FastViT-T8 | ~80 FPS | ~50 FPS | ~35 FPS |

*Actual performance may vary depending on device and environment.* 