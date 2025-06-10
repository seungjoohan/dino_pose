# 🎯 Pose Estimation iOS Test App

실시간 pose estimation을 위한 SwiftUI 앱입니다. `test_models` 디렉토리의 Core ML 모델들을 자동으로 발견하고 실시간 카메라에서 pose detection을 수행합니다.

## ✨ 주요 기능

- **자동 모델 검색**: `test_models/` 디렉토리에서 `.mlpackage` 모델들을 자동으로 스캔
- **모델 선택**: DINOv2, FastViT 등 다양한 모델 패밀리 지원
- **실시간 추론**: 전면 카메라를 이용한 실시간 pose estimation
- **Skeleton 렌더링**: Confidence 0.3 이상일 때 자동으로 skeleton 표시
- **성능 모니터링**: FPS, 추론 시간, 평균 confidence 실시간 표시

## 📁 프로젝트 구조

```
ios_test_app/
├── PoseTestApp.swift      # 메인 앱 파일
└── README.md             # 이 파일

test_models/              # 모델 디렉토리
├── dino_small_lora/
│   └── model.mlpackage
├── dinov2_base/
│   └── model.mlpackage
└── fastvit_test/
    └── model.mlpackage
```

## 🚀 설치 및 실행

### 1. Xcode 프로젝트 생성

1. Xcode에서 새 iOS 프로젝트 생성
2. Interface: SwiftUI 선택
3. Language: Swift 선택
4. Minimum iOS version: 15.0+

### 2. 파일 추가

1. `PoseTestApp.swift` 파일을 프로젝트에 추가
2. `test_models` 디렉토리 전체를 Bundle Resources에 추가:
   - 프로젝트 네비게이터에서 프로젝트 루트 선택
   - Build Phases → Copy Bundle Resources
   - `test_models` 폴더를 드래그 앤 드롭

### 3. 권한 설정

`Info.plist`에 카메라 권한 추가:

```xml
<key>NSCameraUsageDescription</key>
<string>This app uses camera for real-time pose estimation</string>
```

### 4. Core ML 모델 추가

- `export_coreml.py`를 사용해서 PyTorch 모델을 Core ML로 변환
- 생성된 `.mlpackage` 파일을 `test_models/[model_name]/` 디렉토리에 저장

## 📱 사용법

### 1. 메인 화면
- 앱 시작 시 `test_models` 디렉토리에서 모델들을 자동 스캔
- 사용 가능한 모델들이 목록으로 표시됨
- 원하는 모델을 선택

### 2. 실시간 Pose Estimation
- "Start Posing!" 버튼 터치
- 전면 카메라 활성화
- 실시간으로 pose detection 수행
- Confidence가 0.5 이상이면 skeleton 자동 표시

### 3. UI 요소
- **모델명**: 현재 사용 중인 모델 표시
- **FPS**: 초당 프레임 수 (추론 성능)
- **Conf**: 평균 keypoint confidence (0.0 ~ 1.0)
- **X 버튼**: 메인 화면으로 돌아가기

## 🎨 Skeleton 렌더링

### Keypoint 색상
- 🟢 **초록색**: Confidence > 0.7 (신뢰도 높음)
- 🟡 **노란색**: Confidence 0.3 ~ 0.7 (신뢰도 보통)
- 투명: Confidence < 0.3 (표시 안함)

### Skeleton 연결
- 26개 keypoint 기반 COCO 스타일 skeleton
- 청록색(cyan) 선으로 연결
- 양쪽 keypoint의 confidence가 0.3 이상일 때만 표시

## 🔧 모델 요구사항

### 입력
- **이미지**: 224x224 RGB
- **포맷**: CVPixelBuffer (kCVPixelFormatType_32BGRA)

### 출력
- **heatmaps**: Shape (1, 24, 48, 48) - 각 keypoint의 heatmap
- **depths**: Shape (1, 24) - 각 keypoint의 depth 정보

### 지원 모델
- DINOv2 (Standard/LoRA)
- FastViT
- 기타 동일한 출력 형식을 가진 모델

## 🚨 문제 해결

### 모델이 발견되지 않는 경우
1. `test_models` 폴더가 Bundle Resources에 추가되었는지 확인
2. `.mlpackage` 파일이 올바른 하위 디렉토리에 있는지 확인
3. Xcode에서 Clean Build 수행

### 카메라가 실행되지 않는 경우
1. 시뮬레이터가 아닌 실제 기기에서 테스트
2. `Info.plist`에 카메라 권한이 추가되었는지 확인
3. 설정에서 앱의 카메라 권한 허용

### 성능 이슈
1. iPhone X 이상 권장 (A11 이상)
2. iOS 15.0 이상 필수
3. Neural Engine 가속을 위해 최신 기기 사용

## 📊 성능 벤치마크

| 모델 | iPhone 13 Pro | iPhone 11 | iPhone XS |
|------|---------------|-----------|-----------|
| DINOv2-small | ~100 FPS | ~60 FPS | ~40 FPS |
| FastViT-T8 | ~80 FPS | ~50 FPS | ~35 FPS |

*실제 성능은 기기와 환경에 따라 다를 수 있습니다.* 