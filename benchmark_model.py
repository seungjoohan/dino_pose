import torch
import time
from model.dinov2_pose import Dinov2PoseModel
from model.fastvit_pose import FastVitPoseModelLoRA, FastVitPoseModel
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np

def benchmark_model():
    # Device 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("Loading model...")
    # 모델 로딩 (작은 버전 사용)
    model = Dinov2PoseModel(num_keypoints=24, backbone='facebook/dinov2-small')
    # model = FastVitPoseModel(num_keypoints=24, backbone='fastvit_t8.apple_in1k')
    model.to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    
    # 더미 이미지 생성
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    print("Warming up...")
    # 워밍업
    for _ in range(3):
        inputs = processor(dummy_image, return_tensors='pt')
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            _ = model(pixel_values)
    
    print("Benchmarking...")
    # 성능 측정
    times = []
    for i in range(20):
        start = time.time()
        inputs = processor(dummy_image, return_tensors='pt')
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            heatmaps, depths = model(pixel_values)
        end = time.time()
        times.append(end - start)
        print(f'Inference {i+1}: {(end-start)*1000:.1f}ms')
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f'\nResults:')
    print(f'Average inference time: {avg_time*1000:.1f}ms')
    print(f'Average FPS: {fps:.1f}')
    print(f'Model parameters: {model.count_parameters():,}')
    
    # 실시간 처리 가능성 분석
    print(f'\nReal-time Analysis:')
    print(f'Device: {device}')
    print(f'For 30 FPS video: Need <{1000/30:.1f}ms per frame')
    print(f'For 60 FPS video: Need <{1000/60:.1f}ms per frame')
    print(f'Current performance: {"✅ Real-time capable" if fps >= 30 else "❌ Not real-time"}')

if __name__ == "__main__":
    benchmark_model() 