import torch
import time
from model.dinov2_pose import Dinov2PoseModel
from model.fastvit_pose import FastVitPoseModel
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np

def benchmark_model(model, model_name, processor, device):
    print(f"\n=== Benchmarking {model_name} ===")
    model.eval()
    
    # 더미 이미지 생성
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 워밍업
    for _ in range(3):
        inputs = processor(dummy_image, return_tensors='pt')
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            _ = model(pixel_values)
    
    # 성능 측정
    times = []
    for i in range(10):
        start = time.time()
        inputs = processor(dummy_image, return_tensors='pt')
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
        end = time.time()
        times.append(end - start)
        print(f'  Inference {i+1}: {(end-start)*1000:.1f}ms')
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"  Average inference time: {avg_time*1000:.1f}ms")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Real-time capable (30fps): {'✅' if fps >= 30 else '❌'}")
    
    return avg_time, fps

def main():
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
    
    print("Loading models...")
    
    # DINOv2 모델
    dinov2_model = Dinov2PoseModel(num_keypoints=24, backbone='facebook/dinov2-small')
    dinov2_model.to(device)
    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    
    # FastViT 모델
    fastvit_model = FastVitPoseModel(num_keypoints=24, backbone='fastvit_t8.apple_in1k')
    fastvit_model.to(device)
    fastvit_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')  # Same processor
    
    # 벤치마크 실행
    dinov2_time, dinov2_fps = benchmark_model(dinov2_model, "DINOv2-small", dinov2_processor, device)
    fastvit_time, fastvit_fps = benchmark_model(fastvit_model, "FastViT-T8", fastvit_processor, device)
    
    # 비교 결과
    print(f"\n=== Comparison Results ===")
    print(f"DINOv2-small: {dinov2_time*1000:.1f}ms ({dinov2_fps:.1f} FPS)")
    print(f"FastViT-T8:   {fastvit_time*1000:.1f}ms ({fastvit_fps:.1f} FPS)")
    
    speedup = dinov2_time / fastvit_time
    if speedup > 1:
        print(f"DINOv2 is {speedup:.1f}x faster")
    else:
        print(f"FastViT is {1/speedup:.1f}x faster")
    
    print(f"\nDevice used: {device}")
    print(f"Note: FastViT is optimized for mobile devices with Neural Engine/specialized hardware.")
    print(f"Performance may differ significantly on actual iPhone vs CPU inference.")

if __name__ == "__main__":
    main() 