import torch
import sys

def main():
    print("=== System Info ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print("\n=== CUDA Verification ===")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device Index: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / (1024 ** 2):.2f} MB")
        
        # Test basic tensor operation on CUDA
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print("Tensor operation test on CUDA: SUCCESS")
        except Exception as e:
            print(f"Tensor operation test on CUDA: FAILED with error: {e}")
    else:
        print("CUDA is NOT available. Running on CPU instead.")

if __name__ == "__main__":
    main()
