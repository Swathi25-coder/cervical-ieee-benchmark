import sys
import torch

def main():
    print("=" * 40)
    print("ENVIRONMENT CHECK")
    print("=" * 40)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_avail}")
    if cuda_avail:
        print(f"Device Name    : {torch.cuda.get_device_name(0)}")
        print(f"Device Count   : {torch.cuda.device_count()}")
    else:
        print("Note: Running on CPU. Training models locally will be slow; Google Colab with GPU is recommended for full training.")
    print("=" * 40)

if __name__ == "__main__":
    main()
