import torch

def check_gpu():
    """
    Checks for CUDA GPU availability and prints device information.
    """
    print("--- CUDA GPU Check ---")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA is available!")
        print(f"Number of GPUs: {device_count}")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        current_device = torch.cuda.current_device
        print(f"\nCurrent CUDA device: {current_device}")
        print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
    else:
        print("❌ CUDA is not available. PyTorch is using CPU.")

if __name__ == "__main__":
    check_gpu()
