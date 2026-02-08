import torch


def verify_mikeai_rig():
    print("--- MIKEAI HARDWARE VERIFICATION ---")

    # 1. Is PyTorch seeing the ROCm backend?
    gpu_available = torch.cuda.is_available()
    print(f"ROCm/GPU Detected: {gpu_available}")

    if gpu_available:
        # 2. Identify the beast
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using Graphics Card: {gpu_name}")

        # 3. Check memory (Should see ~16GB)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM Available: {total_mem:.2f} GB")

        # 4. Math Test (Forces a tiny calculation on the GPU)
        x = torch.tensor([1.0, 2.0, 3.0]).to('cuda')
        y = x * 2
        print(f"GPU Math Test: {y.cpu().numpy()} (Success if [2, 4, 6])")
    else:
        print("ERROR: GPU not detected. Check if your AMD Adrenalin 26.1.1 drivers are active.")


if __name__ == "__main__":
    verify_mikeai_rig()