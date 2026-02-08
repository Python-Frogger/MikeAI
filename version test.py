import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is GPU available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# This part tells us the ROCm/HIP version specifically
if hasattr(torch.version, 'hip'):
    print(f"HIP (ROCm) version: {torch.version.hip}")
elif hasattr(torch.version, 'cuda'):
    print(f"Software alias version: {torch.version.cuda}")