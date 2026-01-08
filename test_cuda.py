import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.rand(2000, 2000, device="cuda")
    y = torch.mm(x, x)
    print("CUDA computation successful")
else:
    print("CUDA NOT working")
