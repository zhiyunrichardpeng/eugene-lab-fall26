try:
    import torch
    print("PyTorch is successfully installed.")
except ImportError as e:
    print(f"PyTorch is not installed: {e}")


print("torch.cuda.is_available", torch.cuda.is_available())