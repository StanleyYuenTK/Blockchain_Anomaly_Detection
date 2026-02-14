import torch

print("PyTorch version:     ", torch.__version__)
print("CUDA available:      ", torch.cuda.is_available())
print("CUDA version in PyTorch:", torch.version.cuda)          # ← this is what you want
print("cuDNN version:       ", torch.backends.cudnn.version())  # optional

# Bonus — GPU info
if torch.cuda.is_available():
    print("GPU count:           ", torch.cuda.device_count())
    print("Current GPU name:    ", torch.cuda.get_device_name(0))
else:
    print("No GPU detected / CUDA not available")