import torch

if torch.backends.mps.is_available():
    print("✅ MPS (GPU Apple) est bien disponible.")
    x = torch.ones(1, device="mps")
    print(x)
else:
    print("❌ MPS n'est pas disponible. Vérifie ton installation de PyTorch.")