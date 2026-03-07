import torch
from models.MLP import MLP

# Test 1: ReLU + BN
m1 = MLP(784, [256, 128], 10, activation="relu", use_batchnorm=True)
print("=== ReLU + BN ===")
print(m1)

# Test 2: GELU + no BN
m2 = MLP(784, [512], 10, activation="gelu", use_batchnorm=False)
print("\n=== GELU + no BN ===")
print(m2)

# Test 3: forward pass
x = torch.randn(4, 1, 28, 28)
print(f"\nOutput shape: {m1(x).shape}")  # torch.Size([4, 10])