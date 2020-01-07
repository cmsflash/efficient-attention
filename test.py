import torch
from efficient_attention import EfficientAttention


x = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32)
print(EfficientAttention(1, 2, 2, 2)(x))
