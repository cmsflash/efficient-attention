import torch
from decomposed_attention import DecomposedAttention


x = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32)
print(DecomposedAttention(1, 2, 2, 2)(x))
