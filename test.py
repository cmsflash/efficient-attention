import torch
from factorized_attention import FactorizedAttention


x = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32)
print(FactorizedAttention(1, 2, 2, 2)(x))
