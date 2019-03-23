import torch
from torch import nn
from torch.nn import functional as f


class DecomposedAttention(nn.Module):
    
    def __init__(
            self, in_channels, basis_cardinality, head_count, value_channels
        ):
        super().__init__()
        self.in_channels = in_channels
        self.basis_cardinality = basis_cardinality
        self.head_count = head_count
        self.value_channels = value_channels

        self.bases = nn.Conv2d(in_channels, basis_cardinality, 1)
        self.coefficient_sets = nn.Conv2d(in_channels, basis_cardinality, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        bases = self.bases(input_).reshape((n, self.basis_cardinality, h * w))
        coefficient_sets = self.coefficient_sets(input_).reshape(
            n, self.basis_cardinality, h * w
        )
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_basis_cardinality = self.basis_cardinality // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            basis = f.softmax(bases[
                :,
                i * head_basis_cardinality: (i + 1) * head_basis_cardinality,
                :
            ], dim=2)
            coefficient_set = f.softmax(coefficient_sets[
                :,
                i * head_basis_cardinality: (i + 1) * head_basis_cardinality,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = basis @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ coefficient_set
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention
