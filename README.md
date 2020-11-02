# Efficient Attention

An implementation of the [efficient attention](https://arxiv.org/abs/1812.01243) module.

## Description

![](illustration.png)

Efficient attention is an attention mechanism that substantially optimizes the memory and computational efficiency while retaining **exactly** the same expressive power as the conventional dot-product attention. The illustration above compares the two types of attention. The efficient attention module is a drop-in replacement for the non-local module ([Wang et al., 2018](https://arxiv.org/abs/1711.07971)), while it:

- uses less resources to achieve the same accuracy;
- achieves higher accuracy with the same resource constraints (by allowing more insertions); and
- is applicable in domains and models where the non-local module is not (due to resource constraints).

## Implementation details

This repository implements the efficient attention module with softmax normalization, output reprojection, and residual connection.

## Features not in the paper

This repository implements additionally implements the multi-head mechanism which was not in the paper. To learn more about the mechanism, refer to [Vaswani et al.](https://arxiv.org/abs/1706.03762)

## Citation

The [paper](https://arxiv.org/abs/1812.01243) will appear at WACV 2021. If you use, compare with, or refer to this work, please cite

```bibtex
@inproceedings{shen2021efficient,
  author    = {Zhuoran Shen and
               Mingyuan Zhang and
               Haiyu Zhao and
               Shuai Yi and
               Hongsheng Li},
  title     = {Efficient Attention: Attention with Linear Complexities},
  booktitle = {WACV},
  year      = {2021},
}
```
