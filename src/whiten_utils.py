#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZIPFIAN Whitening 工具：根据 unigram 频率拟合加权白化参数 (mu, P)，
并对任意向量执行 (X - mu) @ P 的白化变换。
"""

import torch
from typing import Optional, Tuple

def fit_whitening(
    W: torch.Tensor,
    freq: torch.Tensor,
    topk: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    拟合 Zipfian-style Whitening 参数。
    Args:
      W:    词嵌入矩阵，形状 (V, H)，已在 CPU/GPU 上
      freq: 对应的词频张量，形状 (V,)
      topk: 若指定，仅用 topk 高频词行拟合
    Returns:
      mu:  加权均值，形状 (H,)
      P:   白化矩阵 Σ^{-1/2}，形状 (H, H)
    """
    device = W.device
    V, H = W.shape

    # 归一化频率
    p = freq.to(device)
    p = p / p.sum()

    # 选取 topk 高频词
    if topk is not None and topk < V:
        _, idx = torch.topk(p, topk, largest=True)
        Wk = W[idx]                             # (topk, H)
        pk = p[idx] / p[idx].sum()              # renormalize
    else:
        Wk, pk = W, p

    # 计算加权均值 μ
    mu = (pk.unsqueeze(1) * Wk).sum(dim=0)      # (H,)

    # 中心化并做 sqrt(p) 加权
    Wc = Wk - mu                                # (., H)
    Wp = Wc * torch.sqrt(pk.unsqueeze(1))       # (., H)

    # SVD 分解 Wp = U diag(S) Vt
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)
    # 构造白化矩阵 P = V diag(1/S)
    P = Vt.T @ torch.diag(1.0 / S)              # (H, H)

    return mu, P

def whiten(
    X: torch.Tensor,
    mu: torch.Tensor,
    P: torch.Tensor
) -> torch.Tensor:
    """
    对任意张量 X 做白化： (X - mu) @ P
    Args:
      X: 形状 (..., H)
      mu: 形状 (H,)
      P:  形状 (H, H)
    Returns:
      形状 (..., H) 的白化结果
    """
    return (X - mu) @ P
