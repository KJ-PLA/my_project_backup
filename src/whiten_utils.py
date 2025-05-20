# src/whiten_utils.py
import torch

def fit_whitening(
    W: torch.Tensor,
    freq: torch.Tensor,
    topk: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    用前 topk 高频词拟合 Zipfian‐style 白化，
    返回 (mu, P)，分别是 (H,) 和 (H,H)
    """
    V, H = W.shape
    # 1) 归一化词频
    p = freq / freq.sum()

    # 2) 选 topk 高频词拟合（可选）
    if topk is not None and topk < V:
        _, idx = torch.topk(p, topk, largest=True)
        Wk = W[idx]                              # (topk, H)
        pk = p[idx] / p[idx].sum()              # renormalize
    else:
        Wk, pk = W, p

    # 3) 计算带权均值 μ
    mu = (pk.unsqueeze(1) * Wk).sum(dim=0)     # (H,)

    # 4) 中心化并做 √p 加权
    Wc = Wk - mu                               # (., H)
    Wp = Wc * torch.sqrt(pk.unsqueeze(1))      # (., H)

    # 5) SVD -> Wp = U diag(S) Vt
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)

    # 6) 构造白化矩阵 P = V diag(1/S)
    P = Vt.T @ torch.diag(1.0 / S)             # (H, H)

    return mu, P

def whiten(
    X: torch.Tensor,
    mu: torch.Tensor,
    P: torch.Tensor
) -> torch.Tensor:
    """
    对任意同维度向量 X 做白化：(X - μ) @ P
    X shape is (..., H)
    """
    return (X - mu) @ P
