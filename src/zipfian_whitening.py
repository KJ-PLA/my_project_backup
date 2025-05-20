import torch

def fit_zipfian_whitening(
    W: torch.Tensor,
    p: torch.Tensor,
    topk: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    拟合 Zipfian Whitening：计算带频率加权的均值 μ 和白化矩阵 P
    
    Args:
        W: 词向量矩阵，形状 (V, H)
        p: 归一化后 unigram 词频，形状 (V,)
        topk: 如果不为 None，仅用频率最高的 topk 个词拟合
    
    Returns:
        mu: 形状 (H,) 的加权均值向量
        P:  形状 (H, H) 的白化变换矩阵
    """
    V, H = W.shape
    # 1. 如果指定 topk，则选高频词
    if topk is not None and topk < V:
        # 按 p 降序取前 topk
        _, idx = torch.topk(p, topk, largest=True)
        Wk = W[idx]                     # (topk, H)
        pk = p[idx]
        pk = pk / pk.sum()              # 再次归一化
    else:
        Wk = W                          # 全词表
        pk = p / p.sum()                # 确保和为 1

    # 2. 计算加权均值 μ = Σ_j p_j * Wk_j
    mu = (pk.unsqueeze(1) * Wk).sum(dim=0)  # (H,)

    # 3. 中心化
    W_centered = Wk - mu                    # (topk, H) 或 (V, H)

    # 4. 用 √p 加权——等价于在协方差里加权
    Wp = W_centered * torch.sqrt(pk.unsqueeze(1))  # (., H)

    # 5. 对 Wp 做 SVD -> Wp = U Σ Vᵀ
    U, S, Vt = torch.linalg.svd(Wp, full_matrices=False)

    # 6. 构造 Σ^{-1/2}
    S_inv = 1.0 / S                               # (H,)
    # 7. 白化变换矩阵 P = V Σ^{-1/2}
    P = Vt.T @ torch.diag(S_inv)                  # (H, H)

    return mu, P


def whiten(
    X: torch.Tensor,
    mu: torch.Tensor,
    P: torch.Tensor
) -> torch.Tensor:
    """
    对任意同维度向量 X 做 Zipfian Whitening 变换： (X - μ) @ P
    
    Args:
        X: 待白化向量，形状 (..., H)
        mu: 白化均值，形状 (H,)
        P:  白化变换矩阵，形状 (H, H)
    Returns:
        白化后向量，形状 (..., H)
    """
    return (X - mu) @ P


if __name__ == "__main__":
    # 示例：假设词表大小 V=500000，向量维度 H=300
    V, H = 500_000, 300
    # 随机生成示例词向量和词频
    W0 = torch.randn(V, H)
    freq = torch.rand(V)
    p = freq / freq.sum()

    # —— 1. 拟合 μ、P（只用 top-200k 高频词拟合） —— #
    topk = 200_000
    mu, P = fit_zipfian_whitening(W0, p, topk=topk)
    print("Fitted μ shape:", mu.shape)               # (300,)
    print("Fitted P shape:",  P.shape)               # (300, 300)

    # —— 2. 白化整张词表 —— #
    W_whitened = whiten(W0, mu, P)                  # (500000, 300)
    print("Whitened word vectors shape:", W_whitened.shape)

    # —— 3. 白化随机主题向量 —— #
    K = 50
    T_init = torch.randn(K, H)
    T_whitened = whiten(T_init, mu, P)              # (50, 300)
    print("Whitened topic vectors shape:", T_whitened.shape)

    # —— 4. 白化文档向量示例 —— #
    # 假设有 10 篇文档，每篇通过某 encoder 得到 300 维向量
    docs_emb = torch.randn(10, H)
    docs_whitened = whiten(docs_emb, mu, P)         # (10, 300)
    print("Whitened document vectors shape:", docs_whitened.shape)
