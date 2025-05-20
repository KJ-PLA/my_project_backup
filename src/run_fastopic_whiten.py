# src/run_fastopic_whiten.py

import os
import sys
import torch

# —— 确保能 import fastopic 及 utils —— #
ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + "..")
SRC  = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from fastopic import FASTopic
from whiten_utils import fit_whitening, whiten

# 1. 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_unigram(path: str):
    vocab, freq = torch.load(path)
    return freq  # (V,)

class WhitenedDocEncoder:
    """包装 FASTopic 的文档编码器，在输出后白化嵌入"""
    def __init__(self, base_encoder, mu, P):
        self.base = base_encoder
        self.mu   = mu
        self.P    = P

    def encode(self, docs, show_progress_bar=False, normalize_embeddings=False, convert_to_tensor=True):
        # 将 base_encoder 也假设已在 GPU 上
        d = self.base.encode(
            docs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_tensor=convert_to_tensor
        )  # d: (batch, H) on same device as model
        # whiten 会自动在相同 device 上运算
        return whiten(d, self.mu, self.P)

def main():
    # ———— 2. 初始化并移动 FASTopic 到 GPU ————
    model = FASTopic(num_topics=50).to(device)

    # ———— 3. 加载 20NG 词频 ————
    freq = load_unigram("src/20ng_unigram.pt").to(device)  # (V,)

    # ———— 4. 拟合白化 μ, P —— (在 CPU 上也可，但我们直接在 GPU 上做) ————
    W0 = model.word_emb.weight.data             # (V, H)
    mu, P = fit_whitening(W0, freq, topk=200_000)  # 返回 CPU 张量
    # 搬到 GPU
    mu = mu.to(device)
    P  = P.to(device)

    # ———— 5. 白化并替换 词向量 & 主题向量 ———
    # 把原始权重也移动到 device
    W0 = model.word_emb.weight.data.to(device)
    W_white = whiten(W0, mu, P)
    model.word_emb.weight.data.copy_(W_white)

    T0 = model.topic_emb.weight.data.to(device)  # (K, H)
    T_white = whiten(T0, mu, P)
    model.topic_emb.weight.data.copy_(T_white)

    # ———— 6. 包装并移动文档编码器 ———
    # 如果 doc_embed_model 里有 .to，先调用 .to(device)
    try:
        model.doc_embed_model.to(device)
    except AttributeError:
        pass
    # 用 GPU 上的 mu, P 来包装
    model.doc_embed_model = WhitenedDocEncoder(
        model.doc_embed_model, mu, P
    )

    # ———— 7. 准备训练/测试数据 ————
    # 请根据实际需求实现 load_20ng_split
    def load_20ng_split(corpus_root: str, split: str) -> list[str]:
        texts = []
        split_dir = os.path.join(corpus_root, split)
        for category in os.listdir(split_dir):
            cat_dir = os.path.join(split_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                path = os.path.join(cat_dir, fname)
                with open(path, encoding="latin1") as f:
                    texts.append(f.read())
        return texts

    corpus_root = "/data/20news-bydate"  # 修改为你的路径
    train_docs = load_20ng_split(corpus_root, "train")
    test_docs  = load_20ng_split(corpus_root, "test")

    # ———— 8. 训练 & 评测 ———
    model.fit(train_docs, epochs=10)
    print("Top words per topic:")
    for k, words in enumerate(model.get_topic_words(topk=10)):
        print(f"Topic {k:02d}:", words)

    print("Doc-topic distributions on test set:")
    theta = model.get_doc_topic_distribution(test_docs)
    print(theta)

if __name__ == "__main__":
    main()
