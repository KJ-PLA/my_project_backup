#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端：加载词频 → 拟合白化 → 应用白化 → 训练 FASTopic → 保存/加载模型 → 评测。
运行方式（在项目根目录下）：
    python src/run_fastopic_whiten.py
"""

import os
import sys
import torch

# 确保能 import 根目录下的 fastopic 包和当前 src
ROOT = os.path.abspath(os.path.dirname(__file__) + os.sep + "..")
SRC  = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from fastopic import FASTopic
from whiten_utils import fit_whitening, whiten
from sklearn.datasets import fetch_20newsgroups

# 1. 选择设备（自动检测 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 2. 加载 unigram 词表和频率
def load_unigram(path: str):
    data = torch.load(path)
    return data["vocab"], data["freq"]

# 3. 从本地文件夹加载 20NG train/test 文档列表
def load_20ng_split(corpus_root: str, split: str) -> list[str]:
    texts = []
    split_dir = os.path.join(corpus_root, split)
    for cat in os.listdir(split_dir):
        cat_dir = os.path.join(split_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            path = os.path.join(cat_dir, fname)
            with open(path, encoding="latin1") as f:
                texts.append(f.read())
    return texts

def main():
    # 路径设置：根据本地实际修改 corpus_root
    unigram_path = "src/20ng_unigram.pt"
    corpus_root  = "data/20news-bydate"  # 请改成你本地解压路径

    # 4. 初始化 FASTopic 并搬到设备
    model = FASTopic(num_topics=50).to(device)

    # 5. 加载并归一化频率，搬到设备
    vocab, freq = load_unigram(unigram_path)
    freq = freq.to(device)

    # 6. 拟合白化参数 (仅基于词嵌入)
    W0 = model.word_emb.weight.data.to(device)    # 词嵌入 (V, H)
    mu, P = fit_whitening(W0, freq, topk=200_000)
    mu, P = mu.to(device), P.to(device)

    # 7. 白化并替换 词嵌入 & 主题嵌入
    W_white = whiten(W0, mu, P)
    model.word_emb.weight.data.copy_(W_white)

    T0 = model.topic_emb.weight.data.to(device)   # 主题向量 (K, H)
    T_white = whiten(T0, mu, P)
    model.topic_emb.weight.data.copy_(T_white)

    # 8. 包装文档编码器，使其输出也做白化
    class WhitenedDocEncoder:
        def __init__(self, base, mu, P):
            self.base = base
            self.mu   = mu
            self.P    = P
        def encode(self, docs, **kw):
            d = self.base.encode(docs, **kw)    # (B, H)
            return whiten(d, self.mu, self.P)

    # 将编码器搬到设备（若支持 .to() 方法）
    try:
        model.doc_embed_model.to(device)
    except:
        pass
    model.doc_embed_model = WhitenedDocEncoder(model.doc_embed_model, mu, P)

    # 9. 加载 20NG train/test 文档
    train_docs = load_20ng_split(corpus_root, "train")
    test_docs  = load_20ng_split(corpus_root, "test")

    # 10. 训练 & 保存模型
    print("开始训练 FASTopic …")
    model.fit(train_docs, epochs=10)

    save_path = "fastopic_whitened.zip"
    print("保存模型到", save_path)
    model.save(save_path)

    # 11. 从磁盘加载并评测
    print("加载并评测已保存模型 …")
    loaded = FASTopic.from_pretrained(save_path).to(device)
    # 获取主题-词分布 β
    beta = loaded.get_beta()
    print("β 矩阵形状:", beta.shape)

    # 新文档推断
    doc_topic = loaded.transform(test_docs[:5])
    print("前 5 篇测试文档的主题分布:\n", doc_topic)

    # 继续微调 / 增量训练
    print("继续训练 1 轮 …")
    loaded.fit_transform(train_docs, epochs=1)

if __name__ == "__main__":
    main()
