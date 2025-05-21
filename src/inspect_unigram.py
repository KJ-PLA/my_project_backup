#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def main():
    # 1. 加载词表和频率张量
    data = torch.load("src/20ng_unigram.pt")
    vocab, freq = data["vocab"], data["freq"]  # vocab: list[str], freq: Tensor[V]

    # 2. 打印前 10 条原始词表项（注意不是最高频的）
    print("前 10 条原始词表及其频次：")
    for word, count in zip(vocab[:10], freq[:10].tolist()):
        print(f"  {word!r}: {count}")

    # 3. 找出频次最高的前 10 个词
    topk = 10
    # torch.topk 返回 (values, indices)
    top_counts, top_indices = torch.topk(freq, k=topk, largest=True)
    top_words = [vocab[i] for i in top_indices.tolist()]

    print("\n出现频次最高的 10 个词：")
    for word, count in zip(top_words, top_counts.tolist()):
        print(f"  {word!r}: {count}")

    # 4. 统计信息
    print("\n统计信息：")
    print(f"  词表总长度: {len(vocab)}")
    print(f"  最高频次: {freq.max().item()}")
    print(f"  最低频次: {freq.min().item()}")
    print(f"  总 token 数: {freq.sum().item():.0f}")

if __name__ == "__main__":
    main()
