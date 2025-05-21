#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 20 Newsgroups 数据集的 unigram 词频，并保存到 src/20ng_unigram.pt。
运行方式（在项目根目录下）：
    python src/build_unigram_20ng.py
"""

from collections import Counter
import torch
from sklearn.datasets import fetch_20newsgroups
from typing import List

def simple_tokenize(text: str) -> List[str]:
    """
    最简单的分词：全部小写后按空白切分。
    """
    return text.lower().split()

def main():
    print("1) 加载 20NG 数据集（去除 headers/footers/quotes）")
    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )['data']

    print("2) 分词并统计频次")
    counter = Counter()
    for doc in data:
        tokens = simple_tokenize(doc)
        counter.update(tokens)

    print("3) 构造词表和频率张量")
    vocab, freq = zip(*counter.items())
    freq_tensor = torch.tensor(freq, dtype=torch.float32)

    save_path = "src/20ng_unigram.pt"
    print(f"4) 保存到 {save_path}")
    torch.save({"vocab": list(vocab), "freq": freq_tensor}, save_path)
    print(f"完成，共 {len(vocab)} 个词。")

if __name__ == "__main__":
    main()
