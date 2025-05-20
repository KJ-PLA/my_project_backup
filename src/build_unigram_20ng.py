# src/build_unigram_20ng.py
import os
from collections import Counter
import torch

def load_20ng(root_dir):
    docs = []
    for split in ("train", "test"):
        split_dir = os.path.join(root_dir, split)
        for fname in os.listdir(split_dir):
            path = os.path.join(split_dir, fname)
            with open(path, encoding="latin1") as f:
                docs.append(f.read())
    return docs

def simple_tokenize(text):
    return text.lower().split()

if __name__ == "__main__":
    # TODO: 改成你本地 20NG 数据集的路径
    corpus_root = "/path/to/20news-bydate"
    docs = load_20ng(corpus_root)
    cnt = Counter()
    for doc in docs:
        cnt.update(simple_tokenize(doc))
    vocab, freq = zip(*cnt.items())
    freq = torch.tensor(freq, dtype=torch.float32)
    # 保存到 src/20ng_unigram.pt
    torch.save((list(vocab), freq), "src/20ng_unigram.pt")
    print("Saved unigram to src/20ng_unigram.pt")
