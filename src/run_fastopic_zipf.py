from fastopic import FASTopic
from topmost import Preprocess
from sklearn.datasets import fetch_20newsgroups
import torch
from whiten_utils import fit_whitening,whiten

# 1. 加载语料和预处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

preprocess = Preprocess(vocab_size=10000)

# 2. 初始化模型
model = FASTopic(50, preprocess, device=device)
model.fit_transform(docs, epochs=1)

# 3. 加载词频数据
unigram_data = torch.load("src/20ng_unigram.pt")
freq_vocab = unigram_data["vocab"]
freq_tensor = unigram_data["freq"]

# 4. 对 freq 进行对齐（确保和当前模型 vocab 顺序一致）
vocab = model.vocab
V, H = model.word_embeddings.shape
aligned_freq = torch.ones(V)
word_to_index = {word: i for i, word in enumerate(freq_vocab)}

for i, word in enumerate(vocab):
    if word in word_to_index:
        aligned_freq[i] = freq_tensor[word_to_index[word]]
    else:
        aligned_freq[i] = 1.0  # 缺失词设置为 1（或其他平滑）

# 5. 执行 Zipfian Whitening
W = model.model.word_embeddings.detach().cpu()
mu, P = fit_whitening(W, aligned_freq, topk=5000)
W_white = whiten(W, mu, P)

# ✅ 保存替换前嵌入作为 before
before = W.clone()

# 6. 替换 word_embeddings 为白化后的版本
with torch.no_grad():
    model.model.word_embeddings.copy_(W_white.to(device))

# ✅ 获取替换后的嵌入作为 after
after = model.model.word_embeddings.detach().cpu()

# ✅ 计算余弦相似度
import torch.nn.functional as F
cos_sim = F.cosine_similarity(before, after, dim=1)  # shape [V]

print(f"平均余弦相似度: {cos_sim.mean().item():.6f}")
print(f"最小相似度:  {cos_sim.min().item():.6f}")
print(f"最大相似度:  {cos_sim.max().item():.6f}")

# 7. 正式训练（继续训练）
model.fit_transform(docs)

# 8. 保存新模型
path = "./tmp/fastopic_whitened.zip"
model.save(path)