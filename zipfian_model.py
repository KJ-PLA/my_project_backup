import json
import numpy as np
from fastopic import FASTopic
from sklearn.datasets import fetch_20newsgroups
from collections import Counter

# 自定义基于 Zipf's Law 的嵌入模型
class ZipfEmbedModel:
    def __init__(self, vocab_size=10000, alpha=1.2, min_weight=0.001, stopwords=None):
        self.vocab_size = vocab_size
        self.alpha = alpha  # 权重缩放因子
        self.min_weight = min_weight  # 最小权重
        self.stopwords = stopwords if stopwords else set()  # 停用词集合
        self.word_rankings = None

    def fit_vocab(self, corpus):
        """
        根据整个语料库生成词汇排名，符合 Zipf's Law。
        """
        # 统计词频
        all_words = " ".join(corpus).split()
        word_counts = Counter(all_words)
        # 移除停用词
        for stopword in self.stopwords:
            if stopword in word_counts:
                del word_counts[stopword]
        # 按词频排序，并截取前 vocab_size 个单词
        most_common = word_counts.most_common(self.vocab_size)
        self.word_rankings = {word: rank + 1 for rank, (word, _) in enumerate(most_common)}

    def encode(self, docs, show_progress_bar=False, normalize_embeddings=False):
        """
        根据 Zipf's Law 将文档嵌入转化为稀疏向量。
        """
        if not self.word_rankings:
            raise ValueError("Vocabulary not fitted. Call `fit_vocab` first.")

        embeddings = []
        for doc in docs:
            doc_vector = np.zeros(self.vocab_size)
            words = doc.split()
            for word in words:
                if word in self.word_rankings:
                    rank = self.word_rankings[word]
                    # 计算权重，弱化高频词
                    weight = max(1 / (rank ** self.alpha), self.min_weight)
                    doc_vector[rank - 1] += weight  # 累加权重到对应的位置
            embeddings.append(doc_vector)

        embeddings = np.array(embeddings)
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

# 加载数据集
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

# 定义停用词
stopwords = {"the", "is", "in", "and", "to", "of", "a", "for", "on"}

# 创建基于 Zipf's Law 的嵌入模型
zipf_model = ZipfEmbedModel(vocab_size=10000, alpha=1.5, stopwords=stopwords)
zipf_model.fit_vocab(docs)  # 根据语料库生成词汇排名

# 创建并训练 FASTopic 模型，使用 ZipfEmbedModel
model = FASTopic(75, doc_embed_model=zipf_model)
topic_top_words, doc_topic_dist = model.fit_transform(docs)

# 保存模型输出
def save_model_outputs(topic_words, doc_topic_dist, topic_words_file="zipf_topic_top_words.json", doc_dist_file="zipf_doc_topic_dist.npy"):
    with open(topic_words_file, "w") as f:
        json.dump(topic_words, f)
    np.save(doc_dist_file, doc_topic_dist)

# 保存训练结果
save_model_outputs(topic_top_words, doc_topic_dist)

print("Zipfian model training complete and outputs saved.")