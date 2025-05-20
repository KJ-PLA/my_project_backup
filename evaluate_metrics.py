import json
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# 加载保存的模型输出
def load_model_outputs(topic_words_file="topic_top_words.json", doc_dist_file="doc_topic_dist.npy"):
    with open(topic_words_file, "r") as f:
        topic_words = json.load(f)
    doc_topic_dist = np.load(doc_dist_file)
    return topic_words, doc_topic_dist

# 加载训练结果
topic_top_words, doc_topic_dist = load_model_outputs()

# 加载数据集
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
tokenized_docs = [doc.split() for doc in docs]

# 创建 gensim 字典
dictionary = Dictionary(tokenized_docs)

# **1. 计算 Coherence (C_V)**
top_n_words = 10
top_words_per_topic = [words[:top_n_words] for words in topic_top_words]

coherence_model = CoherenceModel(
    topics=top_words_per_topic,
    texts=tokenized_docs,
    dictionary=dictionary,
    coherence='c_v'
)
coherence_score = coherence_model.get_coherence()

# **2. 计算 Topic Diversity (TD)**
num_topics = len(topic_top_words)
unique_words = set(word for topic in top_words_per_topic for word in topic)
topic_diversity = len(unique_words) / (num_topics * top_n_words)

# **3. 计算 Purity**
true_labels = fetch_20newsgroups(subset='all')['target']
predicted_labels = doc_topic_dist.argmax(axis=1)
contingency_matrix = Counter((true, pred) for true, pred in zip(true_labels, predicted_labels))
purity_score = sum(max(count for (true, pred), count in contingency_matrix.items() if true == t)
                   for t in set(true_labels)) / len(true_labels)

# **4. 计算 NMI (Normalized Mutual Information)**
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)

# **显示结果**
print(f"Coherence (C_V): {coherence_score:.4f}")
print(f"Topic Diversity (TD): {topic_diversity:.4f}")
print(f"Purity: {purity_score:.4f}")
print(f"NMI: {nmi_score:.4f}")
