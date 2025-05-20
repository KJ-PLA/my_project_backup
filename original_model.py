import json
import numpy as np
from fastopic import FASTopic
from sklearn.datasets import fetch_20newsgroups
from topmost.preprocessing import Preprocessing

# 加载数据集
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

# 文本预处理
preprocessing = Preprocessing(vocab_size=10000, stopwords='English')

# 创建并训练 FASTopic 模型
model = FASTopic(75, preprocessing)
topic_top_words, doc_topic_dist = model.fit_transform(docs)

# 保存训练结果
def save_model_outputs(topic_words, doc_topic_dist, topic_words_file="topic_top_words.json", doc_dist_file="doc_topic_dist.npy"):
    with open(topic_words_file, "w") as f:
        json.dump(topic_words, f)
    np.save(doc_dist_file, doc_topic_dist)

# 保存模型输出
save_model_outputs(topic_top_words, doc_topic_dist)

print("Model training complete and outputs saved.")
