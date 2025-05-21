from fastopic import FASTopic
from topmost import Preprocess
from sklearn.datasets import fetch_20newsgroups
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

preprocess = Preprocess(vocab_size=10000)

model = FASTopic(50, preprocess, device=device)
top_words, doc_topic_dist = model.fit_transform(docs)

path = "./tmp/fastopic.zip"
model.save(path)

print("Topic Top Words:", top_words)
print("First 5 Document Topic Distributions:", doc_topic_dist[:5])
