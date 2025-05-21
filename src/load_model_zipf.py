from fastopic import FASTopic
from topmost import Preprocess
from sklearn.datasets import fetch_20newsgroups
import torch

path = "./tmp/fastopic_whitened.zip"
loaded_model = FASTopic.from_pretrained(path)


beta = loaded_model.get_beta()

doc_topic_dist = loaded_model.transform(docs)


print("First 5 Document Topic Distributions:", doc_topic_dist[:5])