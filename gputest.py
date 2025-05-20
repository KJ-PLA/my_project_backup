import torch
print(torch.cuda.is_available())   # True = 可用 GPU
print(torch.cuda.device_count())   # 可用 GPU 数量
print(torch.cuda.get_device_name(0))  # 第一个 GPU 名字
