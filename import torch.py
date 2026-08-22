import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 数据预处理
n_points=10
data = torch.rand(n_points,2)*2-1
labels=(data.norm(dim=1)>0.6).float().unsqueeze(1)

#定义类


print(data)
print(labels)