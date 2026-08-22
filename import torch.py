import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device=torch.device("cuda"if torch.cuda.is_available()else"cpu")

n_points=100
lr=0.02
epochs=100

# 数据预处理
data = torch.rand(n_points,2)*2-1
data = data.to(device)
labels=(data.norm(dim=1)>0.6).float().unsqueeze(1)
labels = labels.to(device)

#定义类

print(data)
print(labels)

#定义类
class cycleclassfic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(2,30)
        self.layer2=nn.Linear(30,1)

    def forward(self,x):
        x=torch.relu(self.layer1(x))
        x=torch.sigmoid(self.layer2(x))
        return x

model=cycleclassfic().to(device)
predictions=model(data)
loss_fn=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr)

#训练模型
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions=model(data)
    loss=loss_fn(predictions,labels)
    loss.backward()
    optimizer.step()#一步一步进行
    if epoch%20==0:
        print(f"Epoch {epoch},Loss:{loss.item()}")

if device.type == 'cuda':
    print("使用了显卡")
else:
    print("使用了cpu")
#可视化结果  
predictions=model(data)
predictions=predictions.cpu()
data=data.cpu()
plt.scatter(data[:,0],data[:,1],c=(predictions.squeeze()>0.6).float(),cmap='coolwarm')
plt.title('Classification Result')   
plt.xlabel("x")
plt.ylabel("y")
plt.show()  

    



