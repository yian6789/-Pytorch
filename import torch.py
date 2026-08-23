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


#手搓Gengerate
def gengerate (self,token_sequ,max_lenth):#token_sequ为当前输入上下文，max_lenth为生成的最大长度
    for _ in range(max_lenth):
        token_inputs=token_sequ[:,-max_lenth:]#每次都只取最后个max_lenth个token作为输入
        logits,loss=self.forward(token_inputs)#前向传播得到logits
        logits=logits[:,-1,:] #在第二个维度（文本维度）切片得到得到最后一个token的logits（原始分数）
        probs=torch.softmax(logits,dim=-1) #将logits转化为概率分布 （即对应词表中每个token的概率）
        next_token=torch.multinomial(probs,num_samples=1)#logits  ->  softmax  ->  probs  ->  multinomial  ->  next_token
        token_sequ=torch.cat((token_sequ,next_token),dim=1)#最后将生成的新token拼接到原先的token—sequences中，作为下一次的输入上下文
    token_sequ=token_sequ[:,-max_lenth:]#最后更新最新的token—sequence



