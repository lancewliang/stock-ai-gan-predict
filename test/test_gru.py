import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
from sklearn.metrics import mean_squared_error
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np
from torch.utils.data import Dataset, DataLoader  
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
class GRU_Regressor(nn.Module):  
    def __init__(self, input_size, output_size):  
        super(GRU_Regressor, self).__init__()  
        
        
         # 第一层GRU  
        self.gru1 = nn.GRU(input_size, 256, num_layers=2, batch_first=True, dropout=0.02)  
        # # 第二层GRU  
        # self.gru2 = nn.GRU(256, 128, batch_first=True, dropout=0.02)  
        # 全连接层进行最终预测  
          
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, output_size)  
  
    def forward(self, x):  
        isdebug =False
        # 初始化隐藏状态  
        h0_1 = torch.zeros(2, x.size(0), 256).to(x.device)  
        # h0_2 = torch.zeros(1, x.size(0), 128).to(x.device)  
        # print(x.size())
        # GRU层  
        out1, _ = self.gru1(x, h0_1)  
        #print(out1.size())
         # 只取最后一个时间步的输出作为第二层GRU的输入（假设我们只需要最后一个时间步的信息）  
        out1 = out1[:, -1, :]  
        #print(out1.size())
        # 前向传播第二层GRU  
        # out2, _ = self.gru2(out1.unsqueeze(1), h0_2)  # unsqueeze增加时间步维度，因为我们只有一个时间步  
        # # 同样只取最后一个时间步的输出（在这个例子中其实只有一个时间步）  
        # # print(out2.size())
        # out2 = out2.squeeze(1)  
      
        out=out1
        # 全连接层  
        out = torch.relu(self.fc1(out))  
        out = torch.relu(self.fc2(out))  
        # print(out.size())
        out = self.fc3(out)  
        #print(out.size())
        return out  
    
    
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"
    
# Load data
X_train = np.load(root+"data/"+number+"/X_train.npy", allow_pickle=True)
y_train = np.load(root+"data/"+number+"/y_train.npy", allow_pickle=True)
X_test = np.load(root+"data/"+number+"/X_test.npy", allow_pickle=True)
y_test = np.load(root+"data/"+number+"/y_test.npy", allow_pickle=True)



print(X_train.shape)
print(y_train.shape)

seq_size = X_train.shape[1]
feature_size = X_train.shape[2]
output_size = y_train.shape[1]

batch_size=64
num_epochs = 200  # 训练轮数  

model = GRU_Regressor(feature_size, output_size)  
  
# 损失函数和优化器（使用Adam优化器并设置weight_decay实现L2正则化效果）  
criterion = nn.MSELoss()  # 假设是回归问题，使用均方误差损失  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  

# 定义一个简单的Dataset类  
class StockDataset(Dataset):  
    def __init__(self, X, y):  
        self.X = X  
        self.y = y  
  
    def __len__(self):  
        return len(self.X)  
  
    def __getitem__(self, idx):  
        return self.X[idx], self.y[idx]  
  

# 构建Dataset和DataLoader  
train_dataset = StockDataset(torch.from_numpy(X_train).to(device,dtype=torch.float32), torch.from_numpy(y_train).to(device,dtype=torch.float32))  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  




# 训练模型  
def train(model, device, train_loader, optimizer, criterion, num_epochs=25):  
    model.train()  
    for epoch in range(num_epochs):  
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
              
            # 梯度清零  
            optimizer.zero_grad()  
              
            # 正向传播  
            outputs = model(inputs)  
            # print(outputs.size())
            # print(labels.size())
            # print(outputs)
            # print(labels)
            
            # 计算损失  
            loss = criterion(outputs, labels)  
              
            # 反向传播和优化  
            loss.backward()  
            optimizer.step()  
          
        if (epoch+1) % 5 == 0:  
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  
  
model = model.to(device)  
  
# 训练模型  
train(model, device, train_loader, optimizer, criterion, num_epochs)  
torch.save(model.state_dict(), root+"data/"+number+"/model.ckpt")
torch.save(model, root+"data/"+number+"/model.pth")

