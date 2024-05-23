import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np
from torch.utils.data import Dataset, DataLoader  

class GRU_Regressor(nn.Module):  
    def __init__(self, input_size, output_size):  
        super(GRU_Regressor, self).__init__()  
        
        self.gru1 = nn.GRU(input_size, 256, batch_first=True, dropout=0.02, bidirectional=False)  
        self.gru2 = nn.GRU(256, 128, dropout=0.02, bidirectional=False)  
          
        self.fc1 = nn.Linear(128, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, output_size)  
  
    def forward(self, x):  
        # 初始化隐藏状态  
        h0_1 = torch.zeros(1, x.size(0), 256).to(x.device)  
        h0_2 = torch.zeros(1, x.size(0), 128).to(x.device)  
  
        # GRU层  
        out1, _ = self.gru1(x, h0_1)  
        print(out1.size())
        out2, _ = self.gru2(out1, h0_2)  
        print(out2.size())
        # 取最后一个时间步的输出作为全连接层的输入  
        out = out2[:, -1, :]  
        print(out.size())
        # 全连接层  
        out = torch.relu(self.fc1(out))  
        out = torch.relu(self.fc2(out))  
        print(out.size())
        out = self.fc3(out)  
        print(out.size())
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

input_size = X_train.shape[1]
feature_size = X_train.shape[2]
output_size = y_train.shape[1]

batch_size=64
num_epochs = 2000  # 训练轮数  

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
train_dataset = StockDataset(X_train, y_train)  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
  
test_dataset = StockDataset(X_test, y_test)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  


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
  
# 测试模型  
def test(model, device, test_loader, criterion):  
    model.eval()  
    test_loss = 0  
    correct = 0  
    with torch.no_grad():  
        for inputs, labels in test_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            test_loss += criterion(outputs, labels).item()  # 累加测试损失  
            _, predicted = torch.max(outputs.data, 1)  # 获取最大值的索引（即预测的类别）  
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数  
  
    test_loss /= len(test_loader.dataset)  # 计算平均测试损失  
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100 * correct / len(test_loader.dataset):.2f}%\n')  

model = model.to(device)  
  
# 训练模型  
train(model, device, train_loader, optimizer, criterion, num_epochs)  
  
# 测试模型  
#test(model, device, test_loader, criterion)