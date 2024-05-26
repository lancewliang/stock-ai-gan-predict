
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np

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
    
# 定义一个简单的Dataset类  
class StockDataset(Dataset):  
    def __init__(self, X, y):  
        self.X = X  
        self.y = y  
  
    def __len__(self):  
        return len(self.X)  
  
    def __getitem__(self, idx):  
        return self.X[idx], self.y[idx]  
  



# 自定义自注意力层  

class SelfAttention(nn.Module):  

    def __init__(self, embed_size, attention_dim):  
        self.embed_size=embed_size
        self.embedding_dim=embed_size
        self.attention_dim=attention_dim
        super(SelfAttention, self).__init__()  
        self.query = nn.Linear(embed_size, embed_size // 8)  
        self.key = nn.Linear(embed_size, embed_size // 8)  
        self.value = nn.Linear(embed_size, embed_size)  
        self.gamma = nn.Parameter(torch.zeros(1))  
    def forward(self, x):  
        batch_size, seq_len, embed_size = x.size()  
  
        # 计算query, key, value  
        query = self.query(x).view(batch_size, seq_len, -1)  
        key = self.key(x).view(batch_size, seq_len, -1)  
        value = self.value(x).view(batch_size, seq_len, -1)  
  
        # 计算attention分数  
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  
        attention_scores = attention_scores / torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))  
  
        # 应用softmax得到attention权重  
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  
  
        # 应用attention权重到value  
        context = torch.matmul(attention_weights, value)  
  
        # 对序列所有位置的上下文表示取平均，或可以选择其他聚合方法  
        context = context.mean(dim=1)  
  
        return context  


class StockCNN(nn.Module):  
    def __init__(self, input_size):  
        super(StockCNN, self).__init__()  
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, 1, 1, 1),  # [64, 128, 128]
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),      # [64, 64, 64]

            nn.Conv1d(64, 128, 1, 1, 1), # [128, 64, 64]
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),      # [128, 32, 32]

            nn.Conv1d(128, 256, 1, 1, 1), # [256, 32, 32]
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),      # [256, 16, 16]
            
        )
        self.attention = SelfAttention(256,256)  # 假设SelfAttention是一个自定义层
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # print(x.size())
        out = self.cnn(x)
        # print (out.size()) 
        # 展平卷积后的特征图，准备输入到自注意力层  
        out = out.transpose(1, 2)  
        # print (out.size()) 
        out = out.contiguous().view(out.size(0), -1, out.size(2))  
        # print (out.size()) 
        out = self.attention(out)  
        #print (out.size())
        # out = out.view(out.size()[0], -1)
        # print (out.size())
        return self.fc(out)
   
   
   
    