import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  



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
        out2, _ = self.gru2(out1, h0_2)  
  
        # 取最后一个时间步的输出作为全连接层的输入  
        out = out2[:, -1, :]  
  
        # 全连接层  
        out = torch.relu(self.fc1(out))  
        out = torch.relu(self.fc2(out))  
        out = self.fc3(out)  
  
        return out  