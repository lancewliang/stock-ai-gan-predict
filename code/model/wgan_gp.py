
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
        
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        
        # 全连接层进行最终预测  
          
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, output_size) 
        self.dropout = nn.Dropout(0.02) 
  
    def forward(self, x):  
        isdebug =False
        # 初始化隐藏状态  
        
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        
        
 
        #print(out1.size())
         # 只取最后一个时间步的输出作为第二层GRU的输入（假设我们只需要最后一个时间步的信息）  
        out4 = out_3[:, -1, :]  
        #print(out1.size())
        # 前向传播第二层GRU  
        # out2, _ = self.gru2(out1.unsqueeze(1), h0_2)  # unsqueeze增加时间步维度，因为我们只有一个时间步  
        # # 同样只取最后一个时间步的输出（在这个例子中其实只有一个时间步）  
        # # print(out2.size())
        # out2 = out2.squeeze(1)  
      
        out=out4
        # 全连接层  
        out = torch.relu(self.fc1(out))  
        out = torch.relu(self.fc2(out))  
        # print(out.size())
        out = self.fc3(out)  
        #print(out.size())
        return out  
    
# 定义一个简单的Dataset类  
class StockDataset(Dataset):  
    def __init__(self, X, y, yc):  
        self.X = X  
        self.y = y    
        self.yc = yc  
    def __len__(self):  
        return len(self.X)  
  
    def __getitem__(self, idx):  
        return self.X[idx], self.y[idx] , self.yc[idx]  
  



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
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2, 0),      # [64, 64, 64]

            nn.Conv1d(64, 128, 1, 1, 1), # [128, 64, 64]
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2, 0),      # [128, 32, 32]

            nn.Conv1d(128, 256, 1, 1, 1), # [256, 32, 32]
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2, 0),      # [256, 16, 16]
            
        )
        self.attention = SelfAttention(256,256)  # 假设SelfAttention是一个自定义层
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
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
        #out = self.attention(out)  
        #print (out.size())
        # out = out.view(out.size()[0], -1)
        # print (out.size())
        return self.fc(out)
    
    
from torch.autograd import grad  

# def gradient_penalty(D, batch_size, real_output, fake_output):
#         """ Calculates the gradient penalty.

#         This loss is calculated on an interpolated image
#         and added to the discriminator loss.
#         """
#         # get the interpolated data
#         alpha = torch.rand(real_output.size(0), real_output.size(1), 1).to(real_samples.device)  
    
#         diff = fake_output - real_output
#         interpolated = real_output + alpha * diff

#         interpolates = interpolates.requires_grad_(True)  
#             # 计算判别器的输出  
#         pred = D(interpolates) 

#         gradients = grad(outputs=pred, inputs=interpolates,  
#                      grad_outputs=torch.ones_like(pred),  
#                      create_graph=True, retain_graph=True, only_inputs=True)[0]  
 
#         # 3. Calcuate the norm of the gradients
#         gradients = gradients.view(gradients.size(0), -1)  
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  

#         return gradient_penalty   


def gradient_penalty(D, real_samples, fake_samples, eps=1e-10):  
    # print(real_samples.size())
    # print(fake_samples.size())
    # 计算随机权重  
    alpha = torch.rand(real_samples.size(0), real_samples.size(1), 1).to(real_samples.device)  
    alpha = alpha.expand_as(real_samples)  
    # 插值  
    # diff = fake_samples - real_samples
    # interpolates = real_samples + alpha * diff
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples.detach()  
    interpolates = interpolates.requires_grad_(True)  
    # 计算判别器的输出  
    d_interpolates = D(interpolates)  

    # 计算关于插值样本的梯度  
    gradients = grad(outputs=d_interpolates, inputs=interpolates,  
                     grad_outputs=torch.ones_like(d_interpolates),  
                     create_graph=True, retain_graph=True, only_inputs=True)[0]  

    # 计算梯度范数并应用惩罚  
    gradients = gradients.view(gradients.size(0), -1)  
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  

    return gradient_penalty   
  
    