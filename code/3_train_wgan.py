import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np
from wgan_gp import *
    
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"
    
# Load data
X_train = np.load(root+"data/"+number+"/X_train.npy", allow_pickle=True)
y_train = np.load(root+"data/"+number+"/y_train.npy", allow_pickle=True)
yc_train = np.load(root+"data/"+number+"/yc_train.npy", allow_pickle=True)
X_test = np.load(root+"data/"+number+"/X_test.npy", allow_pickle=True)
y_test = np.load(root+"data/"+number+"/y_test.npy", allow_pickle=True)
yc_test = np.load(root+"data/"+number+"/yc_test.npy", allow_pickle=True)

seq_size = X_train.shape[1]
feature_size = X_train.shape[2]
output_size = y_train.shape[1]


batch_size=64
num_epochs = 200  # 训练轮数  

generator = GRU_Regressor(feature_size, output_size)  
discriminator = StockCNN(seq_size)
criterion = nn.MSELoss()  # 假设是回归问题，使用均方误差损失  
g_optimizer = optim.Adam(generator.parameters(), lr=0.001, weight_decay=1e-3)  
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001, weight_decay=1e-3)  


def train():