import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np
from vae import VAE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

class VAETrainer():
    num_epochs = 300
    learning_rate = 0.00003
    batch_size = 128
    
    def train(self,train_x):
        feature_size = train_x.shape[1]
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()),self.batch_size, shuffle = False)
        self.model = VAE([feature_size, 400, 400, 400, 10], 10)
                
        self.model = self.model.to(device)   
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        hist = np.zeros( self.num_epochs) 
        for epoch in range( self.num_epochs):
            total_loss = 0
            loss_ = []
            for (x, ) in train_loader:
                x = x.to(device)
                output, z, mu, logVar = self.model(x)
                kl_divergence = 0.5* torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = F.binary_cross_entropy(output, x) + kl_divergence
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
            hist[epoch] = sum(loss_)
            print('[{}/{}] Loss:'.format(epoch+1, self.num_epochs), sum(loss_))

        # plt.figure(figsize=(12, 6))
        # plt.plot(hist)
        
    def doProcess(self,train_x,test_x):
        self.train(train_x)
        self.model.eval()
        _, VAE_train_x, train_x_mu, train_x_var = self.model(torch.from_numpy(train_x).float().to(device))
        _, VAE_test_x, test_x_mu, test_x_var = self.model(torch.from_numpy(test_x).float().to(device))
        n_train_x = np.concatenate((train_x, VAE_train_x.cpu().detach().numpy()), axis = 1)
        n_test_x = np.concatenate((test_x, VAE_test_x.cpu().detach().numpy()), axis = 1)
        return n_train_x,n_test_x
