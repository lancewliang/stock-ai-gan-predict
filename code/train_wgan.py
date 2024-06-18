import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
from numpy import *
import numpy as np
from wgan_gp import *

class WGANGPTrainer():
    batch_size = 32
    num_epochs = 100  # 训练轮数  
    lr =0.0001
    
    
    def train(self):
        early_stopping_patience = 10  # 例如，在验证损失连续10个epoch不下降时停止  
        early_stopping_counter = 0  
        best_val_loss = float('inf')  
        for epoch in range(self.num_epochs):  
            for real_data, real_labels, real_seq_labels in self.train_loader:  
                real_data = real_data.to(device)
                # 训练判别器         
                generated_data = self.generator(real_data)
                # print(generated_data[0])
                # print(real_seq_labels[0])
                # print(generated_data.size())
                # print(real_seq_labels.size())
                generated_data_reshape = generated_data.unsqueeze(dim=1)
                fake_output = torch.cat((real_seq_labels, generated_data_reshape), dim=1)
                
                # print(fake_output[0])
                # print(real_labels[0])
                real_y_reshape = real_labels.unsqueeze(dim=1)
                real_output = torch.cat((real_seq_labels ,real_y_reshape), dim=1)  
                
                # print(real_output[0])
                # Get the logits for the fake images
                D_real = self.discriminator(real_output)
                # Get the logits for real images
                D_fake = self.discriminator(fake_output)
                d_loss_real = torch.mean(D_real)
                d_loss_fake = torch.mean(D_fake)   
                gp = gradient_penalty(self.discriminator, real_output, fake_output)
                d_cost =d_loss_fake-d_loss_real
                # d_loss = d_loss_real + d_loss_fake + gp
                d_loss = d_cost + gp
                # 反向传播和优化
                self.discriminator.zero_grad()
                d_loss.backward(retain_graph = True)
                self.optimizer_D.step()
                
                # --------------------
                #  训练生成器
                # --------------------
                generated_data = self.generator(real_data)
                # reshape the data
                generated_data_reshape = generated_data.unsqueeze(dim=1)
                fake_output = torch.cat((real_seq_labels, generated_data_reshape), dim=1)
                # Get the discriminator logits for fake images
                G_fake = self.discriminator(fake_output)
                # Calculate the generator loss
                g_loss =  -torch.mean(G_fake) 
                self.generator.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], D_Loss: {d_loss.item()}, G_Loss: {g_loss.item()}')
            
    
    
    def doProcess(self,number,root):
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

        print("步长：",seq_size)
        print("特征：",feature_size)
        print("输出：",output_size)
 
        self.generator = GRU_Regressor(feature_size, output_size).to(device)
        self.discriminator = StockCNN(seq_size+1).to(device)

        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=self.lr, weight_decay=1e-3)  
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-3)  
 

        # 构建Dataset和DataLoader  
        train_dataset = StockDataset(
                torch.from_numpy(X_train).to(device,dtype=torch.float32),
                torch.from_numpy(y_train).to(device,dtype=torch.float32),
                torch.from_numpy(yc_train).to(device,dtype=torch.float32))  
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  

        self.train()
        
        torch.save(self.generator, root+"data/"+number+"/model_wgan_gp.pth")