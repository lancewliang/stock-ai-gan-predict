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
from wgan_gp import *

 


class Trader():
    
    def __init__(self): 
        self.trading_fee = 0.001 
        self.cash=5000
        self.trading_size =500
        self.shares=0
        self.stock_value=0
        self.buy_price=0
        self.loss=0
    def buy_decision(self,today_close_price,predicted_price,real_price):  
        if predicted_price > today_close_price:  
            # 计算购买100股所需的资金，包括交易费用  
            if self.shares==0:
                cost = today_close_price * self.trading_size * (1 + self.trading_fee)  
                self.cash -= cost  
                self.shares += self.trading_size  
                self.buy_price = today_close_price
            print(f"购买100股，当前持股：{self.shares}，剩余资金：{self.cash:.2f} 当前价格:{today_close_price:.2f} 预测价格:{predicted_price:.2f} 真实价格:{real_price:.2f}")  
        else:  
            # 如果是跌，但因为我们没有持股，所以不执行卖出操作  
            print(f"预测价格下跌，但无持股可卖。当前价格:{today_close_price:.2f} 预测价格:{predicted_price:.2f} 真实价格:{real_price:.2f}") 
        self.stock_value = today_close_price * self.shares
    def sell_decision(self, today_close_price,predicted_price,real_price):  
        if predicted_price <= today_close_price and self.shares > 0:  
            # 计算卖出全部股票所得的资金，包括交易费用  
            proceeds = today_close_price * self.shares * (1 - self.trading_fee)  
            self.cash += proceeds  
            self.shares = 0  
            pft = today_close_price - self.buy_price
            if pft<0:
                self.loss+=pft*self.trading_size
                print(f"亏损每股:{pft}" )
            print(f"卖出全部股票，获得资金：{proceeds:.2f}，当前持股：{self.shares}，剩余资金：{self.cash:.2f} 当前价格:{today_close_price:.2f} 预测价格:{predicted_price:.2f} 真实价格:{real_price:.2f}")  
        elif predicted_price > today_close_price:  
            # 如果是涨，但我们不执行卖出操作，只是持有  
            print(f"预测价格上涨，持有股票。当前价格:{today_close_price:.2f} 预测价格:{predicted_price:.2f} 真实价格:{real_price:.2f}")  
        self.stock_value = today_close_price * self.shares
        
# %% --------------------------------------- Plot the result  -----------------------------------------------------------------



class TraderTest():
    batch_size=32
    trader = Trader()
    # 测试模型  
    def predTest(self,model, device, test_loader, criterion):  
        model.eval()  
        test_loss = 0  
        correct = 0  
        y_pred = torch.tensor([], dtype=torch.float32)  
        with torch.no_grad():  
            for inputs, labels,yc in test_loader:  
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)  
                y_pred = torch.cat((y_pred, outputs.view(-1, 1)), dim=0)    
        # test_loss = criterion(y_pred, y_test_tensor).item()  # 累加测试损失     
        # print(f'\nTest set: Average loss: {test_loss:.4f}')  
        return y_pred


    def plot_testdataset_result(self,root,number,trader,X_test, y_test,pred_test):

        
        y_scaler = load(open(root+"data/"+number+"/y_scaler.pkl", 'rb'), allow_pickle=True)
        test_predict_index = np.load(root+"data/"+number+"/index_test.npy", allow_pickle=True)

        rescaled_real_y = y_scaler.inverse_transform(y_test)
        rescaled_predicted_y = y_scaler.inverse_transform(pred_test)
        print(rescaled_real_y.shape)
        print(rescaled_predicted_y.shape)
        cashs_np = []
        for i in range(rescaled_real_y.shape[0]):
            if i==0:
                cashs_np.append(trader.cash-5000)
                continue
                
            today_price = rescaled_real_y[i-1][0]
            predicted_price =rescaled_predicted_y[i][0]
            real_price = rescaled_real_y[i][0]
            trader.sell_decision(today_price,predicted_price,real_price)
            trader.buy_decision(today_price,predicted_price,real_price)
            cashs_np.append(trader.cash+trader.stock_value-5000)
        print("形状",rescaled_real_y.shape)    
        print("现金",trader.cash)    
        print("损失",trader.loss)    
        print("总资产",trader.cash+trader.stock_value)    
        arrcachs = np.array(cashs_np).reshape((rescaled_real_y.shape[0], 1))
        # Plot the predicted result
        plt.figure(figsize=(16, 8))
        plt.plot(rescaled_real_y)
        plt.plot(rescaled_predicted_y, color='r')
        

        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("real", "predicted","cachs"), loc="upper left", fontsize=16)
        plt.title("The result of Testing", fontsize=20)
        plt.show()
        
        
        plt.figure(figsize=(16, 8))
        plt.plot(arrcachs) 
        

        plt.xlabel("Date")
        plt.ylabel("cachs")
        plt.legend(("cachs"), loc="upper left", fontsize=16)
        plt.title("The result of Testing", fontsize=20)
        plt.show()
        # Calculate RMSE
        # predicted = predict_result["predicted_mean"]
        # real = real_price["real_mean"]
        RMSE = np.sqrt(mean_squared_error(rescaled_predicted_y, rescaled_real_y))
        #print('-- Test RMSE -- ', RMSE)

        return RMSE

    def doProcess(self,number,root):
        # Load data
        X_train = np.load(root+"data/"+number+"/X_train.npy", allow_pickle=True)
        y_train = np.load(root+"data/"+number+"/y_train.npy", allow_pickle=True)
        X_test = np.load(root+"data/"+number+"/X_test.npy", allow_pickle=True)
        y_test = np.load(root+"data/"+number+"/y_test.npy", allow_pickle=True)
        yc_test = np.load(root+"data/"+number+"/yc_test.npy", allow_pickle=True)

  
        # 损失函数和优化器（使用Adam优化器并设置weight_decay实现L2正则化效果）  
        criterion = nn.MSELoss()  # 假设是回归问题，使用均方误差损失  

        y_test_tensor = torch.from_numpy(y_test).to(device,dtype=torch.float32)  
        yc_test_tensor= torch.from_numpy(yc_test).to(device,dtype=torch.float32)  
        test_dataset = StockDataset(torch.from_numpy(X_test).to(device,dtype=torch.float32), y_test_tensor,yc_test_tensor)  
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  

        model = torch.load(root+"data/"+number+"/model_wgan_gp.pth")  
        # 测试模型  
        y_pred = self.predTest(model, device, test_loader, criterion)

        
        test_RMSE = self.plot_testdataset_result(root,number,self.trader,X_test, y_test,y_pred)
        print("----- Test_RMSE_LSTM -----", test_RMSE)
