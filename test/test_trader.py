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
  
# 测试模型  
def test(model, device, test_loader, criterion):  
    model.eval()  
    test_loss = 0  
    correct = 0  
    y_pred = torch.tensor([], dtype=torch.float32)  
    with torch.no_grad():  
        for inputs, labels in test_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            y_pred = torch.cat((y_pred, outputs.view(-1, 1)), dim=0)    
    test_loss = criterion(y_pred, y_test_tensor).item()  # 累加测试损失     
    print(f'\nTest set: Average loss: {test_loss:.4f}')  
    return y_pred

    
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"
    
# Load data
X_train = np.load(root+"data/"+number+"/X_train.npy", allow_pickle=True)
y_train = np.load(root+"data/"+number+"/y_train.npy", allow_pickle=True)
X_test = np.load(root+"data/"+number+"/X_test.npy", allow_pickle=True)
y_test = np.load(root+"data/"+number+"/y_test.npy", allow_pickle=True)
batch_size=64
  
# 损失函数和优化器（使用Adam优化器并设置weight_decay实现L2正则化效果）  
criterion = nn.MSELoss()  # 假设是回归问题，使用均方误差损失  



y_test_tensor = torch.from_numpy(y_test).to(device,dtype=torch.float32)  
test_dataset = StockDataset(torch.from_numpy(X_test).to(device,dtype=torch.float32), y_test_tensor)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

model = torch.load(root+"data/"+number+"/model.pth")  

# 测试模型  
y_pred = test(model, device, test_loader, criterion)


# %% --------------------------------------- Plot the result  -----------------------------------------------------------------
## TRAIN DATA




class Trader():
    
    def __init__(self): 
        self.trading_fee = 0.001 
        self.cash=5000
        self.shares=0
        self.stock_value=0
        self.buy_price=0
        self.loss=0
    def buy_decision(self,today_close_price,predicted_price,real_price):  
        if predicted_price > today_close_price:  
            # 计算购买100股所需的资金，包括交易费用  
            if self.shares==0:
                cost = today_close_price * 100 * (1 + self.trading_fee)  
                self.cash -= cost  
                self.shares += 100  
                self.buy_price = today_close_price
            print(f"购买100股，当前持股：{self.shares}，剩余资金：{self.cash:.1f} 当前价格:{today_close_price:.1f} 预测价格:{predicted_price:.1f} 真实价格:{real_price:.1f}")  
        else:  
            # 如果是跌，但因为我们没有持股，所以不执行卖出操作  
            print(f"预测价格下跌，但无持股可卖。当前价格:{today_close_price:.1f} 预测价格:{predicted_price:.1f} 真实价格:{real_price:.1f}") 
        self.stock_value = today_close_price * self.shares
    def sell_decision(self, today_close_price,predicted_price,real_price):  
        if predicted_price <= today_close_price and self.shares > 0:  
            # 计算卖出全部股票所得的资金，包括交易费用  
            proceeds = today_close_price * self.shares * (1 - self.trading_fee)  
            self.cash += proceeds  
            self.shares = 0  
            pft = today_close_price - self.buy_price
            if pft<0:
                self.loss+=pft*100
                print(f"亏损每股:{pft}" )
            print(f"卖出全部股票，获得资金：{proceeds:.1f}，当前持股：{self.shares}，剩余资金：{self.cash:.1f} 当前价格:{today_close_price:.1f} 预测价格:{predicted_price:.1f} 真实价格:{real_price:.1f}")  
        elif predicted_price > today_close_price:  
            # 如果是涨，但我们不执行卖出操作，只是持有  
            print(f"预测价格上涨，持有股票。当前价格:{today_close_price:.1f} 预测价格:{predicted_price:.1f} 真实价格:{real_price:.1f}")  
        self.stock_value = today_close_price * self.shares
        
# %% --------------------------------------- Plot the result  -----------------------------------------------------------------

trader = Trader()


def plot_testdataset_result(X_test, y_test,pred_test):

     
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
    print(rescaled_real_y.shape)    
    print(trader.cash)    
    print(trader.loss)    
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

 

test_RMSE = plot_testdataset_result(X_test, y_test,y_pred)
print("----- Test_RMSE_LSTM -----", test_RMSE)