#https://blog.csdn.net/pan_1214_/article/details/131868536
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
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


class Position():
    sign=""
    date = ""
    price=0
    size=0
    
    
class TurtleTradingStrategy():
    batch_size = 32
    def init(self,x,rescaled_real_y,rescaled_predicted_y):
        self.data=x
        self.rescaled_real_y = rescaled_real_y  #交易数据
        self.rescaled_predicted_y = rescaled_predicted_y
        self.capital = 200000  # 初始资金
        self.risk_percentage=0.2
        self.positions = [] #当前头寸
        self.histroy = [] #交易历史
        self.capitalhistroy = [] #交易历史
        self.i=0
       
    def predTest(self,model, device, test_loader):  
        model.eval()  
        y_pred = torch.tensor([], dtype=torch.float32)  
        with torch.no_grad():  
            for inputs, labels,yc in test_loader:  
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)  
                y_pred = torch.cat((y_pred, outputs.view(-1, 1)), dim=0)    
        return y_pred
    
    def getPredPrice(self,root,number):
        X_test = np.load(root+"data/"+number+"/X_test.npy", allow_pickle=True)
        y_test = np.load(root+"data/"+number+"/y_test.npy", allow_pickle=True)
        yc_test = np.load(root+"data/"+number+"/yc_test.npy", allow_pickle=True)
  
        y_test_tensor = torch.from_numpy(y_test).to(device,dtype=torch.float32)  
        yc_test_tensor= torch.from_numpy(yc_test).to(device,dtype=torch.float32)  
        test_dataset = StockDataset(torch.from_numpy(X_test).to(device,dtype=torch.float32), y_test_tensor,yc_test_tensor)  
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)  

        model = torch.load(root+"data/"+number+"/model_wgan_gp.pth")  
        # 测试模型  
        y_pred = self.predTest(model, device, test_loader)
        y_scaler = load(open(root+"data/"+number+"/y_scaler.pkl", 'rb'), allow_pickle=True)
        test_predict_index = np.load(root+"data/"+number+"/index_test.npy", allow_pickle=True)

        rescaled_real_y = y_scaler.inverse_transform(y_test)
        rescaled_predicted_y = y_scaler.inverse_transform(y_pred)
        return rescaled_real_y,rescaled_predicted_y
    
    def split_train_test(self,data):
        total = len(data)
        train_size = round(total * 0.8)
       
        data_test = data[train_size:]
        return data_test
    
    def doProcess(self,number,root):        
         
        rescaled_real_y,rescaled_predicted_y = self.getPredPrice(root,number)
      
        dataset = pd.read_csv(root+"data/"+number+"/prepared_data.csv", parse_dates=['日期'])
        x = self. split_train_test(dataset[25:-2])
        self.init(x,rescaled_real_y,rescaled_predicted_y)
        
        
        self.data['sign'] = self.data.apply(self.doProcessTracking, axis=1)
        
        print("总资产", self.capitalhistroy[-1])
        
        plt.figure(figsize=(16, 8))
        plt.plot(self.data['收盘'])
        plt.plot(self.data['short_20_low'], color='g')
        plt.plot(self.data['short_20_high'], color='g') 
        plt.plot(self.data['long_40_low'], color='r')
        plt.plot(self.data['long_40_high'], color='r') 
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("real", "low","high" ), loc="upper left", fontsize=16)
        plt.title("The result of Testing", fontsize=20)
        plt.show()
        
        plt.figure(figsize=(16, 8))
 
        plt.plot(self.capitalhistroy, color='r')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("real", "low","high","cachs"), loc="upper left", fontsize=16)
        plt.title("The result of Testing", fontsize=20)
        plt.show()
        print('finish')
            
    
    def doProcessTracking(self,row):
        date = row['日期']
        price = row['收盘']
        previous_close = row['上一日收盘']
        short_20_low = row['short_20_low']
        short_20_high = row['short_20_high']
        long_low = row['long_40_low']
        long_high = row['long_40_high'] 
        atr = row['atr'] 
        price2 = self.rescaled_real_y[self.i]
        predicted_price = self.rescaled_predicted_y[self.i]
        price=predicted_price
        position_times = len(self.positions)
        sign='nothing'
        if  position_times>0 :
            last_entry_price = self.positions[-1].price
            #有头寸
            if self.calculate_stop_losses(last_entry_price, position_times,price, atr) or self.should_exit_market(price, short_20_low, short_20_high, "long"):
                # 止损，或者离市(止赢钱)
                self.sellAll(date,atr,price)  
                sign='sell_all'              
            elif "buy"==self.tracking(price, previous_close,short_20_low,short_20_high,long_low,long_high,last_entry_price,atr,position_times):
                #加仓.最多加仓4个单位。
                if position_times<=4:
                    self.buy(date,atr,price)
                    sign='buy'                  
        else:
            #无头寸
            if "buy" == self.determine_position(price, previous_close,short_20_low,short_20_high,long_low,long_high):
                #建仓
                self.buy(date,atr,price)
                sign='buy'

        self.logCapitalhistroy(price)
        self.i = self.i+1
        return sign  
        
    def logCapitalhistroy(self,price):
        position_size = self.getTotalPositions(self.positions)
        _capital = self.capital+price*position_size 
        self.capitalhistroy.append(_capital)    
                
    def buy(self,date,atr,price):
        position_size = self.calculate_position_size( atr,self.capital , price,self.risk_percentage)
        p = Position()
        p.date=date
        p.price = price
        p.size=position_size
        p.sign="buy"
        self.positions.append(p) 
        self.histroy.append(p) 
        self.capital = self.capital-price*position_size 
        
    def sellAll(self,date,atr,price):
        position_size = self.getTotalPositions(self.positions)
        p = Position()
        p.date=date
        p.price = price
        p.size=position_size
        p.sign="sell_all"
        self.positions.clear()
        self.histroy.append(p) 
        self.capital = self.capital+price*position_size
    
    def getTotalPositions(self,positions):
        total = 0
        for x in positions:
            total+=x.size
        return total
    
    def prepareData(self):
        self.data['上一日收盘'] = self.data['收盘'].shift(1)
        self.data['上一日最低'] = self.data['最低'].shift(1)
        self.data['上一日最高'] = self.data['最高'].shift(1)


        self.data['short_20_low'] = self.data['上一日最低'].rolling(window=20, min_periods=1).min()  
        self.data['short_20_high'] = self.data['上一日最高'].rolling(window=20, min_periods=1).max() 
        self.data['long_40_low'] = self.data['上一日最低'].rolling(window=40, min_periods=1).min()  
        self.data['long_40_high'] = self.data['上一日最高'].rolling(window=40, min_periods=1).max() 
 
        print(self.data)
        
    def doProcessATR(self,data):
        period =14
        data['tr'] = 0.0           
        data['tr'] = data.apply(self.calculate_TR, axis=1)    
        # 计算ATR  
        data['atr'] = data['tr'].rolling(window=period).mean()  
        data.iloc[:period-1] = np.nan  # 前period-1个值无法计算ATR，设为NaN  
        
    def calculate_TR(self,row):
        high = row['最高']
        low = row['最低']
        previous_high = row['上一日最高']
        previous_low = row['上一日最低']
        previous_close = row['上一日收盘']
        """
        max(high-low,abs(high-previous_close),abs(low-previous_close))
        """
        if previous_high == previous_low:
            high - previous_close
        return max(high-low, abs(high - previous_close), abs(low - previous_close))

    def calculate_position_size(self, atr, account_size, point_value, risk_percentage=0.01):  
        """  
        计算海龟交易法的头寸规模。  
        参数:  
            atr (float): ATR值  
            account_size (float): 账户规模（资金总额）  
            point_value (float): 合约每一点所代表的价值   对于股票交易，每一点的价值通常就是股票的价格本身
            risk_percentage (float, 可选): 愿意承担的风险百分比，默认为0.01（即1%）  
        返回:  
            position_size (int): 计算得到的头寸规模（向下取整）  
        """  

        # 计算绝对波动幅度  
        absolute_volatility = atr * point_value  
        # 计算头寸规模  
        position_size = int(account_size * risk_percentage / absolute_volatility)  
        return position_size  
    

    def determine_position(self,price, previous_close,short_low,short_high,long_low,long_high):
        """  
        入市算法，是否建仓
            1、20日突破为基础的偏短系统-突破定义为超过前20日的最高价或最低价。
            2、40日突破为基础的较简单的长线系统-只要有一个信号显示价格超过了前40日最高价或最低价就建立头寸。
        
        参数:  
             price 当前价格,
             previous_close 昨天收盘价
             short_low  短期最低点， 20天最低价格
             short_high 短期最高点,  20天最高价格
             long_low,  长期最低点， 40天最低价格
             long_high  长期最高点， 40天最高价格
        """   
        
        if price > short_high and previous_close<short_high:
            #突破为基础的偏短系统
            return "buy"
        if price > long_high and previous_close<long_high:
            #突破为基础的长线系统
            return "buy"
        
        #卖空逻辑
        
        if price < short_low and previous_close>short_low:
            #突破为基础的偏短系统
            return "sell"
        if price < long_high and previous_close>long_low:
            #突破为基础的长线系统
            return "sell"
    
    def tracking(self,price, previous_close,short_low,short_high,long_low,long_high,last_buy_price,N,position_times):
        if position_times >0 :
            price > last_buy_price +0.5*N
            return "buy"
        else:
            return "nothing"
            
        
        
    def calculate_stop_losses(self,last_entry_price, postion_times,price, N): 
        """
        止损
        """     
        if (last_entry_price - 2*N ) < price :
            return True
        else:
            return False
        
    def should_exit_market(self,current_price, low, high, position_direction):  
        """  
        根据海龟交易法判断是否应该离市。  
        当价格波动与头寸方向相反，即价格涨势中出现多头头寸或价格跌势中出现空头头寸时，如果价格突破了过去20日的最低价（对于多头头寸）或最高价（对于空头头寸），那么所有头寸中的所有单位都会退出市场。
        :param current_price: 当前市场价格
        :param recent_prices: 过去20日/20日（2种时间策略都可以）的价格列表（从最新到最旧）  
        :param position_direction: 头寸方向，'long'表示多头，'short'表示空头  
        :return: 如果应该离市则返回True，否则返回False 
        """  

        # if len(recent_prices) < 20:  
        #     raise ValueError("recent_prices must contain at least 20 prices.")  
        if position_direction == 'long':  
            # 对于多头头寸，检查是否跌破了过去20日的最低价
            return current_price <= low 
        elif position_direction == 'short':  
            # 对于空头头寸，检查是否突破了过去20日的最高价  
            return current_price >= high  
        else:  
            raise ValueError("position_direction must be either 'long' or 'short'.")  


 