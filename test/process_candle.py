import mplfinance as mpf
import pandas as pd
#https://github.com/matplotlib/mplfinance
import numpy as np

root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"
# 创建示例数据


class DataCandleProcess():
    
    
    def get_X(self,n_steps_in, x_step, X_data_train):
        X_train = list()

        length1 = len(X_data_train)
        for i in range(0, length1, x_step):
            subdata = X_data_train.iloc[i: i + n_steps_in].copy()
            X_value_train = pd.DataFrame(subdata,columns=X_data_train.columns)  
            X_value_train['Date'] = pd.to_datetime(X_value_train['Date'])  
            X_value_train.set_index('Date', inplace=True)  
            if len(X_value_train) == n_steps_in  :
                X_train.append(X_value_train)

        return X_train
    
    def doProcess(self,number,root):
        
        csv_file = root +"data/"+ number +"/dataset_train.csv"
        data = pd.read_csv(csv_file) 
        # data = pd.read_csv(csv_file) 
        data.rename(columns={'日期':'Date', '开盘':'Open', '最低':'Low' , '最高':'High' , '收盘':'Close', '成交量(百万手)':'Volume'}, inplace=True)
        data.index.name = 'Date'
        n_steps_in = 22
        seqed_data = self.get_X(n_steps_in, 20, data)
        
        for i in range(len(seqed_data)):
            data_in_step = seqed_data[i]
            # 创建蜡烛图并添加技术指标
            add_plot = [
                mpf.make_addplot(data_in_step['ma21'], color='b'),
                # mpf.make_addplot(data['MA50'], color='r')
            ]
            datestr = str(data_in_step.index[0])
            file_name = "candle-"+datestr+".png"
            candle_file = root +"data/"+ number +"/candle/"+file_name
            fig = mpf.plot(data_in_step, type='candle',                      
                    ylabel='Price',  
                    addplot=add_plot , volume=True,savefig=candle_file
                    
                    )
            print(file_name)

dcp = DataCandleProcess()
dcp.doProcess(number,root)