# 你扮演一位python工程师
# 读取一个份cvs文件，使用pands库，
# cvs有8列，其中7列分别为数字，第8列为日期
# 需要计算出7个数字的如下值:和数/中位数/连续的个数/单数个数，
# 将这些计算出来的数值变成新的列，保存成为新的文件

from turtle_trading_strategy import TurtleTradingStrategy


import numpy as np  
import pandas as pd

class PrepareData():
    def calculate_rsi(self, data, window):  
        diff = data.diff(1)  
        gain = (diff.where(diff > 0, 0)).fillna(0)  
        loss = (-diff.where(diff < 0, 0)).fillna(0)  
        avg_gain = gain.rolling(window=window, min_periods=1).mean()  
        avg_loss = loss.rolling(window=window, min_periods=1).mean()  
        rs = avg_gain / avg_loss  
        rsi = 100 - 100 / (1 + rs)  
        return rsi  
    
    def prepare(self,df):
        #反转行的顺序
        df = df.iloc[::-1]  
        # df['未来一天的收盘'] =  df['涨跌额'].shift(-1)
        df['成交量(百万手)'] =  df['成交量(手)']/1000000        
        df['成交金额(十亿)'] =  df['成交金额(万)']/100000
        df['ma7'] = df['收盘'].rolling(window=7).mean()
        df['ma21'] = df['收盘'].rolling(window=21).mean()
        df['ma40'] = df['收盘'].rolling(window=40).mean()
        # Create MACD
        df['200ema'] = df['收盘'].ewm( span=200).mean()  
        df['100ema'] = df['收盘'].ewm( span=100).mean()  
        df['26ema'] = df['收盘'].ewm( span=26).mean()  
        df['12ema'] = df['收盘'].ewm( span=12).mean()  
        df['MACD'] = (df['12ema']-df['26ema'])
        # Create Bollinger Bands
        df['20sd'] = df['收盘'].rolling(window=20).std()  
        df['upper_band'] = df['ma21'] + (df['20sd']*2)
        df['lower_band'] = df['ma21'] - (df['20sd']*2)
        # Create Exponential moving average
        df['ema'] = df['收盘'].ewm(com=0.5).mean()
        # Create Momentum
        df['logmomentum'] = np.log(df['收盘']-1)
        # 
        
        df['上一日收盘'] = df['收盘'].shift(1)
        df['上一日最低'] = df['最低'].shift(1)
        df['上一日最高'] = df['最高'].shift(1)


        df['short_20_low'] = df['上一日最低'].rolling(window=20, min_periods=1).min()  
        df['short_20_high'] = df['上一日最高'].rolling(window=20, min_periods=1).max() 
        df['long_40_low'] = df['上一日最低'].rolling(window=40, min_periods=1).min()  
        df['long_40_high'] = df['上一日最高'].rolling(window=40, min_periods=1).max() 
        df['rsi'] = self.calculate_rsi(df['收盘'],14)
        tts = TurtleTradingStrategy()
        tts.doProcessATR(df)
        
        
        return df
    
    #Getting the Fourier transform features
    def get_fourier_transfer(dataset):
        # Get the columns for doing fourier
        data_FT = dataset[['日期', '收盘']]

        close_fft = np.fft.fft(np.asarray(data_FT['收盘'].tolist()))
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

        fft_list = np.asarray(fft_df['fft'].tolist())
        fft_com_df = pd.DataFrame()
        for num_ in [3, 6, 9]:
            fft_list_m10 = np.copy(fft_list);
            fft_list_m10[num_:-num_] = 0
            fft_ = np.fft.ifft(fft_list_m10)
            fft_com = pd.DataFrame({'fft': fft_})
            fft_com['absolute_of_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
            fft_com['angle_of_' + str(num_) + '_comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
            fft_com = fft_com.drop(columns='fft')
            fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)
        return fft_com_df


    def do_prepare(self,number,root):
        df = pd.read_csv(root+"data/"+number+"/"+number+".csv")  
        dataset = self.prepare(df)
        dataset.to_csv(root+"data/"+number+"/prepared_data.csv", index=False)
        print("处理完成，新文件已保存为'data/"+number+"/prepared_data.csv'")




  


#Drop the first 21 rows
#For doing the fourier
# dataset = T_df.iloc[20:,:].reset_index(drop=True)


    
