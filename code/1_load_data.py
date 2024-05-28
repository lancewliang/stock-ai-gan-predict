# 你扮演一位python工程师
# 读取一个份cvs文件，使用pands库，
# cvs有8列，其中7列分别为数字，第8列为日期
# 需要计算出7个数字的如下值:和数/中位数/连续的个数/单数个数，
# 将这些计算出来的数值变成新的列，保存成为新的文件



import numpy as np  
import pandas as pd
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"


def prepare(df):

    #反转行的顺序
    df = df.iloc[::-1]  
    # df['未来一天的收盘'] =  df['涨跌额'].shift(-1)
    df['成交量(百万手)'] =  df['成交量(手)']/1000000
    
    df['成交金额(十亿)'] =  df['成交金额(万)']/100000
    
    
    df['ma7'] = df['收盘'].rolling(window=7).mean()
    df['ma21'] = df['收盘'].rolling(window=21).mean()
    
    # Create MACD
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
    return df
    
    # print(df)  
    
    
    

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

#Get Fourier features
# dataset_F = get_fourier_transfer(dataset)
# Final_data = pd.concat([dataset, dataset_F], axis=1)
    
    
# def label(row):  
#     x = 0
    
#     col = row['变化率']
#     if  col >0.2:
#         x =3
#         if col>0.2 and col<=2:
#             x=4
#         if col>2 and col<=4:
#             x=5
#         if col>4 and col<6:
#             x=6
#         if col>6 and col<10:
#             x=7
#     elif col<0: 
#         x=2
#         if col<-2:
#             x=1
#         if col<-5:
#             x=0
# return x

number = "601857"

df = pd.read_csv(root+"data/"+number+"/"+number+".csv")    
dataset = prepare(df)

#Drop the first 21 rows
#For doing the fourier
# dataset = T_df.iloc[20:,:].reset_index(drop=True)

dataset.to_csv(root+"data/"+number+"/prepared_data.csv", index=False)
    
print("处理完成，新文件已保存为'data/"+number+"/prepared_data.csv'")