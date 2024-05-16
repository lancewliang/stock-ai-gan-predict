# 你扮演一位python工程师
# 读取一个份cvs文件，使用pands库，
# cvs有8列，其中7列分别为数字，第8列为日期
# 需要计算出7个数字的如下值:和数/中位数/连续的个数/单数个数，
# 将这些计算出来的数值变成新的列，保存成为新的文件



import numpy as np  
import pandas as pd
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"


def prepare():
    df = pd.read_csv(root+"data/601857.csv")
    # 假设数字列的名称分别是'col1', 'col2', ..., 'col7'  
    # 使用str.split()方法按空格分割字符串，并扩展结果到新的列  
    # 注意：expand=True会将分割后的列表扩展为DataFrame的列  
    df = df.iloc[::-1]  
    df['变化'] =  round(df['收盘'].shift(-1) - df['收盘'] ,2)
    
    df['变化率'] =round( df['变化']*100/df['收盘'],2)
    
    df['标签'] = df.apply(label, axis=1)  
    
    
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
    df['momentum'] = df['收盘']-1
    
    
    # 
    
    df.to_csv(root+"data/prepared_data.csv", index=False)
    
    print("处理完成，新文件已保存为'data/prepared_data.csv'")
    # print(df)  

def label(row):  
    x = 0
    
    col = row['变化率']
    if  col >0.2:
        x =3
        if col<=3:
            x=4
        if  col>3:
            x=5
        if  col>6:
            x=6
    elif col<0: 
        x=2
        if col<-3:
            x=1
        if  col<-6:
            x=0
         
        
    return x
    
prepare()