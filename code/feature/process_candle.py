import mplfinance as mpf
import pandas as pd
#https://github.com/matplotlib/mplfinance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') # 关闭matplotlib的交互功能，避免内存泄露
# from mplfinance.original_flavor import Line2D, Rectangle # 调取mplfinance包中的基础函数，用来画线和柱状图
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from PIL import Image
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"
# 创建示例数据
# 相比于mplfinance包的源代码，改变了一个小细节，让蜡烛芯更粗
def new_candlestick(ax, quotes, width=0.2, colorup='k', colordown='r', alpha=1.0, ochl=True):

    OFFSET = width / 2.0

    lines = []
    patches = []
    
    for index, q in quotes.iterrows():
        t = q['Date']
        open = q['Open']
        close = q['Close']
        high = q['High']
        low = q['Low']
        

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close
        #蜡烛图的上下影线
        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=2.0, # 原来是1.0
            antialiased=True,
        )
        #每天的蜡烛体
        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
        
    #画移动平均线
    mavline = Line2D(
            xdata=quotes['Date'], ydata=quotes['ma21'],
            color='b',
            linewidth=3.0, # 原来是1.0
            antialiased=True,
    )
    ax.add_line(mavline)
    ax.autoscale_view()

    return lines, patches
def new_candlestick_ohlc(ax, quotes, width=0.2, colorup='k', colordown='r', alpha=1.0):
    # 调用自定义的new_candlestick，原来是candlestick
    return new_candlestick(ax, quotes, width=width, colorup=colorup, colordown=colordown, alpha=alpha, ochl=False)

class DataCandleProcess():
    
    #根据x_step 天画蜡烛图,用来判别形态
    def get_X(self,n_steps_in, x_step, X_data_train):
        X_train = list()

        length1 = len(X_data_train)
        for i in range(0, length1, x_step):
            subdata = X_data_train.iloc[i: i + n_steps_in]
            subdata = subdata[['Date','Open', 'Low','High', 'Close','Volume','ma21']]
            subdata['DateStr'] = subdata['Date'].copy()
            #自己画图,需要把日期转成坐标
            subdata.loc[:, 'Date'] = range(len(subdata))
            X_value_train = pd.DataFrame(subdata,columns=subdata.columns) 
            #自己画图,不需要把date专程日期 
            # X_value_train['Date'] = pd.to_datetime(X_value_train['Date'])  
            # X_value_train.set_index('Date', inplace=True)  
            if len(X_value_train) == n_steps_in  :
                X_train.append(X_value_train)
        return X_train
    
    def saveCandle(self,data_in_step):
        data_in_step['color'] = data_in_step.apply(lambda row: 1 if row['Close'] >= row['Open'] else 0, axis=1)
        # print(data_in_step.head())
        # 创建蜡烛图            
        datestr = str(data_in_step['DateStr'].values[0])
        file_name = "candle-"+datestr+".jpg"
        candle_file = root +"data/"+ number +"/candle/"+file_name
        fig = plt.figure(figsize=(10, 10))  
        grid = plt.GridSpec(10, 10, wspace=0, hspace=0)
        ax1 = fig.add_subplot(grid[0:9, 0:10]) # 设置K线图的尺寸
        ax1.patch.set_facecolor('black') # 设置为黑色背景，其三个通道为均为0    
        new_candlestick_ohlc(ax1, data_in_step, width=0.9, colorup='red', colordown='green', alpha=1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([]) 
        ax2 = fig.add_subplot(grid[8:9, 0:10])
        ax2.patch.set_facecolor('black')
        # 收盘价高于开盘价为红色，反之为绿色
        redset = data_in_step.query('color==1')
        greenset = data_in_step.query('color==0')
        ax2.bar(redset['Date'], redset['Volume'], width=0.9, color='red') 
        ax2.bar(greenset['Date'], greenset['Volume'], width=0.9, color='green')
        # plt.xticks(rotation=30) 
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])   
        fig.savefig(candle_file, bbox_inches='tight', dpi=60)
        
        # 读取原始图片，对图片矩阵进行裁剪后再次保存图片
        image_matrix = np.array(plt.imread(candle_file))
        new_matrix = image_matrix[6:422, 16:470, :]              
        new_image = Image.fromarray(new_matrix)
        new_image.save(candle_file)

        plt.clf()
        plt.close('all') # 关闭画布，避免占用内存
        
        print(file_name)
        
    def doProcess(self,number,root):
        
        csv_file = root +"data/"+ number +"/dataset_train.csv"
        data = pd.read_csv(csv_file) 
        # data = pd.read_csv(csv_file) 
        data.rename(columns={'日期':'Date', '开盘':'Open', '最低':'Low' , '最高':'High' , '收盘':'Close', '成交量(百万手)':'Volume'}, inplace=True)
        # data.index.name = 'Date'
        n_steps_in = 22
        seqed_data = self.get_X(n_steps_in, 1, data)
        
        for i in range(len(seqed_data)):
            data_in_step = seqed_data[i]
            self.saveCandle(data_in_step)

dcp = DataCandleProcess()
dcp.doProcess(number,root)