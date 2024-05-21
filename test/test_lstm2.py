import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import TensorDataset, DataLoader  
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# 定义LSTM模型  
class LSTMModel(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size):  
        super(LSTMModel, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)  
  
    def forward(self, x):  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # 初始化隐藏状态  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # 初始化细胞状态  
        out, _ = self.lstm(x, (h0, c0))  # LSTM层，输出序列和最终状态未使用  
        print(out.size())
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出  
        print(out.size())
        return out  
  
# 假设您的数据集已经准备好了  
# X_train: (200, 20, 5) 的时间序列数据  
# y_train: (200, 5) 的标签数据  
  
  
  


root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"
def normalize_data(data):  
    print(data.size())
    """  
    Normalize the data to [0, 1] range.  
  
    :param data: NumPy array to be normalized  
    :return: Normalized NumPy array  
    """  
    # 计算数据的最小值和最大值  
    data_min, _ = torch.min(data, dim=-1, keepdim=True)  

    data_max, _ = torch.max(data, dim=-1, keepdim=True)  

    # data_min = np.min(data, axis=(0, 1))  # 假设data的形状是(samples, time_steps, features)  
    # data_max = np.max(data, axis=(0, 1))  
  
    # 归一化数据  
    # 使用广播机制将mins和maxs扩展到与data相同的形状，然后执行逐元素的归一化  
    normalized_data = (data - data_min) / (data_max - data_min + 1e-7)  # 防止除以零  
  
    return normalized_data  


def transfer2dTo3dWithSequenceLen(sequence_length,df):
    # 3. 初始化一个空列表来存储三维数据块（每个数据块是一个时间步长）  
    data_cubes = []  
    
    # 4. 逆序遍历数据（从最后一行开始），除了前7行之外（因为我们需要至少7行来形成一个时间步长）  
    for i in range(len(df) , sequence_length - 1, -1):  
        if i < sequence_length:  # 如果i小于7，则没有足够的行来形成一个时间步长，跳过  
            break  
        # 选择当前行及其之前的7行（不包括当前行）  
        start_idx = i - sequence_length  
        end_idx = i  
        current_slice = df.iloc[start_idx:end_idx]  
         
        # 将数据转换为NumPy数组  
        numpy_array = current_slice[[
            '开盘', '最低','最高',# '成交量(手)',#'换手率',
            '收盘','ma7','ma21','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','momentum']].astype('float32').values   
        # 如果你需要，可以在这里对数据进行进一步处理（例如，标准化、归一化等）  
        
        # 将NumPy数组添加到列表中  
        data_cubes.append(numpy_array)  
    
    # 因为我们是逆序遍历的，所以需要将列表反转回来以保持原始的顺序（如果需要的话）  
    data_cubes.reverse()  
    
    # 5. 将列表中的NumPy数组转换为PyTorch的三维张量  
    # 首先确定张量的形状（时间步长数量, 时间步长内的行数, 特征数量）  
    num_cubes = len(data_cubes)  
    
    num_rows_per_cube = sequence_length  # 因为每个时间步长有7行  
    num_features = data_cubes[0].shape[1]  # 假设所有时间步长有相同的特征数量  
    tensor_3d = torch.zeros((num_cubes, num_rows_per_cube, num_features), dtype=torch.float32)  

    # 将NumPy数组填充到三维张量中  
    for i, numpy_cube in enumerate(data_cubes): 
        tensor_3d[i] = torch.from_numpy(numpy_cube)
    # normalized_data = normalize_data(tensor_3d)  
    return tensor_3d
 


def load_traning_data(sequence_length):  
    file = root+"data/prepared_data.csv"
    x_df = pd.read_csv(file)   
    x_df = x_df.iloc[25:] 
    #删除最后一行
    df_len= len(x_df)

    split_index = int(df_len*0.8)
    
    # split_index = 500
    df_traing = x_df.iloc[0:split_index]  
    df_test = x_df.iloc[split_index:]  
        
    X = transfer2dTo3dWithSequenceLen(sequence_length,df_traing)
    print(X.size())
    # print(X)
    
    y_label_df= x_df.iloc[sequence_length:split_index+1]
    y_label=y_label_df[['开盘', '最低','最高',# '成交量(手)',#'换手率',
            '收盘','ma7','ma21','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','momentum']].astype('float32').values  
    
    Y = torch.from_numpy(y_label).to(device,dtype=torch.float32)  
    print(Y.size())
    # print(Y)
    # X_test = transfer2dTo3dWithSequenceLen(sequence_length,df_test)
    # Y_test = torch.from_numpy(df_test['标签'].astype('int32').values ).to(device,dtype=torch.long)  
    
    return X.to(device) , Y 



  
# 定义超参数  
input_size = 14  # 特征数量  
hidden_size = 32  # LSTM隐藏层单元数  
num_layers = 1  # LSTM层数  
output_size = 14  # 输出特征数量  
learning_rate = 0.01  
batch_size = 32  
num_epochs = 500  
  

# 将数据转换为torch张量  
X_train,y_train = load_traning_data(5)
#y_train = torch.tensor(y_train, dtype=torch.float32)  
  
  
  
# 实例化模型  
model = LSTMModel(input_size, hidden_size, num_layers, output_size)  
  
# 定义损失函数和优化器  
criterion = nn.MSELoss()  # 假设是回归问题，使用均方误差损失  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  
# 创建数据加载器  
train_dataset = TensorDataset(X_train, y_train)  
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  
  
# 训练模型  
for epoch in range(num_epochs):  
    for i, (inputs, labels) in enumerate(train_loader):  
        # 梯度清零  
        optimizer.zero_grad()  
          
        # 前向传播  
        outputs = model(inputs)  
          
        # 计算损失  
        loss = criterion(outputs, labels)  
          
        # 反向传播和优化  
        loss.backward()  
        optimizer.step()  
          
        if (i+1) % 10 == 0:  
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')  
  
# 模型训练完成后，可以使用它进行预测  
# ...  
  
# 注意：这里未包含验证和测试步骤，您可能需要在训练循环中添加这些步骤以监控模型性能