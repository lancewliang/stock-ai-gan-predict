
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd
from torch.utils.data import Dataset, DataLoader  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

np.random.seed(0)  

# 定义超参数  
input_size = 14  # 因子X的特征数量  
hidden_size = 128  # LSTM隐藏层大小  
num_layers = 1  # LSTM层数  
num_classes=7 #涨跌

# output_size_reg = 1  # 标签Y的回归特征数量 
num_epochs = 2000  # 训练轮数  
batch_size = 64  # 批处理大小  
learning_rate = 0.001  # 学习率  
  
# 生成训练和测试数据  
# num_samples = 2  # 生成1000个样本  
sequence_length = 22  # 每个样本有10个时间步长  

# LSTM模型定义  
class LSTMClassifier(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  
        super(LSTMClassifier, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, num_classes) 
  
    def forward(self, x):  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]
        
        class_output = self.fc(out)
        return class_output 
    
    
 
# 初始化模型、损失函数和优化器  
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)  
 
criterion = nn.CrossEntropyLoss()  # 二损失函数，用于多标签分类  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  
# 将模型和数据移动到GPU（如果有的话），否则使用CPU  

model = model.to(device)  



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
    df_traing = x_df.iloc[0:split_index]  
    df_test = x_df.iloc[split_index:]  
        
    X = transfer2dTo3dWithSequenceLen(sequence_length,df_traing)
    Y = torch.from_numpy(df_traing['标签'].astype('int32').values ).to(device,dtype=torch.long)  
    
    X_test = transfer2dTo3dWithSequenceLen(sequence_length,df_test)
    Y_test = torch.from_numpy(df_test['标签'].astype('int32').values ).to(device,dtype=torch.long)  
    
    return X.to(device) , Y, X_test.to(device) , Y_test




# 将numpy数组转换为PyTorch张量  
X_train, y_train,X_test, y_test = load_traning_data(sequence_length)

# 定义一个简单的Dataset类  
class StockDataset(Dataset):  
    def __init__(self, X, y):  
        self.X = X  
        self.y = y  
  
    def __len__(self):  
        return len(self.X)  
  
    def __getitem__(self, idx):  
        return self.X[idx], self.y[idx]  
  
 

# 构建Dataset和DataLoader  
train_dataset = StockDataset(X_train, y_train)  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
  
test_dataset = StockDataset(X_test, y_test)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  



# 训练模型  
def train(model, device, train_loader, optimizer, criterion, num_epochs=25):  
    model.train()  
    for epoch in range(num_epochs):  
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
              
            # 梯度清零  
            optimizer.zero_grad()  
              
            # 正向传播  
            outputs = model(inputs)  
            # print(outputs.size())
            # print(labels.size())
            # print(outputs)
            # print(labels)
            
            # 计算损失  
            loss = criterion(outputs, labels)  
              
            # 反向传播和优化  
            loss.backward()  
            optimizer.step()  
          
        if (epoch+1) % 5 == 0:  
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  
  
# 测试模型  
def test(model, device, test_loader, criterion):  
    model.eval()  
    test_loss = 0  
    correct = 0  
    with torch.no_grad():  
        for inputs, labels in test_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            test_loss += criterion(outputs, labels).item()  # 累加测试损失  
            _, predicted = torch.max(outputs.data, 1)  # 获取最大值的索引（即预测的类别）  
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数  
  
    test_loss /= len(test_loader.dataset)  # 计算平均测试损失  
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100 * correct / len(test_loader.dataset):.2f}%\n')  

model = model.to(device)  
  
# 训练模型  
train(model, device, train_loader, optimizer, criterion, num_epochs)  
  
# 测试模型  
test(model, device, test_loader, criterion)