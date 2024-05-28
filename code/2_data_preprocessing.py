


import numpy as np  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
root = "/home/lanceliang/cdpwork/ai/ai-stock/stockai/"

number = "601857"

dataset = pd.read_csv(root+"data/"+number+"/prepared_data.csv", parse_dates=['日期'])
# dataset.to_csv("dataset.csv", index=False)


# Replace 0 by NA
dataset.replace(0, np.nan, inplace=True)



 # Check NA and fill them
dataset.isnull().sum()
# dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill(), dataset.iloc[:, 1:].bfill()]).groupby(level=0).mean()
print(dataset.columns)

# Get features and target
X_value = pd.DataFrame(dataset[['开盘', '最低','最高', 
            '收盘','ma7','ma21','26ema','12ema','MACD','20sd','upper_band','lower_band',
            'ema',
            'logmomentum',
            '成交量(百万手)', 
            '成交金额(十亿)', 
            ]])
print(X_value.head())
y_value = pd.DataFrame(dataset[['收盘']])
print(y_value.head())
 

# Normalized the data
X_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

X_scaler.fit(X_value)
y_scaler.fit(y_value)


X_scale_dataset = X_scaler.fit_transform(X_value)
y_scale_dataset = y_scaler.fit_transform(y_value)

dump(X_scaler, open(root+"data/"+number+"/X_scaler.pkl", 'wb'))
dump(y_scaler, open(root+"data/"+number+"/y_scaler.pkl", 'wb'))

# X_scale_dataset = X_value.values
# y_scale_dataset = y_value.values
#步数
n_steps_in = 3
print("步数:",n_steps_in)
#特征数
n_features = X_value.shape[1]
print("特征数:",n_features)
n_steps_out = 1



def get_X_y(X_data, y_data):
    X = list()
    y = list()
    yc = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        yc_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == n_steps_in and len(y_value) == 1:
            X.append(X_value)
            y.append(y_value)
            yc.append(yc_value)

    return np.array(X), np.array(y), np.array(yc)


X, Y, YC = get_X_y(X_scale_dataset, y_scale_dataset)

print(X.shape)
X = X[25:-2,:,:]
#连续3天的特征
print(X[0])


print(Y.shape)
Y = Y[25:-2,:]
#第4天的收盘价
print(Y[0])
print(YC.shape)
YC = YC[25:-2,:,:]

#连续3天的收盘价
print(YC[0])



# get the train test predict index
def predict_index(dataset, X_train, n_steps_in, n_steps_out):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index
    print(train_predict_index)
    print(test_predict_index[1:1 + 1])
    return train_predict_index, test_predict_index

# Split train/test dataset
def split_train_test(data):
    train_size = round(len(X) * 0.8)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test



X_train, X_test, = split_train_test(X)
y_train, y_test, = split_train_test(Y)
yc_train, yc_test, = split_train_test(YC)
index_train, index_test, = predict_index(dataset, X_train, n_steps_in, n_steps_out)




np.save(root+"data/"+number+"/X_train.npy", X_train)
np.save(root+"data/"+number+"/y_train.npy", y_train)
np.save(root+"data/"+number+"/X_test.npy", X_test)
np.save(root+"data/"+number+"/y_test.npy", y_test)
np.save(root+"data/"+number+"/yc_train.npy", yc_train)
np.save(root+"data/"+number+"/yc_test.npy", yc_test)
np.save(root+"data/"+number+"/index_train.npy", index_train)
np.save(root+"data/"+number+"/index_test.npy", index_test)