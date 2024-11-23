


import numpy as np  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from train_vae import VAETrainer 

class DataPreProcess():
    n_steps_in = 3
    n_steps_out = 1
    
    def get_y_yc(self,n_steps_in,n_steps_out, y_data):
        y = list()
        yc = list()
        length=len(y_data)
        for i in range(0, length, 1):
            y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
            yc_value = y_data[i: i + n_steps_in][:, :]
            if len(yc_value) == n_steps_in and len(y_value) == 1:      
                y.append(y_value)
                yc.append(yc_value)
            else:
                print("3")
        return np.array(y), np.array(yc)
    
    def get_X(self,n_steps_in,X_data_train, X_data_test):
        vae = VAETrainer()
        X_data_train,X_data_test = vae.doProcess(X_data_train,X_data_test)
        print(X_data_train.shape)
        X_train = list()
        X_test = list() 

        length1 = len(X_data_train)
        for i in range(0, length1, 1):
            X_value_train = X_data_train[i: i + n_steps_in][:, :]
            if len(X_value_train) == n_steps_in  :
                X_train.append(X_value_train)

        length2 = len(X_data_test)-n_steps_in 
        for i in range(0, length2, 1):           
            X_value_test = X_data_test[i: i + n_steps_in][:, :]
            if len(X_value_test) == n_steps_in  :
                X_test.append(X_value_test)
            else:
                print("1")
 
        return np.array(X_train), np.array(X_test)


    # get the train test predict index
    def predict_index(self,dataset, X_train, n_steps_in, n_steps_out):
        # get the predict data (remove the in_steps days)
        train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index
        print(train_predict_index)
        print(test_predict_index[1:1 + 1])
        return train_predict_index, test_predict_index

    # Split train/test dataset
    def split_train_test(self,train_size,data):
        data_train = data[0:train_size]
        data_test = data[train_size:]
        return data_train, data_test


    def doProcess(self,number,root):
        dataset = pd.read_csv(root+"data/"+number+"/prepared_data.csv", parse_dates=['日期'])
        # Replace 0 by NA
        dataset.replace(0, np.nan, inplace=True)
         # Check NA and fill them
        dataset.isnull().sum()
        # dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill(), dataset.iloc[:, 1:].bfill()]).groupby(level=0).mean()
        print(dataset.columns)

        # Get features and target
        X_value = pd.DataFrame(dataset[['开盘', '最低','最高', 
            '收盘','ma7','ma21','200ema','100ema','26ema','12ema','MACD','20sd','upper_band','lower_band',
            'ema','rsi',
            'logmomentum',
            '成交量(百万手)', 
            '成交金额(十亿)', 
            ]])
        print(X_value.head())
        y_value = pd.DataFrame(dataset[['收盘']])
        print(y_value.head())
 
        # Normalized the data
        X_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))

        X_scaler.fit(X_value)
        y_scaler.fit(y_value)


        X_scale_dataset = X_scaler.fit_transform(X_value)
        y_scale_dataset = y_scaler.fit_transform(y_value)

        dump(X_scaler, open(root+"data/"+number+"/X_scaler.pkl", 'wb'))
        dump(y_scaler, open(root+"data/"+number+"/y_scaler.pkl", 'wb'))

        # X_scale_dataset = X_value.values
        # y_scale_dataset = y_value.values


 
        #步数
        print("步数:",self.n_steps_in)
        #特征数
        n_features = X_value.shape[1]
        print("特征数:",n_features)
        
        dataset = dataset[25:-2]
       
        X_scale_dataset = X_scale_dataset[25:-2]
        y_scale_dataset = y_scale_dataset[25:-2]       
      
        train_size = round( len(dataset) * 0.8)
        dataset_train, dataset_test  = self.split_train_test(train_size,dataset)
        dataset_train.to_csv(root+"data/"+number+"/dataset_train.csv", index=False)
        dataset_test.to_csv(root+"data/"+number+"/dataset_test.csv", index=False)
        #迁移学习
        
        X_data_train, X_data_test  = self.split_train_test(train_size,X_scale_dataset)
        X_train, X_test, = self.get_X(self.n_steps_in,X_data_train, X_data_test)
        Y, YC = self.get_y_yc(self.n_steps_in,self.n_steps_out, y_scale_dataset)
        
        # X, Y, YC = self.get_X_y(self.n_steps_in,self.n_steps_out,X_scale_dataset, y_scale_dataset)

        # print(X.shape)
        # X = X[25:-2,:,:]
        #X连续3天的特征
        # print(X[0])
        print(Y.shape)
        # Y = Y[25:-2,:]
        #Y第4天的收盘价
        print(Y[0])
        print(YC.shape)
        # YC = YC[25:-2,:,:]
        #YC连续3天的收盘价
        print(YC[0])
      
        

        # X_train, X_test, = self.split_train_test(train_size,X)
        y_train, y_test, = self.split_train_test(train_size,Y)
        yc_train, yc_test, = self.split_train_test(train_size,YC)
        index_train, index_test, = self.predict_index(dataset, X_train, self.n_steps_in, self.n_steps_out)




        np.save(root+"data/"+number+"/X_train.npy", X_train)
        np.save(root+"data/"+number+"/y_train.npy", y_train)
        np.save(root+"data/"+number+"/X_test.npy", X_test)
        np.save(root+"data/"+number+"/y_test.npy", y_test)
        np.save(root+"data/"+number+"/yc_train.npy", yc_train)
        np.save(root+"data/"+number+"/yc_test.npy", yc_test)
        np.save(root+"data/"+number+"/index_train.npy", index_train)
        np.save(root+"data/"+number+"/index_test.npy", index_test)
        print("np 文件保存完毕")