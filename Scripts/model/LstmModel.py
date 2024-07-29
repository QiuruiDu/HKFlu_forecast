import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import preprocessing
import collections
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os  
import torch
from torch import nn
from itertools import chain
from tqdm import tqdm
import copy
import random
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math


class LstmDataset():
    # mydataset需要的几个函数：
    # normalization
    # split_windows
    # to tensor
    def __init__(self, sequence_length = 7, batch_size = 2, pred_stamp = 1):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.pred_stamp = pred_stamp
        self.max = {}
        self.min = {}

    def _init_scaler(self, df_train):
        """
        compute the max and min by using train data
        ----------------------------
        df_train: the train dataframe
        """
        for c in df_train.columns:
            self.max[c] = np.max(df_train[c].values)
            self.min[c] = np.min(df_train[c].values)

    def get_scaler(self):
        return self.max, self.min

    def maxmin_normalization(self, df):
        """
        ---------
        max-min normalization
        df: dataframe
        """
        data = copy.deepcopy(df)
        for c in df.columns:
            if c in self.max.keys():
                data[c] = (data[c]-self.min[c])/(self.max[c]-self.min[c])
            else:
                print("the columns ",c," does not exist!")
        data = data.astype(np.float32)
        # print("maxmin_normalization, the min_date = ", data.index.min(), ", the max_date=", data.index.max())
        return data.values
    
    def inverse_normalization(self, df):
        for c in df.columns:
            df[c] = df[c]*(self.max[c]-self.min[c])+self.min[c]
        return df
    
    def get_rate_scaler(self):
        return self.max['rate'], self.min['rate']
    
    def split_windows(self, data):
        """
        ----------
        对序列数据按照时间窗划分
        输入data为多维数组
        """
        x=[]
        y=[]
        # shuffle_x = []
        for i in range(len(data)-self.sequence_length-self.pred_stamp+1): # range的范围需要减去时间步长和1
            _x = data[i:(i+self.sequence_length),:] # (sequence_length, input_dim)
            _y = data[(i+self.sequence_length):(i+self.sequence_length+self.pred_stamp),-1] # (pred_stamp, 1), pred_stamp从0开始
            x.append(_x)
            y.append(_y)
        x, y = np.array(x), np.array(y)
        # print("the original shape is ", data.shape, ", after split, the shape is: ", x.shape, y.shape)
        return x, y
    
    def to_tensor(self, data):
        """
        from ndarray to torch.tensor 
        ------------------------------
        data : nd-array
        """
        return Variable(torch.Tensor(np.array(data)))

    def get_train_val_test_dataset(self, df, test_size = 0, sample_rate = None, train_pure_data = False, validation = True):
        """
        get the train, validation and test data from df dataframe
        ----------------------------------
        df : DataFrame
        test_size: the n size of test data
        """

        # print("-------------- begin data dealing -----------------")
        dataset = copy.deepcopy(df).values
        n = df.shape[0]
        self._init_scaler(df_train = df.iloc[0:(n-test_size),:])
        train_dataset = self.maxmin_normalization(df.iloc[0:(n-test_size),:])
        # print("original train size is : ",train_dataset.shape)
        test_dataset = self.maxmin_normalization(df.iloc[(n-test_size-self.sequence_length):,:]) #test data preview sequence length rows
        # print("original test size is : ",test_dataset.shape)
        # split window
        # print("train_dataset shape = ", train_dataset.shape)
        x_data, y_data = self.split_windows(train_dataset)
        # print("test_dataset shape = ", test_dataset.shape)
        x_test, y_test = self.split_windows(test_dataset)
        # split train and validation data
        if sample_rate is not None:
            # sample_rate = np.random.uniform(0.85, 0.95)
            data_ind = np.random.choice(x_data.shape[0], size=int(x_data.shape[0]*sample_rate), replace=True)
            x_data_sample, y_data_sample = x_data[data_ind,:], y_data[data_ind]
        else:
            x_data_sample, y_data_sample = copy.deepcopy(x_data), copy.deepcopy(y_data)
        if validation == True:
            val_size = math.ceil(x_data_sample.shape[0] * 0.9)
            x_train, y_train = x_data_sample[0:val_size,:,:],y_data_sample[0:val_size]
            x_val, y_val = x_data_sample[val_size:,:,:], y_data_sample[val_size:]
        else:
            x_train, y_train = x_data_sample,y_data_sample
            x_val, y_val = x_data_sample, y_data_sample
        # turn to tensor 
        x_data, y_data = self.to_tensor(x_data), self.to_tensor(y_data)
        x_train, y_train = self.to_tensor(x_train), self.to_tensor(y_train)
        x_val, y_val = self.to_tensor(x_val), self.to_tensor(y_val)
        x_test, y_test = self.to_tensor(x_test), self.to_tensor(y_test)
        train_dataset = TensorDataset(x_train, y_train)
        if self.batch_size > 0:
            train_dataloader = DataLoader(dataset=train_dataset,batch_size=self.batch_size, shuffle=True, drop_last=False)
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            test_datadict = {'x_data':x_test, 'y_data':y_test}
            if train_pure_data is True:
                train_datadict = {'x_data':x_data,'y_data':y_data}
                return train_dataloader, val_datadict, test_datadict, train_datadict
            return train_dataloader, val_datadict, test_datadict
        else:
            train_datadict = {'x_data':x_train, 'y_data':y_train}
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            test_datadict = {'x_data':x_test, 'y_data':y_test}
            return train_datadict, val_datadict, test_datadict

    def get_all_dataset(self, df, test_size = 0):
        """
        get the x data and y data for the whole dataframe
        -----------------
        df : dataframe
        """
        n = df.shape[0]
        self._init_scaler(df_train = df.iloc[0:(n-test_size),:])
        _dataset = self.maxmin_normalization(df)
        # _dataset = df.values
        x_data, y_data = self.split_windows(_dataset)
        x_data, y_data = self.to_tensor(x_data), self.to_tensor(y_data)
        my_dataset = TensorDataset(x_data, y_data)
        data_dict = {'x_data': x_data, 'y_data': y_data}
        if self.batch_size > 0:
            my_dataloader = DataLoader(dataset=my_dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
            return my_dataloader, data_dict
        else:
            return data_dict


class LstmModel(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_dim, mid_dim=8, hidden_lstm_layers = 3, dropout_rate = 0.5, num_directions = 1, MCDropout = 0.5):
        super(LstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = mid_dim
        self.hidden_lstm_layers = hidden_lstm_layers
        self.sequence_dim = sequence_dim
        self.num_directions = num_directions
        if num_directions == 1:
            self.bidirectional = False
        else:
            self.bidirectional = True
        self.MCDropout = MCDropout
        self.lstm = nn.LSTM(input_size = input_dim
                                ,hidden_size = mid_dim # 输出的维度是中间层的维度
                                ,num_layers = hidden_lstm_layers 
                                ,batch_first = True
                                ,dropout = dropout_rate
                                ,bidirectional = self.bidirectional
                                )
        self.pred = nn.Sequential(
            # nn.BatchNorm1d(self.sequence_dim),
            # nn.ReLU(),
            nn.Linear(mid_dim*self.num_directions, output_dim)
        )
    def MC_dropout(self, x):
        return F.dropout(x, p=self.MCDropout, training=True, inplace=True)

    def forward(self, input):
        # lstm model
        # print("input shape", input.shape)
        batch_size, seq_len = input.size()[0], input.size()[1] # input = (batch_size, sequence_len, input_dim)
        h_0 = torch.randn(self.num_directions*self.hidden_lstm_layers, input.size(0), self.hidden_size, device=input.device)
        # print("h0 shape", h_0.shape)
        c_0 = torch.randn(self.num_directions*self.hidden_lstm_layers, input.size(0), self.hidden_size, device=input.device)
        # if torch.cuda.is_available():
        #     h_0, c_0 = h_0.to('cuda'), c_0.to('cuda')
        output_y, _ = self.lstm(input, (h_0, c_0)) #output = (batch_size, sequence_len, mid_dim)
        if self.MCDropout is not None:
            output_y = self.MC_dropout(output_y.clone())
        pred_y = self.pred(output_y.clone())  # (batch_size, sequence_len, out_dim)
        pred_y = pred_y[:, -1, :].clone()  # (batch_size, out_dim)

        return pred_y


class LstmTrain():
    def __init__(self, model):
        self.model = model

    def train(self, train_dataloader, val_datadict, num_epochs, lr = 5e-3, early_stopping = None, loss_weight = None, verboose = 0):
        """
        train the model
        ---------------------
        train_dataloader : The Dataloader of training data
        val_dataloader : The Dataloader of validation data
        lr : learning rate, default = 5e-3
        """
        # single stamp
        if loss_weight is None:
            loss_function = self.loss_func_single
        else:
            # multiple stamps
            loss_function = self.loss_func_apart
        # loss_function = nn.SmoothL1Loss()
        
        # if torch.cuda.is_available():
        #     loss_function = loss_function.to('cuda')
        
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr = lr)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) # 根据epoch降低学习率
        scheduler = ReduceLROnPlateau(optimizer, 'min') # 根据指标降低学习率 # , factor=0.5, patience=10

        # begin training
        iter=0
        best_model = None
        min_val_loss = 1e6
        val_loss = 1e8
        min_train_loss = 1e8
        for epochs in range(num_epochs): 
            train_loss = []
            for i,(batch_x, batch_y) in enumerate(train_dataloader):
                # if torch.cuda.is_available():
                #     batch_x, batch_y = batch_x.to('cuda'), batch_y.to('cuda')
                # print("batch x shape",batch_x.shape, ", y shape ",batch_y.shape)
                pred_y = self.model(batch_x)
                # loss = loss_function(pred_y, batch_y) # 计算损失
                loss = loss_function(batch_y, pred_y, loss_weight)
                train_loss.append(loss.item())
                optimizer.zero_grad()   # 将每次传播时的梯度累积清除
                loss.backward() # 反向传播
                optimizer.step()
                iter = iter + 1

            val_loss = self.get_val_loss(val_datadict['x_data'], val_datadict['y_data'], loss_weight)
            scheduler.step(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_train_loss = np.mean(train_loss)
                best_model = copy.deepcopy(self.model)
                early_stopping_counter = 0
            else:
                early_stopping_counter = early_stopping_counter + 1
            if verboose == 2:
                if epochs % 10 == 0:
                    print("epoch: %d, iter: %d, train_loss: %1.6f, val_loss: %1.6f, best val_loss: %1.6f and corresponding train_loss: %1.6f" % (
                        epochs+1, iter, np.mean(train_loss), val_loss, min_val_loss, min_train_loss))
            # early stopping 的实现
            if early_stopping is not None and early_stopping_counter > early_stopping: # 若验证集损失值连续10个epochs仍然没有实现向下突破，则宣布训练结束了
                if verboose == 2 or verboose == 1:
                    print("early stopping at epoches: %d. iter: %d, best val_loss: %1.6f and corresponding train_loss: %1.6f" % (epochs+1, iter, min_val_loss, min_train_loss))
                break

            self.model.train()
        if best_model is not None:
            if verboose == 2:
                print("min validation loss = %1.7f" % (min_val_loss))
            self.model = best_model

    def loss_func_apart(self, y_true, y_pred, loss_weight = [0.5, 0.5]):
        """
        self defination loss function, I hope to give the different weight of different week forecasting
        ----------------------
        y_true: the dimension should be (batch_size, output_dim)
        y_pred: the same dimension with y_true
        loss_weight: a list that have the length of output_dim
        """
        # criterion = nn.MSELoss(size_average=True)
        if len(loss_weight) != y_true.shape[1]:
            raise Exception('weight and prediction has no same length')
        criterion = nn.SmoothL1Loss(size_average=True)
        loss_ = torch.Tensor([0.0])
        for i in range(len(loss_weight)):
            loss_t = criterion(y_true[:,i:(i+1)], y_pred[:, i:(i+1)])
            loss_ = loss_ + loss_t
        
        
        return loss_
    
    def loss_func_single(self, y_true, y_pred, loss_weight = None):
        # criterion = nn.MSELoss(size_average=True)
        criterion = nn.SmoothL1Loss(size_average=True)
        return criterion(y_true, y_pred)


    def get_val_loss(self, x_val, y_val, loss_weight):
        self.model.eval()
        y_pred = self.model(x_val)
        # print('validation shape: ',y_pred.shape)
        if loss_weight is None:
            val_loss = self.loss_func_single(y_val, y_pred,loss_weight)
        else:
            val_loss = self.loss_func_apart(y_val, y_pred, loss_weight)

        return val_loss

    

    def predict_xy(self, x_data, y_data):
        """
        predict
        --------------
        dataset : (sequence_length, input_dim) * n
        """
        self.model.eval()
        # if not x_data.is_cuda and torch.cuda.is_available():
        #     x_data = x_data.to('cuda')
        #     y_data = y_data.to('cuda')
        with torch.no_grad():
            y_pred = self.model(x_data)
        
        return y_data.data.cpu().numpy(), y_pred.data.cpu().numpy()
    
    def output_model(self):
        return self.model



class LstmWeightedTrain():
    def __init__(self, model):
        self.model = model

    def train(self, train_dataloader, val_datadict, num_epochs, lr = 5e-3, early_stopping = None, loss_weight = None, verboose = 0, upper_weight = None):
        """
        train the model
        ---------------------
        train_dataloader : The Dataloader of training data
        val_dataloader : The Dataloader of validation data
        lr : learning rate, default = 5e-3
        """
        # single stamp
        if loss_weight is None:
            loss_function = self.loss_func_single
        else:
            # multiple stamps
            loss_function = self.loss_func_apart
        # loss_function = nn.SmoothL1Loss()
        
        # if torch.cuda.is_available():
        #     loss_function = loss_function.to('cuda')
        
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr = lr)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1) # 根据epoch降低学习率
        scheduler = ReduceLROnPlateau(optimizer, 'min') # 根据指标降低学习率 # , factor=0.5, patience=10

        # begin training
        iter=0
        best_model = None
        min_val_loss = 1e6
        val_loss = 1e8
        min_train_loss = 1e8
        for epochs in range(num_epochs): 
            train_loss = []
            for i,(batch_x, batch_y) in enumerate(train_dataloader):
                # if torch.cuda.is_available():
                #     batch_x, batch_y = batch_x.to('cuda'), batch_y.to('cuda')
                # print("batch x shape",batch_x.shape)
                pred_y = self.model(batch_x)
                # loss = loss_function(pred_y, batch_y) # 计算损失
                loss = loss_function(batch_y, pred_y, loss_weight, upper_weight)
                train_loss.append(loss.item())
                optimizer.zero_grad()   # 将每次传播时的梯度累积清除
                loss.backward() # 反向传播
                optimizer.step()
                iter = iter + 1

            val_loss = self.get_val_loss(val_datadict['x_data'], val_datadict['y_data'], loss_weight)
            scheduler.step(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_train_loss = np.mean(train_loss)
                best_model = copy.deepcopy(self.model)
                early_stopping_counter = 0
            else:
                early_stopping_counter = early_stopping_counter + 1
            if verboose == 2:
                if epochs % 10 == 0:
                    print("epoch: %d, iter: %d, train_loss: %1.6f, val_loss: %1.6f, best val_loss: %1.6f and corresponding train_loss: %1.6f" % (
                        epochs+1, iter, np.mean(train_loss), val_loss, min_val_loss, min_train_loss))
            # early stopping 的实现
            if early_stopping is not None and early_stopping_counter > early_stopping: # 若验证集损失值连续10个epochs仍然没有实现向下突破，则宣布训练结束了
                if verboose == 2 or verboose == 1:
                    print("early stopping at epoches: %d. iter: %d, best val_loss: %1.6f and corresponding train_loss: %1.6f" % (epochs+1, iter, min_val_loss, min_train_loss))
                break

            self.model.train()
        if best_model is not None:
            if verboose == 2:
                print("min validation loss = %1.7f" % (min_val_loss))
            self.model = best_model

    def loss_func_apart(self, y_true, y_pred, loss_weight = [0.5, 0.5], upper_weight = None):
        """
        self defination loss function, I hope to give the different weight of different week forecasting
        ----------------------
        y_true: the dimension should be (batch_size, output_dim)
        y_pred: the same dimension with y_true
        loss_weight: a list that have the length of output_dim
        """
        # criterion = nn.MSELoss(size_average=True)
        if upper_weight is None:
            if len(loss_weight) != y_true.shape[1]:
                raise Exception('weight and prediction has no same length')
            criterion = nn.SmoothL1Loss(size_average=True)
            loss_ = torch.Tensor([0.0])
            for i in range(len(loss_weight)):
                loss_t = criterion(y_true[:,i:(i+1)], y_pred[:, i:(i+1)])
                loss_ = loss_ + loss_t
            
            
            return loss_
    
    def loss_func_single(self, y_true, y_pred, loss_weight = None, upper_weight = None):
        """
        ------------------
        y_true shape = (batch_size, prediction_horizon)
        """
        # criterion = nn.MSELoss(size_average=True)
        criterion = nn.SmoothL1Loss(size_average=False)
        if upper_weight is None:
            return criterion(y_true, y_pred)
        else:
            # med = y_true.median()
            y_med = torch.full(y_true.shape, y_true.median())
            # print("y_med shape = ", y_med.shape)
            # print(y_med[0:2,0:2])
            y_true_upper_ind = torch.gt(y_true, y_med)
            y_true_upper = y_true.mul(y_true_upper_ind) 
            y_pred_upper = y_pred.mul(y_true_upper_ind)
            y_true_lower_ind = torch.le(y_true, y_med)
            y_true_lower = y_true.mul(y_true_lower_ind) 
            y_pred_lower = y_pred.mul(y_true_lower_ind)

            loss_upper = criterion(y_true_upper, y_pred_upper)
            loss_lower = criterion(y_true_lower, y_pred_lower)
            return upper_weight*loss_upper+(1-upper_weight)*loss_lower


    def get_val_loss(self, x_val, y_val, loss_weight):
        self.model.eval()
        y_pred = self.model(x_val)
        # print('validation shape: ',y_pred.shape)
        if loss_weight is None:
            val_loss = self.loss_func_single(y_val, y_pred,loss_weight)
        else:
            val_loss = self.loss_func_apart(y_val, y_pred, loss_weight)

        return val_loss

    

    def predict_xy(self, x_data, y_data):
        """
        predict
        --------------
        dataset : (sequence_length, input_dim) * n
        """
        self.model.eval()
        # if not x_data.is_cuda and torch.cuda.is_available():
        #     x_data = x_data.to('cuda')
        #     y_data = y_data.to('cuda')
        with torch.no_grad():
            y_pred = self.model(x_data)
        
        return y_data.data.cpu().numpy(), y_pred.data.cpu().numpy()
    
    def output_model(self):
        return self.model
