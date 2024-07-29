"""
LSTM model train and predict
Yearly rolling 
Author: Du
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random
import torch
import joblib
import copy
import os
import sys
sys.path.append(".")
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from model.LstmModel import LstmDataset, LstmModel, LstmWeightedTrain, LstmTrain
from tools.data import DataTool
from tools.plot import Plot_
############################################### data preparation ##########################################
model_name = 'LSTM_v3_nontuning_rolling_v2'
mode = 'test8'
test_start_date = '2005-11-01'
test_end_date = None#'2016-06-30'

start_time = datetime.now()
pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path = pwd#os.path.abspath(os.path.dirname(pwd)+os.path.sep+".") 
print("pwd = ", pwd, ", origin_path = ", origin_path)
path = origin_path + '/Data/data_no_absent.csv'
col_list = ['temp.max','temp.min', 'relative.humidity',
                      'total.rainfall', 'solar.radiation',
                      'monthid', 'weekid', 'rate']
dr = DataTool()
df_o = dr.data_output(path, col_list, mode = 'log')
df = copy.deepcopy(df_o)
# df = df.loc[df.index>pd.to_datetime('2009-12-31'),:]
# print("finally, start date = ", df.index.min())

############################################### modeling ##########################################
study = joblib.load("./model_hyperparam/Separate_LSTM.pkl") #load
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

seq_length1 = trial.params['seq_length1']
mid_dim1 = trial.params['mid_dim1']
hidden_layers1 = trial.params['hidden_layers1']
dropout_rate1 = trial.params['dropout_rate1']
lr1 = trial.params["lr1"]
batch_size1 = trial.params["batch_size1"]
# upper_weight1 = trial.params['upper_weight1']

seq_length2 = trial.params['seq_length2']
mid_dim2 = trial.params['mid_dim2']
hidden_layers2 = trial.params['hidden_layers2']
dropout_rate2 = trial.params['dropout_rate2']
lr2 = trial.params["lr2"]
batch_size2 = trial.params["batch_size2"]


pred_stamp = 9

def one_bootstrap(i, df, test_size, total_pred_horizon = 5):
    torch.manual_seed(i)
    random.seed(i)
    np.random.seed(i)
    ## week 1 : model building
    pred_stamp = 1
    input_dim = df.shape[1]
    output_dim = pred_stamp
    data_deal1 = LstmDataset(sequence_length=seq_length1, 
                        batch_size=batch_size1, 
                        pred_stamp=pred_stamp)
    train_dataloader1, val_datadict1, test_datadict1 = data_deal1.get_train_val_test_dataset(copy.deepcopy(df), test_size = test_size, 
                                                                                          sample_rate = None)
    rate_max1, rate_min1 = data_deal1.get_rate_scaler()

    x_test1, y_test1 = test_datadict1['x_data'], test_datadict1['y_data']
    # print("--- week 1 split ----")
    # print(x_test1.shape, y_test1.shape)
    model1 = LstmModel(input_dim, output_dim, sequence_dim=seq_length1, 
                  mid_dim=mid_dim1, hidden_lstm_layers=hidden_layers1, dropout_rate=dropout_rate1, num_directions = 1, MCDropout = None)
    model_train1 = LstmTrain(model1)
    model_train1.train(train_dataloader1, val_datadict1, num_epochs = 60, lr=lr1, early_stopping = None, verboose=2)
    # get predict result
    _, y_test_pred1 = model_train1.predict_xy(x_test1, y_test1)

    ## week after 1
    input_dim = df.shape[1]
    pred_stamp = total_pred_horizon - 1
    output_dim = pred_stamp
    data_deal4 = LstmDataset(sequence_length=seq_length2, 
                            batch_size=batch_size2, 
                            pred_stamp=pred_stamp)
    train_dataloader4, val_datadict4, test_datadict4 = data_deal4.get_train_val_test_dataset(copy.deepcopy(df), test_size = test_size, 
                                                                                          sample_rate = None)
    rate_max4, rate_min4 = data_deal4.get_rate_scaler()
    model4 = LstmModel(input_dim, output_dim, sequence_dim=seq_length2, 
                    mid_dim=mid_dim2, hidden_lstm_layers=hidden_layers2, dropout_rate=dropout_rate2, num_directions = 1, MCDropout = None)
    model_train4 = LstmTrain(model4)
    model_train4.train(train_dataloader4, val_datadict4, num_epochs = 60, lr=lr2, early_stopping = None, verboose=2)
    # delay one day
    x_test4, y_test4 = test_datadict4['x_data'][1:,:,:], test_datadict4['y_data'][1:,:]
    # 替换rate为预估值
    for i in range(x_test4.shape[0]):
        x_test4[i,-1,-1] = torch.from_numpy(y_test_pred1[i]) 
    _, y_test_pred4 = model_train4.predict_xy(x_test4, y_test4)

    # 逆标准化
    y_test_pred1 = y_test_pred1*(rate_max1 - rate_min1)+rate_min1
    y_test_pred1 = np.exp(y_test_pred1)
    y_test_pred4 = y_test_pred4*(rate_max4 - rate_min4)+rate_min4
    y_test_pred4 = np.exp(y_test_pred4)

    return y_test_pred1, y_test_pred4

def one_rolling(df, test_start, test_end, pred_horizon, exp_mode = True, bootstrap_times = 100, random_state = None):
    df_t = copy.deepcopy(df.loc[df.index <= pd.to_datetime(test_end),:])
    test_size = df_t.loc[df_t.index > test_start,:].shape[0]
    # print("---------------------------- test_start = ", test_start, ', test_end = ',test_end,', df_t shape = ',df_t.shape,', test_size = ', test_size)
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start,:])
    re_test = dr.origin_re_output(df_test, left_len=0, pred_len = pred_horizon, exp_mode=exp_mode)

    for bst in range(bootstrap_times):
        print("-------------------------- bootstrap = ", bst, " ----------------------------------")
        seed_ = random_state if random_state is not None else bst
        y_pred_i1,y_pred_i4 = one_bootstrap(i=seed_, df = df_t, test_size = test_size, total_pred_horizon=pred_stamp)
        print("pred4_shape = ",y_pred_i4.shape)
        re_test_pred = pd.DataFrame(y_pred_i1[0:y_pred_i4.shape[0]], columns = [f'boot_{bst}'])
        re_test_pred['week_ahead'] = 0
        for i in range(pred_stamp-1):
            re_t = pd.DataFrame(y_pred_i4[:,i], columns = [f'boot_{bst}'])
            re_t['week_ahead'] = i+1
            re_test_pred = pd.concat([re_test_pred, re_t], ignore_index=True)
        re_test = pd.concat([re_test, re_test_pred[[f'boot_{bst}']]], axis=1)
    
    return re_test

test_start_date = pd.to_datetime('2003-11-01')
max_year_range = 20
year_step = 1
rolling_dates = [test_start_date + timedelta(days = 52 * 7 * i) for i in range(0,max_year_range,year_step) if test_start_date + timedelta(days = 52 * 7 * i) < (pd.to_datetime('2019-07-14')-timedelta(days = (pred_stamp-1) * 7))]
rolling_dates.append(pd.to_datetime('2019-07-14')-timedelta(days = (pred_stamp-1) * 7))

df_test_total = copy.deepcopy(df.loc[df.index > test_start_date,:])
re_test_total = pd.DataFrame()
# dr.origin_re_output(df_test, left_len=0, pred_len = pred_horizon, exp_mode=exp_mode)

for i_date in range(len(rolling_dates)-1):
    test_start, test_end = rolling_dates[i_date], rolling_dates[i_date+1]+timedelta(days = (pred_stamp-1) * 7)
    print("----------------------------- i_date = ", i_date,", test_start = ", test_start, ', test_end = ',test_end)
    re_test = one_rolling(df, test_start, test_end, pred_stamp, bootstrap_times = 1, random_state = i_date)
    re_test_total = pd.concat([re_test_total, re_test], axis=0)

############################################### save ##########################################
re_test_total.rename(columns={'boot_0':'point'}, inplace=True)
re_test_total['point_avg'] = re_test_total['point']
# dr.point_write(re = re_test_total, origin_path=origin_path, mode = mode, model_name=model_name)

end_time = datetime.now()
print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
print("The running time totally =", (end_time-start_time).seconds," seconds.")      