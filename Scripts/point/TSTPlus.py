"""
TsTPlus model train and predict, based on tsai package
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
import copy
import os
import sys
sys.path.append(".")
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from model.LstmModel import LstmDataset
from tools.data import DataTool
from tools.plot import Plot_
import joblib
import optuna
from tsai.all import TimeSplitter, TSForecasting, TSStandardize, TSForecaster, mse, ShowGraph
from fastai.callback.tracker import SaveModelCallback
############################################### data preparation ##########################################
model_name = 'TSTPlus_v3_nontuning_rolling_v2'
mode = 'test8'
test_start_date = '2005-11-01'
test_end_date = None

start_time = datetime.now()
pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path = pwd
print("pwd = ", pwd, ", origin_path = ", origin_path)
path = origin_path + '/Data/data_no_absent.csv'
col_list = ['temp.max','temp.min', 'relative.humidity',
                      'total.rainfall', 'solar.radiation',
                      'monthid', 'weekid', 'rate']
dr = DataTool()
df_o = dr.data_output(path, col_list, mode = 'log')
df = copy.deepcopy(df_o)

############################################### modeling ##########################################
study = joblib.load("./model_hyperparam/Separate_TSTPlus.pkl") #load
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

head = trial.params['n_heads']
d_model=trial.params['d_model']
depth = trial.params['n_layers']
dropout= trial.params['dropout']
att_dropout=trial.params['attn_dropout']
learning_rate = trial.params['learning_rate']
b_s=trial.params['batch_size']

seq_length = 14
batch_size = 0
pred_stamp = 9

def one_bootstrap(i, df, test_size, total_pred_horizon):
    torch.manual_seed(i)
    random.seed(i)
    np.random.seed(i)
    print("---- test size = ", test_size, ", total prediction horizon = ", total_pred_horizon)
    # week 1
    pred_stamp = 1
    data_deal1 = LstmDataset(sequence_length=seq_length, 
                        batch_size=batch_size, 
                        pred_stamp=pred_stamp)
    train_datadict1, _, test_datadict1 = data_deal1.get_train_val_test_dataset(df, test_size = test_size, sample_rate = None, validation=False)
    rate_max1, rate_min1 = data_deal1.get_rate_scaler()

    x_train1, y_train1= train_datadict1['x_data'].permute(0,2,1), train_datadict1['y_data']
    x_test1, y_test1= test_datadict1['x_data'].permute(0,2,1), test_datadict1['y_data']
    print("11111 x shape=", x_train1.shape,"y shape", y_train1.shape)
    print("x shape=", x_train1.shape,"y shape", y_train1.shape)
    val_size = np.random.choice(np.array(range(78,104)),size = 1)[0]
    splits1 = TimeSplitter(valid_size=val_size, fcst_horizon=pred_stamp, show_plot=False)(y_train1) 
    print("--- week 1 split ----")
    print(x_train1[splits1[0]].shape, y_train1[splits1[0]].shape)
    print(x_train1[splits1[1]].shape, y_train1[splits1[1]].shape)
    print(x_test1.shape, y_test1.shape)
    fcst1 = TSForecaster(x_train1, y_train1, splits=splits1, bs=b_s, arch="TSTPlus",
                                metrics=mse, train_metrics=True, seed=i, verbose=False)
    # 
    fcst1.fit_one_cycle(50,lr_max=learning_rate,cbs=[SaveModelCallback(monitor='valid_loss')])
    raw_preds, target, preds1 = fcst1.get_X_preds(x_test1)

    # week after 1
    pred_stamp = total_pred_horizon - 1
    data_deal2 = LstmDataset(sequence_length=seq_length, 
                        batch_size=batch_size, 
                        pred_stamp=pred_stamp)
    train_datadict2, _, test_datadict2 = data_deal2.get_train_val_test_dataset(df, test_size = test_size, sample_rate = None, validation=False)
    rate_max2, rate_min2 = data_deal2.get_rate_scaler()

    x_train2, y_train2= train_datadict2['x_data'].permute(0,2,1), train_datadict2['y_data']
    x_test2, y_test2= test_datadict2['x_data'].permute(0,2,1), test_datadict2['y_data']
    x_test2, y_test2 = x_test2[1:,:,:], y_test2[1:,:]
    print("11111 x shape=", x_train2.shape,"y shape", y_train2.shape)
    print("x shape=", x_train2.shape,"y shape", y_train2.shape)
    splits2 = TimeSplitter(valid_size=val_size, fcst_horizon=pred_stamp, show_plot=False)(y_train2) 
    print("--- week after 1, split ----")
    print(x_train2[splits2[0]].shape, y_train2[splits2[0]].shape)
    print(x_train2[splits2[1]].shape, y_train2[splits2[1]].shape)
    print(x_test2.shape, y_test2.shape)
    # replace 'rate' to the estimated one
    for i in range(x_test2.shape[0]):
        x_test2[i,-1,-1] = preds1[i]
    print(x_test2.shape, y_test2.shape)
    fcst2 = TSForecaster(x_train2, y_train2, splits=splits2, bs=b_s, arch="TSTPlus",
                                metrics=mse, train_metrics=True, seed=i, verbose=False)
    fcst2.fit_one_cycle(50,lr_max=learning_rate, cbs=[SaveModelCallback(monitor='valid_loss')])
    raw_preds, target, preds2 = fcst2.get_X_preds(x_test2)

    preds1 = preds1*(rate_max1 -  rate_min1)+rate_min1
    preds1 = np.exp(preds1)
    preds2 = preds2*(rate_max2 - rate_min2)+rate_min2
    preds2 = np.exp(preds2)
    print("pred shape = ", preds1.shape, preds2.shape)
    return preds1, preds2

def one_rolling(df, test_start, test_end, pred_horizon, exp_mode = True, bootstrap_times = 100, random_state = None):
    df_t = copy.deepcopy(df.loc[df.index<=pd.to_datetime(test_end),:])
    test_size = df_t.loc[df_t.index > test_start,:].shape[0]
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start,:])
    re_test = dr.origin_re_output(df_test, left_len=0, pred_len = pred_horizon, exp_mode=exp_mode)

    for bst in range(bootstrap_times):
        print("-------------------------- bootstrap = ", bst, " ----------------------------------")
        seed_ = random_state if random_state is not None else bst
        y_pred_i1,y_pred_i4 = one_bootstrap(seed_, df_t, test_size, total_pred_horizon=pred_stamp)
        
        re_test_pred = pd.DataFrame(y_pred_i1[0:y_pred_i4.shape[0]], columns = [f'boot_{bst}'])
        re_test_pred['week_ahead'] = 0
        for i in range(pred_horizon-1):
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

df_test_total = copy.deepcopy(df.loc[df.index >= test_start_date,:])
re_test_total = pd.DataFrame()

for i_date in range(len(rolling_dates)-1):
    test_start, test_end = rolling_dates[i_date], rolling_dates[i_date+1]+timedelta(days = (pred_stamp-1) * 7)
    print("----------------------------- i_date = ", i_date,", test_start = ", test_start, ', test_end = ',test_end)
    re_test = one_rolling(df, test_start, test_end, pred_stamp, bootstrap_times = 1, random_state=i_date)
    re_test_total = pd.concat([re_test_total, re_test], axis=0)

############################################### save ##########################################
re_test_total.rename(columns={'boot_0':'point'}, inplace=True)
re_test_total['point_avg'] = re_test_total['point']
dr.point_write(re = re_test_total, origin_path=origin_path, mode = mode, model_name=model_name)

end_time = datetime.now()
print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
print("The running time totally =", (end_time-start_time).seconds," seconds.")      