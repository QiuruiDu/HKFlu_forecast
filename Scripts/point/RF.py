"""
RF model train and predict
Yearly rolling 
Author: Du
"""

## import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random
import copy
import os
import sys
sys.path.append(".")
from datetime import datetime,timedelta
from model.MLModel import RFmodel, MLDataset
from tools.data import DataTool
from tools.plot import Plot_
############################################### preparation ##########################################
model_name = 'rf_rolling'
mode = 'test8'
test_start_date = '2005-11-01'
test_end_date = None#'2016-06-30'

start_time = datetime.now()
pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path = pwd#os.path.abspath(os.path.dirname(pwd)+os.path.sep+".") 
print("pwd = ", pwd, ", origin_path = ", origin_path)
path = origin_path + '/Data/rolling_data_before_covid.parquet'
col_list = ['temp.max','temp.min', 'relative.humidity',
                      'total.rainfall', 'solar.radiation',
                      'monthid', 'weekid', 'rate', 'date_analysis']
dr = DataTool()
df_o = dr.data_output(path, col_list, mode = 'log')
df_o['date_analysis'] = pd.to_datetime(df_o['date_analysis'])
df = copy.deepcopy(df_o)
print("finally, start date = ", df.index.min())

df_all = df.loc[df['date_analysis'] == max(df['date_analysis']),].drop('date_analysis', axis = 1)

############################################### fit and predict ##########################################
pred_stamp = 9
exp_mode = True
cov_list = ['temp.max','temp.min', 'relative.humidity',
                      'total.rainfall', 'solar.radiation']

def cv_param(df, random_state, test_start, pred_horizon):
    """ 
    first, update the parameter
    second, fit model and make prediction
    -----------------------------------------
    """
    np.random.seed(random_state)
    random.seed(random_state)
    train_analysis_end = max([date_ for date_ in df.date_analysis.unique() if date_ <= test_start])
    # print("train_analysis_end = ", train_analysis_end)
    df_train = copy.deepcopy(df.loc[df.date_analysis == train_analysis_end,:])
    df_train = df_train.drop('date_analysis', axis = 1)
    # print("df_train.shape = ", df_train.shape)
    myrf = RFmodel()
    data_deal = MLDataset()
    # 1. -------------------- get the best parameter ------------
    ### train data prepare
    train_datadict, _ = data_deal.get_train_data(df_train, max_rate_lag=14, cov_list = cov_list, max_cov_lag=14, pred_horizon = pred_horizon, validation=False)
    x_train, y_train = train_datadict['x_data'], train_datadict['y_data']
    # print(x_train.shape, y_train.shape)
    params = {'n_estimators': [20,40,80,160,200],
          'max_depth': [2,4,8,16],
          'random_state':[random_state]
          }
    myrf.CV_train_(x_train, y_train, fold_num = 5, param_dict = params)
    # myrf.fit_(x_train, y_train)

    return myrf, data_deal, train_datadict

def fit_and_predict(df, model, data_deal, train_datadict, pred_start, pred_end, pred_horizon, random_state):
    np.random.seed(random_state)
    random.seed(random_state)
    # re fit with sampling 
    x_train, y_train = train_datadict['x_data'].values, train_datadict['y_data'].values
    train_ind = np.random.choice(x_train.shape[0], size=x_train.shape[0], replace=True)
    x_train,y_train = x_train[train_ind,:],y_train[train_ind,:]
    model.fit_(x_train, y_train, random_state)
    # test data preparation
    max_lag = data_deal.output_max_lag()
    test_start_date = pred_start+timedelta(days=-int(7*max_lag))
    test_end_date = pred_end#+timedelta(days = (pred_horizon-1) * 7)
    # print("for generating predict data, start date = ",test_start_date,", end date = ",test_end_date )
    df_test_o = df.loc[(df.index >= test_start_date) & (df.index <= test_end_date),:]
    # print("df_test_origin shape = ", df_test_o.shape, ", min date = ", df_test_o.index.min(),", max_date = ", df_test_o.index.max())
    df_test= data_deal.get_test_data(df_test_o)
    # print("for df_test, min = ", df_test.index.min(),", max = ",df_test.index.max())
    x_test, y_test = df_test.iloc[:,0:-pred_horizon].values, df_test.iloc[:,-pred_horizon:].values
    # print(x_test.shape, y_test.shape)
    # make prediction
    y_test_hat = model.predict_(x_test)
    rate_max, rate_min = data_deal.output_rate_scaler()
    y_test, y_test_hat = y_test*(rate_max - rate_min)+rate_min , y_test_hat*(rate_max - rate_min)+rate_min 
    y_test, y_test_hat = np.exp(y_test), np.exp(y_test_hat)
    return y_test, y_test_hat


test_start_date = pd.to_datetime('2003-11-01')
max_year_range = 20
year_step = 1
rolling_dates = [test_start_date + 
                 timedelta(days = 52 * 7 * i) for i in range(0,max_year_range,year_step) if test_start_date + timedelta(days = 52 * 7 * i) < (pd.to_datetime('2019-07-14')-timedelta(days = (pred_stamp-1) * 7))]
rolling_dates.append(pd.to_datetime('2019-07-14')-timedelta(days = (pred_stamp-1) * 7))

df_test_total = copy.deepcopy(df.loc[df.index >= test_start_date,:])
re_test_total = pd.DataFrame()
# dr.origin_re_output(df_test, left_len=0, pred_len = pred_horizon, exp_mode=exp_mode)

bootstrap_times = 1
for i_date in range(len(rolling_dates)-1):
    print("this prediction interval = ", rolling_dates[i_date], rolling_dates[i_date+1])
    test_start, test_end = rolling_dates[i_date], rolling_dates[i_date+1]+timedelta(days = (pred_stamp-1) * 7)
    df_t = copy.deepcopy(df_all.loc[df_all.index<=pd.to_datetime(test_end),:])
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start,:])
    re_test = dr.origin_re_output(df_test, left_len=0, pred_len = pred_stamp, exp_mode=exp_mode)
    print("----------------------------- i_date = ", i_date,", test_start = ", test_start, ', test_end = ',test_end)
    mymodel, data_deal, train_datadict = cv_param(df, random_state = i_date, test_start = test_start, pred_horizon = pred_stamp)
    for bst in range(bootstrap_times):
        print("-------------------------- bootstrap = ", bst, " ----------------------------------")
        y_true,y_pred = fit_and_predict(df_all, mymodel, data_deal, train_datadict, test_start, test_end, pred_stamp, random_state=i_date)
        re_test_pred = pd.DataFrame()
        for i in range(pred_stamp):
            re_t = pd.DataFrame(y_pred[:,i], columns = [f'boot_{bst}'])
            re_t['week_ahead'] = i
            re_test_pred = pd.concat([re_test_pred, re_t], ignore_index=True)
        re_test = pd.concat([re_test, re_test_pred[[f'boot_{bst}']]], axis=1)
    re_test_total = pd.concat([re_test_total, re_test], axis=0)

############################################### save ##########################################
re_test_total.rename(columns={'boot_0':'point'}, inplace=True)
re_test_total['point_avg'] = re_test_total['point']
# dr.point_write(re = re_test_total, origin_path=origin_path, mode = mode, model_name=model_name)

end_time = datetime.now()
print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
print("The running time totally =", (end_time-start_time).seconds," seconds.") 