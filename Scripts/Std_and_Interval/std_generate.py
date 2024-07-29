import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random
import os
import copy
import sys 
from scipy.stats import norm
sys.path.append(".")
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso, LassoCV
from tools.plot import Plot_
from tools.data import DataTool

#################################### read data #################################

model_list = [
     'baseline',
     'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'InTimePlus_v3_nontuning_rolling_v2',
  'LSTM_v3_nontuning_rolling_v2',
  'GRU_v3_nontuning_rolling_v2',
  'TSTPlus_v3_nontuning_rolling',
   'SAE',
  'NBE',
  'AWAE','AWBE'
]

mode = 'test8'

pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path = pwd#os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
model_path = origin_path + '/Results/Point/'
# window_size = 20
for window_size in [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:

     # date_list_o = pd.date_range('2007-11-04','2019-07-14',freq='W')
     date_list_o = pd.date_range('2003-11-16','2019-07-14',freq='W') # for the first day, leave at least  days to caculate the std
     for model_name in model_list:
          print("model = ", model_name)
          path_ = model_path+f'forecast_{model_name}_{mode}.csv'
          df = pd.read_csv(path_)
          df['date'] = pd.to_datetime(df['date'])
          quantile_col = [f'lower_{i}' for i in [2,5,10,20,30,40,50,60,70,80,90]]
          quantile_col.extend([f'upper_{i}' for i in [90, 80, 70, 60,50,40,30,20,10,5,2]])
          for ic in quantile_col:
               df[f'{ic}'] = 0.0
          for wi in range(9):
               df_w = df.loc[df.week_ahead == wi,:]
               date_list = [idate for idate in date_list_o if idate in list(df_w.date.unique())]
               # date_list = list(df_w.date.unique())
               for di in date_list:
                    di = pd.to_datetime(di)
                    df_wd = df_w.loc[(df_w.date>di-timedelta(days = 7*window_size))&((df_w.date<=di)),:]
                    df_wd['ydiff'] = df_wd['point'] - df['true']
                    sd = df_wd.ydiff.std()
                    point = df_wd.loc[df_wd.date == di,'point'].values[0]
                    interval_list = norm.ppf(q=[0.01, 0.025, 0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99], loc=point, scale=sd)
                    for ic in range(len(quantile_col)):
                         df.loc[(df.week_ahead == wi)&(df.date == di),f'{quantile_col[ic]}'] = interval_list[ic] if interval_list[ic]>0 else 0.0

          if not os.path.exists(origin_path+f'/Results/Point/'):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(origin_path+f'/Results/Point/')
          df.to_csv(origin_path + f'/Results/Interval_ydiff_raw/interval{window_size}_{model_name}_{mode}.csv', index = False)