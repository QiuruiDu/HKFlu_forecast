"""
    data read
    Richael-2023/6/16
    --------------------
    read raw data
"""

import pandas as pd 
import numpy as np 
import copy
import os

class DataTool():
    def __init__(self):
        pass

    def _read_csv(self, path):
        df_raw = pd.read_csv(path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace = True)
        return df_raw 
    
    def _read_pq(self, path):
        df_raw = pd.read_parquet(path)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace = True)
        return df_raw 
    
    def _log(self, df, col_):
        df[col_] = np.log(df[col_].values)
        return df
    
    def _log_diff(self, df, col_):
        df[f'{col_}_yesterday'] = df[col_].shift(1)
        df = df.iloc[1:,:]
        df[f'{col_}_log_diff'] = np.log(df[col_]/df[f'{col_}_yesterday'])
        return df
    
    def data_output(self, path, col_list, y_col = 'rate', mode = 'log'):
        data_type = path.split(".", 1)
        if data_type[-1] == 'csv':
            df_raw = self._read_csv(path)
        elif data_type[-1] == 'parquet':
            df_raw = self._read_pq(path)
        else:
            raise RuntimeError('No defined data type! Only parquet or csv data type allowed.')
        df = df_raw[col_list].iloc[1:,:]
        df[f'{y_col}'] = df[f'{y_col}'].astype(np.float64)
        if mode == 'log':
            df = self._log(df, y_col)
            return df
        elif mode == '10log':
            df[f'{y_col}'] = 10*df[f'{y_col}']
            df = self._log(df, y_col)
            return df
        elif mode == 'log_diff':
            df = self._log_diff(df, y_col)
            df.rename(columns = {f'{y_col}':'rate_today'}, inplace = True)
            return df
        elif mode == 'true':
            return df
    
    def origin_re_output(self, df1, left_len = 0, pred_len = 5, exp_mode = True):
        df = copy.deepcopy(df1)
        if exp_mode == True:
            rate_values = df['rate'].values
            df.loc[:,'rate'] = np.exp(rate_values)
        if left_len > 0:
            df = df.iloc[left_len:,:]
        re = pd.DataFrame() 
        for i in range(pred_len):
            # 1-3 列
            if pred_len-i-1> 0:
                re_t = df.iloc[i:-(pred_len-i-1),:][['rate']]
            else:
                re_t = df.iloc[i:,:][['rate']]
            re_t['week_ahead'] = i
            re_t = re_t.reset_index(drop=False, inplace=False)
            re = pd.concat([re, re_t],ignore_index=True)
        
        re.rename(columns={'rate':'true'}, inplace=True)
        return re
    
    def quantile_write(self, re, origin_path, model_name, mode,bootstrap_times = 100):
        if bootstrap_times == 100:
            sample_path = origin_path+f'/Results/Samples/sample_{model_name}_{mode}.csv'
        elif bootstrap_times < 100:
            sample_path = origin_path+f'/Results/Samples/sample_{model_name}_{mode}_{bootstrap_times}.csv'
        re.to_csv(sample_path, index=False)
        quantile_list = [0.01, 0.025]
        quantile_list.extend(list(np.array(range(5,50,5))/100))
        quantile_list.extend(list(np.array(range(55,100,5))/100))
        quantile_list.extend([0.975, 0.99])
        quantile_list = np.array(quantile_list)

        quantile_col = [f'lower_{i}' for i in [2,5,10,20,30,40,50,60,70,80,90]]
        quantile_col.extend([f'upper_{i}' for i in [90, 80, 70, 60,50,40,30,20,10,5,2]])

        re_quan = copy.deepcopy(re[['true','week_ahead','date']])
        for q in range(len(quantile_list)):
            re_quan.loc[:,quantile_col[q]] = re.drop(['true','date','week_ahead'], axis = 1).quantile(quantile_list[q], axis = 1)
        re_quan['point'] = re.drop(['true','date','week_ahead'], axis = 1).quantile(0.5, axis = 1)
        re_quan['point_avg'] = re.drop(['true','date','week_ahead'], axis = 1).mean(axis = 1)
        re_quan['var'] = 'iHosp'
        re_quan['region'] = 'HK'
        re_quan['model'] = model_name
        if bootstrap_times == 100:
            quantile_path = origin_path+f'/Results/Quantiles/quantile_{model_name}_{mode}.csv'
        elif bootstrap_times < 100:
            quantile_path = origin_path+f'/Results/Quantiles/quantile_{model_name}_{mode}_{bootstrap_times}.csv'
        re_quan.to_csv(quantile_path, index  =False)

        print("The sample result has been saved in",sample_path,".")
        print("The quantile result has been saved in",quantile_path,".")


    def sample_to_quantile(self, origin_path, model_name, mode):
        sample_path = origin_path+f'/Results/Samples/sample_{model_name}_{mode}.csv'
        re = pd.read_csv(sample_path)
        bst = re.shape[1]-3
        col_names = ['date','true','week_ahead'] + [f'boot_i' for i in range(bst)]
        re.columns = col_names
        re['date'] = pd.to_datetime(re['date'])
        quantile_list = [0.01, 0.025]
        quantile_list.extend(list(np.array(range(5,50,5))/100))
        quantile_list.extend(list(np.array(range(55,100,5))/100))
        quantile_list.extend([0.975, 0.99])
        quantile_list = np.array(quantile_list)

        quantile_col = [f'lower_{i}' for i in [2,5,10,20,30,40,50,60,70,80,90]]
        quantile_col.extend([f'upper_{i}' for i in [90, 80, 70, 60,50,40,30,20,10,5,2]])
        re_quan = copy.deepcopy(re[['true','week_ahead','date']])
        for q in range(len(quantile_list)):
            re_quan.loc[:,quantile_col[q]] = re.drop(['true','date','week_ahead'], axis = 1).quantile(quantile_list[q], axis = 1)
        re_quan['point'] = re.drop(['true','date','week_ahead'], axis = 1).quantile(0.5, axis = 1)
        re_quan['point_avg'] = re.drop(['true','date','week_ahead'], axis = 1).mean(axis = 1)
        re_quan['var'] = 'iHosp'
        re_quan['region'] = 'HK'
        re_quan['model'] = model_name

        quantile_path = origin_path+f'/Results/Quantiles/quantile_{model_name}_{mode}.csv'
        re_quan.to_csv(quantile_path, index  =False)

        print("The sample result read from: ",sample_path,".")
        print("The quantile result has been saved in",quantile_path,".")

    def point_write(self, re, origin_path, model_name, mode):
        if not os.path.exists(origin_path+f'/Results/Point/'):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(origin_path+f'/Results/Point/')
        forecast_path = origin_path+f'/Results/Point/forecast_{model_name}_{mode}.csv'
        re.to_csv(forecast_path, index=False)
        print("The point forecast result has been saved in ",forecast_path,".")