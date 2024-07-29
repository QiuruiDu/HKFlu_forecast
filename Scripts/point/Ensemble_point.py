"""

    the code of all ensemble models
    Author: QR Du

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso, LassoCV
import random
import os
from copy import deepcopy
import sys 
sys.path.append(".")
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso, LassoCV
from tools.plot import Plot_
from tools.data import DataTool

class AVG_ENSEMBLE():
    def __init__(self, model_list, forecasting_mode, origin_path, rolling_start_date) -> None:
        super(AVG_ENSEMBLE, self).__init__()
        self.model_list = model_list
        self.forecasting_mode = forecasting_mode
        if self.forecasting_mode == 'test8':
            self.max_pred_horizon = 8
        else:
            self.max_pred_horizon = 4

        self.rolling_start_date = rolling_start_date
        self.origin_path = origin_path
        self.model_path = self.origin_path + '/Results/Point/'

    def get_origin_date(self, df):
        d1 = df['date'] 
        td = df['week_ahead']
        return d1+timedelta(days = -td * 7)
    
    def get_bst_data(self, df_analysis, point_col):
        df_analysis_bst = deepcopy(df_analysis[['week_ahead','date',f'{point_col}','model']])
        df_analysis_bst['date_origin'] = df_analysis_bst.apply(self.get_origin_date, axis = 1)
        df_analysis_bst = df_analysis_bst.loc[df_analysis_bst['date_origin']>= self.rolling_start_date,:]
        
        df_analysis_bst['date'] = pd.to_datetime(df_analysis_bst['date'])
        df_analysis_bst = pd.merge(df_analysis_bst, df_analysis.loc[df_analysis['model'] == self.model_list[0],['date','week_ahead','true']], on=['date','week_ahead'], how='inner')
        #
        
        return df_analysis_bst

    def compute_rmse(self, df):
        df1 = deepcopy(df)
        pred_horizons = df1.week_ahead.unique()
        rmse_list = []
        for i in pred_horizons:
            df_t = df1.loc[df1.week_ahead == i, :]
            rmse_t = np.sqrt(np.mean((df_t['true'].values - df_t['point_avg'].values)**2))
            rmse_list.append(rmse_t)
        return np.mean(np.array(rmse_list))


    def compute_weighted_rmse(self, df, mode = 'Newton', lambda_ = 0.1):
        df1 = deepcopy(df)
        if mode == 'Newton':
            df1['rank'] = df1.groupby('week_ahead').date.rank(ascending = False)
            df1['decay_coef'] = df1[['rank']].applymap(lambda x: np.exp(-lambda_*x))
            df1.loc[df1.decay_coef < 1e-3, 'decay_coef'] = 1e-3
            df1['decay_weight'] = df1['decay_coef']/np.sum(df1['decay_coef'].values)
    
        pred_horizons = df1.week_ahead.unique()
        rmse_list = []
        for i in pred_horizons:
            df_t = df1.loc[df1.week_ahead == i, :]
            df_rmse = ((df_t['true'].values - df_t['point_avg'].values)**2)*(df_t['decay_weight'].values)
            rmse_t = np.sqrt(np.mean(df_rmse))
            rmse_list.append(rmse_t)
        return np.mean(np.array(rmse_list))
    
    def SAE(self):
        model_name = 'SAE'

        df_test = pd.DataFrame()
        m = self.model_list[0]
        path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
        df_mt_o = pd.read_csv(path_mt)
        df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
        df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
        df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
        df_mt_o['model'] = m
        df_test = pd.concat([df_test, df_mt_o])
        for m in self.model_list[1:]:
            path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
            df_mt_o = pd.read_csv(path_mt)
            df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
            df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
            df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
            df_mt_o['model'] = m
            df_test = pd.concat([df_test, df_mt_o])

        df_full = deepcopy(df_test)
        df_full['date_origin'] = df_full.apply(self.get_origin_date, axis = 1)

        start_time = datetime.now()

        date_analysises = df_full.date.unique()
        date_analysises = date_analysises[(date_analysises >= pd.to_datetime('2004-07-01'))&(
            date_analysises <= pd.to_datetime('2019-07-14'))]#-timedelta(days = 7*(max_pred_horizon))
        
        re = pd.DataFrame()
        for l in range(len(date_analysises)):
            print("-------------------------------------------  ", date_analysises[l], '  ----------------------------------------------------')
            df_full_date_analysis = df_full.loc[df_full.date < pd.to_datetime(date_analysises[l]),:]
            re_t = df_full.loc[df_full['date'] == date_analysises[l],:]
            print(" ---- model_length =", len(re_t.model.unique()))
            if len(re_t.model.unique()) == len(self.model_list):
                # get coef
                col_name = 'point_avg'
                df_analysis_bst = self.get_bst_data(df_full_date_analysis, col_name)
                res_rmse = df_analysis_bst.groupby('model').apply(self.compute_rmse)
                
                rmse_top2 = res_rmse.sort_values().head(2)
                ensemble_list = rmse_top2.index.to_list()
                print(ensemble_list)

                df_model_count = re_t[['week_ahead','model']].groupby('week_ahead').count()
                re_t1 = re_t.loc[re_t.model.isin(ensemble_list),:].groupby(['date','week_ahead']).mean().reset_index(drop = False)
                re = pd.concat([re, re_t1])


        print(re.columns)

        end_time = datetime.now()
        print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
        print("The running time totally =", (end_time-start_time).seconds," seconds.") 

        re1 = deepcopy(re)
        re1['var'] = 'iHosp'
        re1['model'] = model_name
        re1['region'] = 'HK'
        
        re_final = deepcopy(re1[df_mt_o.columns]) 
        re_final['date'] = pd.to_datetime(re_final['date'])

        re_final.to_csv(f'{self.origin_path}/Results/Point/forecast_{model_name}_{self.forecasting_mode}.csv', index = False)


    def AWAE(self, lambda_):
        model_name = 'AWAE'
        df_test = pd.DataFrame()
        m = self.model_list[0]
        path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
        df_mt_o = pd.read_csv(path_mt)
        df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
        df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
        df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
        df_mt_o['model'] = m
        df_test = pd.concat([df_test, df_mt_o])
        for m in self.model_list[1:]:
            path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
            df_mt_o = pd.read_csv(path_mt)
            df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
            df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
            df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
            df_mt_o['model'] = m
            df_test = pd.concat([df_test, df_mt_o])

        df_full = deepcopy(df_test)
        df_full['date_origin'] = df_full.apply(self.get_origin_date, axis = 1)

        start_time = datetime.now()

        date_analysises = df_full.date.unique()
        date_analysises = date_analysises[(date_analysises >= pd.to_datetime('2004-07-01'))&(date_analysises <= pd.to_datetime('2019-07-14'))] 
        mu = 0
        
        re = pd.DataFrame()
        for l in range(len(date_analysises)):
            # l = 1
            print("-------------------------------------------  pred_date", date_analysises[l], '  ----------------------------------------------------')
            df_full_date_analysis = df_full.loc[df_full.date < pd.to_datetime(date_analysises[l]),:] # historical for train
            re_t = df_full.loc[df_full['date'] == date_analysises[l],:] # make forecast for the certain day 
            print(" ---- model_length =", len(re_t.model.unique()))
            re_t['coef_equal'] = 0.0
            re_t['decay_coef'] = 0.0
            # get coef
            for wi in [np.array([0]),np.array(range(1, self.max_pred_horizon+1))]:
                print("-------------- week = ", wi, " ---------------")
                if len(re_t.loc[re_t.week_ahead.isin(wi),'model'].unique()) == len(self.model_list):
                    df_full_date_analysis_w = df_full_date_analysis.loc[df_full_date_analysis.week_ahead.isin(wi),:]
                    df_test_t_wi = re_t.loc[re_t.week_ahead.isin(wi),:]
                    # get coef
                    col_name = 'point_avg'
                    df_analysis_bst = self.get_bst_data(df_full_date_analysis_w, col_name)
                    res_rmse = df_analysis_bst.groupby('model').apply(self.compute_weighted_rmse, mode='Newton', lambda_ = lambda_)
                    
                    rmse_top2 = res_rmse.sort_values().head(2)
                    ensemble_list = rmse_top2.index.to_list()

                    print(ensemble_list)

                    df_model_count = df_test_t_wi.groupby('model').count().reset_index()
                    model_benchmark = df_model_count.loc[df_model_count.date == df_model_count.date.min(),'model'].values[0]
                    wi1 = df_test_t_wi.loc[df_test_t_wi.model == model_benchmark,'week_ahead'].unique()
                    
                    re_t1 = re_t.loc[re_t.week_ahead.isin(wi1),:]
                    re_t1 = re_t1.loc[re_t1.model.isin(ensemble_list),:].groupby(['date','week_ahead']).mean().reset_index(drop = False)
                    re = pd.concat([re, re_t1])
                else:
                    print("Not exist.")
                    re_t = re_t.drop(re_t[re_t.week_ahead.isin(wi)].index)
                    

        print(re.columns)

        end_time = datetime.now()
        print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
        print("The running time totally =", (end_time-start_time).seconds," seconds.") 

        re1 = deepcopy(re)
        re1['var'] = 'iHosp'
        re1['model'] = model_name
        re1['region'] = 'HK'
        
        re_final = deepcopy(re1[df_mt_o.columns]) 
        re_final['date'] = pd.to_datetime(re_final['date'])

        re_final.to_csv(f'{self.origin_path}/Results/Point/forecast_{model_name}_{self.forecasting_mode}.csv', index = False)



class BLENDING():
    def __init__(self, model_list, forecasting_mode, origin_path, rolling_start_date) -> None:
        super(BLENDING, self).__init__()
        self.model_list = model_list
        self.forecasting_mode = forecasting_mode
        if self.forecasting_mode == 'test8':
            self.max_pred_horizon = 8
        else:
            self.max_pred_horizon = 4

        self.rolling_start_date = rolling_start_date
        self.origin_path = origin_path
        self.model_path = self.origin_path + '/Results/Point/'


    def get_origin_date(self, df):
        d1 = df['date'] 
        td = df['week_ahead']
        return d1+timedelta(days = -td * 7)
    
    def get_bst_data(self, df_analysis, model_list, point_col):
        df_analysis_bst = deepcopy(df_analysis[['week_ahead','date',f'{point_col}','model']])
        df_analysis_bst['date_origin'] = df_analysis_bst.apply(self.get_origin_date, axis = 1)
        df_analysis_bst = df_analysis_bst.loc[df_analysis_bst['date_origin']>= self.rolling_start_date,:]
        df_analysis_bst = df_analysis_bst.pivot_table(index=['week_ahead','date'],columns='model',values=f'{point_col}').reset_index(drop = False)
        df_analysis_bst['date'] = pd.to_datetime(df_analysis_bst['date'])
        df_analysis_bst = pd.merge(df_analysis_bst, df_analysis.loc[df_analysis['model'] == model_list[0],['date','week_ahead','true']], on=['date','week_ahead'], how='inner')
        df_analysis_bst = df_analysis_bst.dropna(axis = 0)
        
        return df_analysis_bst
    

    def get_bst_coef_nbe(self, df_bst_analysis, model_list):
        seed_ = 2023
        np.random.seed(seed_)
        random.seed(seed_)
        
        alpha_range = np.array(range(1,30))/10 # define alpha range 
        ############################# normal lasso
        df_bst_p1 = deepcopy(df_bst_analysis[model_list+['true']])
        # LassoCV
        lasso_ = LassoCV(random_state = 2023, alphas=alpha_range, cv=5, fit_intercept = False).fit(df_bst_p1[model_list].values,df_bst_p1['true'])
        # best alpha
        best_alpha = lasso_.alpha_ 
        print("best_alpha = ", best_alpha)

        lasso1 = Lasso(random_state = 2023, alpha = best_alpha, fit_intercept = False) # 默认alpha =1 
        lasso1.fit(df_bst_p1[model_list].values,df_bst_p1['true'])
        for ii in range(len(model_list)):
            print("-- ",model_list[ii], " : ",lasso1.coef_[ii])
        print('---- val score = ', lasso1.score(df_bst_p1[model_list].values,df_bst_p1['true']))

        ############################# weighted lasso
        df_bst_p2 = deepcopy(df_bst_analysis[model_list+['true','week_ahead','date']])
        df_bst_p2['rank'] = df_bst_p2.groupby('week_ahead').date.rank(ascending = False)
        df_bst_p2['decay_coef'] = df_bst_p2[['rank']].applymap(lambda x: np.exp(-0.1*x))
        lasso_ = LassoCV(random_state = 2023, alphas=alpha_range,cv=5, fit_intercept = False).fit(df_bst_p2[model_list].values, df_bst_p2['true'], sample_weight = df_bst_p2['decay_coef'])
        best_alpha = lasso_.alpha_ 
        print("best_alpha = ", best_alpha)
        lasso2 = Lasso(random_state = 2023, alpha = best_alpha, fit_intercept = False) # 默认alpha =1 
        lasso2.fit(df_bst_p2[model_list].values,df_bst_p2['true'], sample_weight = df_bst_p2['decay_coef'])
        for ii in range(len(model_list)):
            print("-- ",model_list[ii], " : ",lasso2.coef_[ii])
        print('---- val score = ', lasso2.score(df_bst_p2[model_list].values,df_bst_p2['true'],sample_weight=df_bst_p2['decay_coef']))

        return lasso1, lasso2
    
    def NBE(self):
        model_name = 'NBE'

        df_test = pd.DataFrame()
        m = self.model_list[0]
        path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
        df_mt_o = pd.read_csv(path_mt)
        df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
        df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
        df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
        df_mt_o['model'] = m
        df_test = pd.concat([df_test, df_mt_o])
        for m in self.model_list[1:]:
            path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
            df_mt_o = pd.read_csv(path_mt)
            df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
            df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
            df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
            df_mt_o['model'] = m
            df_test = pd.concat([df_test, df_mt_o])

        df_full = deepcopy(df_test)

        start_time = datetime.now()

        date_analysises = df_full.date.unique()
        date_analysises = date_analysises[(date_analysises >= pd.to_datetime('2004-07-01'))&(date_analysises <= pd.to_datetime('2019-07-14'))] #
        mu = 1
        
        re = pd.DataFrame()
        for l in range(len(date_analysises)):
            # l = 1
            print("-------------------------------------------  pred_date", date_analysises[l], '  ----------------------------------------------------')
            df_full_date_analysis = df_full.loc[df_full.date < pd.to_datetime(date_analysises[l]),:]
            df_test_t = df_full.loc[df_full['date'] == date_analysises[l],:]
            
            re_t = df_test_t[['date','week_ahead','true']].groupby(['date','week_ahead']).mean().reset_index()
            re_t['pred_equal'] = 0.0
            re_t['pred_decay'] = 0.0
            # get coef
            for wi in [np.array([0]),np.array(range(1, self.max_pred_horizon+1))]:
                print("-------------- week = ", wi, " ---------------")
                if len(df_test_t.loc[df_test_t.week_ahead.isin(wi),'model'].unique()) == len(self.model_list):
                    df_full_date_analysis_w = df_full_date_analysis.loc[df_full_date_analysis.week_ahead.isin(wi),:]
                    df_test_t_wi = df_test_t.loc[df_test_t.week_ahead.isin(wi),:]
                    col_name = 'point_avg'
                    df_analysis_bst = self.get_bst_data(df_full_date_analysis_w, self.model_list, col_name)
                    lasso1, lasso2 = self.get_bst_coef_nbe(df_analysis_bst, self.model_list)
                    df_test_t_wi_bst = self.get_bst_data(df_test_t_wi, self.model_list, col_name)
                    df_model_count = df_test_t_wi.groupby('model').count().reset_index()
                    
                    model_benchmark = df_model_count.loc[df_model_count.date == df_model_count.date.min(),'model'].values[0]
                    wi1 = df_test_t_wi.loc[df_test_t_wi.model == model_benchmark,'week_ahead'].unique()
                    
                    re_t1 = re_t.loc[re_t.week_ahead.isin(wi1),:]
                    
                    re_t1.loc[re_t1.week_ahead.isin(wi1),'pred_equal'] = lasso1.predict(df_test_t_wi_bst[self.model_list].values)
                    re_t1.loc[re_t1.week_ahead.isin(wi1),'pred_decay'] = lasso2.predict(df_test_t_wi_bst[self.model_list].values)
                    re_t1[f'{col_name}'] = mu*re_t1['pred_equal'].values+(1-mu)*re_t1['pred_decay'].values
                    re = pd.concat([re, re_t1])
                else:
                    print("Not exist.")
                    re_t = re_t.drop(re_t[re_t.week_ahead.isin(wi)].index)


        print(re.columns)

        end_time = datetime.now()
        print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
        print("The running time totally =", (end_time-start_time).seconds," seconds.") 


        re1 = deepcopy(re)
        re1['var'] = 'iHosp'
        re1['model'] = model_name
        re1['region'] = 'HK'
        re1['point'] = re1['point_avg']
        re_final = deepcopy(re1[df_mt_o.columns]) 
        re_final['date'] = pd.to_datetime(re_final['date'])
        re_final[f'{col_name}'] = re_final[f'{col_name}'].apply(lambda x: max(0, x))
        re_final['point'] = re_final['point'].apply(lambda x: max(0, x))

        re_final.to_csv(f'{self.origin_path}/Results/Point/forecast_{model_name}_{self.forecasting_mode}.csv', index = False)


    def get_bst_coef_awbe(self, df_bst_analysis, model_list, lambda_):
        seed_ = 2023
        np.random.seed(seed_)
        random.seed(seed_)
        
        alpha_range = np.array(range(1,30))/10 # define alpha range 
        ############################# normal lasso
        df_bst_p1 = deepcopy(df_bst_analysis[model_list+['true']])
        # LassoCV
        lasso_ = LassoCV(random_state = 2023, alphas=alpha_range, cv=5, fit_intercept = False).fit(df_bst_p1[model_list].values,df_bst_p1['true'])
        # best alpha
        best_alpha = lasso_.alpha_ 
        print("best_alpha = ", best_alpha)

        lasso1 = Lasso(random_state = 2023, alpha = best_alpha, fit_intercept = False) # 默认alpha =1 
        lasso1.fit(df_bst_p1[model_list].values,df_bst_p1['true'])
        for ii in range(len(model_list)):
            print("-- ",model_list[ii], " : ",lasso1.coef_[ii])
        print('---- val score = ', lasso1.score(df_bst_p1[model_list].values,df_bst_p1['true']))
        coef1 = lasso1.coef_

        ############################# weighted lasso
        df_bst_p2 = deepcopy(df_bst_analysis[model_list+['true','week_ahead','date']])
        df_bst_p2['rank'] = df_bst_p2.groupby('week_ahead').date.rank(ascending = False)
        df_bst_p2['decay_coef'] = df_bst_p2[['rank']].applymap(lambda x: np.exp(-lambda_*x))
        df_bst_p2.loc[df_bst_p2.decay_coef < 1e-3, 'decay_coef'] = 1e-3
        lasso_ = LassoCV(random_state = 2023, alphas=alpha_range,cv=5, fit_intercept = False).fit(df_bst_p2[model_list].values, df_bst_p2['true'], sample_weight = df_bst_p2['decay_coef'])
        best_alpha = lasso_.alpha_ 
        print("best_alpha = ", best_alpha)
        lasso2 = Lasso(random_state = 2023, alpha = best_alpha, fit_intercept = False) # 默认alpha =1 
        lasso2.fit(df_bst_p2[model_list].values,df_bst_p2['true'], sample_weight = df_bst_p2['decay_coef'])
        for ii in range(len(model_list)):
            print("-- ",model_list[ii], " : ",lasso2.coef_[ii])
        print('---- val score = ', lasso2.score(df_bst_p2[model_list].values,df_bst_p2['true'],sample_weight=df_bst_p2['decay_coef']))
        coef2 = lasso2.coef_
        return coef1, coef2

    def AWBE(self, lambda_ = 0.1):
        model_name = 'AWBE'

        df_test = pd.DataFrame()
        m = self.model_list[0]
        path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
        df_mt_o = pd.read_csv(path_mt)
        df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
        df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
        df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
        df_mt_o['model'] = m
        df_mt_o = df_mt_o[['date', 'true', 'week_ahead', 'point', 'point_avg', 'model']]
        df_test = pd.concat([df_test, df_mt_o])
        for m in self.model_list[1:]:
            path_mt = self.model_path+f'forecast_{m}_{self.forecasting_mode}.csv'
            df_mt_o = pd.read_csv(path_mt)
            df_mt_o['date'] = pd.to_datetime(df_mt_o['date'])
            df_mt_o = df_mt_o.loc[df_mt_o.date > self.rolling_start_date,:]
            df_mt_o = df_mt_o.loc[df_mt_o.date <= pd.to_datetime('2019-07-14'),:]
            df_mt_o['model'] = m
            df_mt_o = df_mt_o[['date', 'true', 'week_ahead', 'point', 'point_avg', 'model']]
            df_test = pd.concat([df_test, df_mt_o])

        df_full = deepcopy(df_test)

        start_time = datetime.now()

        date_analysises = df_full.date.unique()
        date_analysises = date_analysises[(date_analysises >= pd.to_datetime('2004-07-01'))&(date_analysises <= pd.to_datetime('2019-07-14'))] #
        mu = 0
        
        re = pd.DataFrame()
        for l in range(len(date_analysises)):
            # l = 1
            print("-------------------------------------------  pred_date", date_analysises[l], '  ----------------------------------------------------')
            df_full_date_analysis = df_full.loc[df_full.date < pd.to_datetime(date_analysises[l]),:]
            re_t = df_full.loc[df_full['date'] == date_analysises[l],:]
            print(" ---- model_length =", len(re_t.model.unique()))
            re_t['coef_equal'] = 0.0
            re_t['decay_coef'] = 0.0
            # get coef
            for wi in [np.array([0]),np.array(range(1, self.max_pred_horizon+1))]:
                print("-------------- week = ", wi, " ---------------")
                if len(re_t.loc[re_t.week_ahead.isin(wi),'model'].unique()) == len(self.model_list):
                    df_full_date_analysis_w = df_full_date_analysis.loc[df_full_date_analysis.week_ahead.isin(wi),:]
                    df_test_t_wi = re_t.loc[re_t.week_ahead.isin(wi),:]
                    col_name = 'point_avg'
                    df_analysis_bst = self.get_bst_data(df_full_date_analysis_w, self.model_list, col_name)
                    coef1, coef2 = self.get_bst_coef_awbe(df_analysis_bst, self.model_list, lambda_ = lambda_)
                    df_model_count = df_test_t_wi.groupby('model').count().reset_index()
                    
                    model_benchmark = df_model_count.loc[df_model_count.date == df_model_count.date.min(),'model'].values[0]
                    wi1 = df_test_t_wi.loc[df_test_t_wi.model == model_benchmark,'week_ahead'].unique()
                    
                    re_t1 = re_t.loc[re_t.week_ahead.isin(wi1),:]
                    
                    for i in range(len(self.model_list)):
                        m = self.model_list[i]
                        re_t1.loc[re_t1['model'] == m,'coef_equal'] = coef1[i]
                        re_t1.loc[re_t1['model'] == m,'decay_coef'] = coef2[i]
                    re_t1[f'{col_name}'] = mu*(re_t1[f'{col_name}'].values * re_t1['coef_equal'].values)+(1-mu)*(re_t1[f'{col_name}'].values * re_t1['decay_coef'].values)
                    re_t1['point'] = mu*(re_t1['point'].values * re_t1['coef_equal'].values)+(1-mu)*(re_t1['point'].values * re_t1['decay_coef'].values)
                    re = pd.concat([re, re_t1])
                else:
                    print("Not exist.")
                    re_t = re_t.drop(re_t[re_t.week_ahead.isin(wi)].index)
                    

        print(re.columns)

        end_time = datetime.now()
        print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
        print("The running time totally =", (end_time-start_time).seconds," seconds.") 


        re1 = deepcopy(re).drop(['model','coef_equal','true','decay_coef'], axis = 1)
        re1 = re1.groupby(['date','week_ahead']).sum().reset_index()
        re1['var'] = 'iHosp'
        re1['model'] = model_name
        re1['region'] = 'HK'
        val_true = re[['date','week_ahead','true']].groupby(['date','week_ahead']).mean().reset_index()
        re_final = pd.merge(re1, val_true, on=['date','week_ahead'], how='inner')
        re_final = deepcopy(re_final[df_mt_o.columns]) 
        re_final['date'] = pd.to_datetime(re_final['date'])
        re_final[f'{col_name}'] = re_final[f'{col_name}'].apply(lambda x: max(0, x))
        re_final['point'] = re_final['point'].apply(lambda x: max(0, x))
        re_final = re_final.sort_values(by = ['date','week_ahead'])

        re_final.to_csv(f'{self.origin_path}/Results/Point/forecast_{model_name}_{self.forecasting_mode}.csv', index = False)



if __name__ == '__main__':
    model_list = [
            'ARIMA_rolling',
            'GARCH_rolling',
            'RF_rolling',
            'XGB_rolling',
            'InTimePlus_v3_nontuning_rolling_v2',
            'LSTM_v3_nontuning_rolling_v2',
            'GRU_v3_nontuning_rolling_v2',
            'TSTPlus_v3_nontuning_rolling'
        ]
    mode = 'test8'

    pwd=os.path.abspath(os.path.dirname(os.getcwd()))
    origin_path = pwd

    rolling_start_date = pd.to_datetime('2003-11-01')
    decay_mode = 12
    lambda_ = -np.log(0.01)/decay_mode

    # avg ensemble
    my_avg_ensemble = AVG_ENSEMBLE(model_list=model_list, forecasting_mode=mode, origin_path=origin_path, rolling_start_date=rolling_start_date)
    my_avg_ensemble.SAE()
    my_avg_ensemble.AWAE(lambda_ = lambda_)

    # blending
    my_blending = BLENDING(model_list=model_list, forecasting_mode=mode, origin_path=origin_path, rolling_start_date=rolling_start_date)
    my_blending.NBE()
    my_blending.AWBE(lambda_ = lambda_)