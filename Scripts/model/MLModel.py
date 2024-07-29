import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# from statsmodels.tsa.arima_model import ARIMA
# from pmdarima import auto_arima
import copy
import math
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import Lasso, LassoCV


class MLDataset():
    # mydataset需要的几个函数：
    # normalization
    # split_windows
    # to tensor
    def __init__(self):
        self.max = {}
        self.min = {}
        self.lag_mode = None
        self.rate_lag = None
        self.cov_list = None
        self.max_test_lag = None
        self.cov_lagdict = None
        self.max_lag_order = 0
        self.pred_horizon = 0

    def _init_scaler(self, df_train):
        """
        compute the max and min by using train data
        ----------------------------
        df_train: the train dataframe
        """
        for c in df_train.columns:
            self.max[c] = np.max(df_train[c].values)
            self.min[c] = np.min(df_train[c].values)
        # print(self.max)
        # print(self.min)

    def output_scaler(self):
        return self.max, self.min

    def maxmin_normalization(self, df):
        """
        max-min normalization，对数据进行标准化处理
        ---------
        df: dataframe
        """
        data = copy.deepcopy(df)
        for c in df.columns:
            if c in self.max.keys():
                data[c] = (data[c]-self.min[c])/(self.max[c]-self.min[c])
            else:
                print("the columns ",c," does not exist!")
        data = data.astype(np.float64)
        return data
    
    def inverse_normalization(self, df):
        for c in df.columns:
            df[c] = df[c]*(self.max[c]-self.min[c])+self.min[c]
        return df
    
    def output_rate_scaler(self):
        """
        get the max and min scaler of rate variable
        ----------------
        """
        return self.max['rate'], self.min['rate']

    def _get_cov_best_lag(self, df):
        col_origin = list(df.columns)
        df_cor = pd.DataFrame(columns=['var','lag','pearson'])
        ### 计算各阶自变量滞后的相关系数结果
        for i in range(len(self.cov_list)):
            df_i = df[['rate', self.cov_list[i]]]
            # if i == 0:
            #     print(df_i)
            for lag in range(1, self.max_cov_lag+1):
                df_i[self.cov_list[i]] = df_i[self.cov_list[i]].shift(lag) #shift(正值)表示向下shift，以前的shift到现在
                tmp_cor = df_i.corr()
                # print(cov_list[i], tmp_cor)
                df_cor.loc[len(df_cor.index),:] = np.array([self.cov_list[i], lag, tmp_cor.loc['rate',self.cov_list[i]]])

        df_cor['pearson'] = df_cor['pearson'].astype('float64').abs()
        df_cor['lag'] = df_cor['lag'].astype('int')
        idx = df_cor.groupby('var')['pearson'].idxmax()
        df_lag = df_cor.iloc[idx,:][['var', 'lag']]
        # print("--- df_lag-------")
        # print(df_lag)
        cov_lagdict = dict()
        for i in range(df_lag.shape[0]):
            # print("best lag : ", df_lag.iloc[i,:]['var'], df_lag.iloc[i,:]['lag'])
            cov_lagdict[df_lag.iloc[i,:]['var']] = df_lag.iloc[i,:]['lag']
        
        ## 计算各阶rate滞后的相关系数结果
        df_cor_rate = pd.DataFrame(columns=['var','lag','pearson'])
        for l in range(1, self.max_rate_lag+1):
            df_i = copy.deepcopy(df[['rate']])
            df_i['rate_lagged'] = df_i['rate'].shift(l)
            tmp_cor = df_i.corr()
            df_cor_rate.loc[len(df_cor_rate.index),:] = np.array(['rate', l, tmp_cor.loc['rate','rate_lagged']])
        df_cor_rate['pearson'] = df_cor_rate['pearson'].astype('float64').abs()
        df_cor_rate['lag'] = df_cor_rate['lag'].astype('int')
        df_cor_rate = df_cor_rate.loc[df_cor_rate.pearson > 0.5,:]
        rate_lag = max(df_cor_rate.lag)
        cov_lagdict['rate'] = rate_lag
        # 得到最终的最优的滞后
        self.cov_lagdict = cov_lagdict
        print("the lags having best correlation for covariates : ", self.cov_lagdict)
              
    def output_best_lag(self):
        return self.cov_lagdict
    
    def output_max_lag(self):
        return self.max_lag_order
    
    def _deal_cov_lag_train(self, df):
        """
        auto choose lag order by using correlation method, and then finish lag
        -----------------------------
        df : DataFrame
        cov_list: the covriate list
        """
        self._get_cov_best_lag(df = copy.deepcopy(df))

        df_here = copy.deepcopy(df)
        max_lag = 0
        for k in self.cov_lagdict.keys():
            if k not in df_here.columns:
                print(k," is not a column of data!!!")
            else:
                lag_num =  self.cov_lagdict[k]
                # print(k," lag ",lag_num," order")
                max_lag = max(max_lag, lag_num)
                for l in range(1, lag_num+1):
                    df_here[f'{k}_{l}d'] = df_here[k].shift(l) 

        self.max_lag_order = max_lag
        # col_drop = list(set(col_origin).union(set(cov_list)))
        # col_drop.remove('rate')
        # col_drop = list(set(col_drop))
        df_re = df_here.drop(self.cov_list, axis = 1).dropna()
        col_sorted = list(df_re.drop('rate', axis = 1).columns)
        col_sorted.append('rate')
        df_re = df_re[col_sorted]
        return df_re
    
    def _deal_cov_lag_test(self, df):
        """
        auto choose lag order by using correlation method, and then finish lag
        -----------------------------
        df : DataFrame
        cov_lagdict: lag dict
        """
        df_here = copy.deepcopy(df)
        col_origin = list(df_here.columns)
        ## 进行滞后处理
        for k in self.cov_lagdict.keys():
            if k not in df_here.columns:
                print(k," is not a column of data!!!")
            else:
                lag_num =  self.cov_lagdict[k]
                # print(k," lag ",lag_num," order")
                for l in range(1, lag_num+1):
                    df_here[f'{k}_{l}d'] = df_here[k].shift(l) 
        
        # col_drop = list(set(col_origin).union(set(self.cov_list)))
        # col_drop.remove('rate')
        # col_drop = list(set(col_drop))
        df_re = df_here.drop(self.cov_list, axis = 1).dropna()
        col_sorted = list(df_re.drop('rate', axis = 1).columns)
        col_sorted.append('rate')
        df_re = df_re[col_sorted]
        return df_re
    
    def _init_lag_func(self, max_rate_lag = 1, cov_list = None, max_cov_lag = 0):
        if max_cov_lag <= 0:
            raise Exception('the lag_mode is auto, but lag_mode and max_cov_lag is not matched!')
        self.max_rate_lag = max_rate_lag
        self.cov_list = cov_list
        self.max_cov_lag = max_cov_lag
        self.cov_lagdict = None

    def get_train_data(self, df, max_rate_lag = 1, cov_list = None, max_cov_lag = 0, pred_horizon = 1, validation = False):
        """
        get the train and validation data from df dataframe
        ----------------------------------
        df : DataFrame of train data
        """
        # normalization
        self.pred_horizon = pred_horizon
        df_train = copy.deepcopy(df)
        self._init_scaler(df_train = df_train)
        df_train = self.maxmin_normalization(df_train)
        # print("original train size is : ",df_train.shape)
        ## deal lag for covariates
        self._init_lag_func(max_rate_lag=max_rate_lag, cov_list=cov_list, max_cov_lag=max_cov_lag)
        df_train1 = self._deal_cov_lag_train(df_train)
        for i in range(pred_horizon):
            df_train1[f'rate_y{i}'] = df_train1['rate'].shift(-i)
        df_train1 = df_train1.drop('rate', axis = 1).dropna()
        # print("train data min_date = ", min(df_train1.index), ", max_date = ", max(df_train1.index))
        # dataset = df_train.values
        x_train, y_train = df_train1.iloc[:,0:-pred_horizon], df_train1.iloc[:,-pred_horizon:]
        # based on the validation mode, return different result
        if validation is False:
            train_datadict = {'x_data':x_train,
                          'y_data':y_train}
            return train_datadict, self.max_lag_order
        # split train and validation data
        if validation is True:
            train_ind = np.random.choice(x_train.shape[0], size=int(x_train.shape[0]*0.85), replace=False)
            val_ind = np.setdiff1d(np.array(range(x_train.shape[0])), train_ind)
            x_train, y_train = x_train[train_ind,:],y_train[train_ind,:]
            x_val, y_val = x_train[val_ind,:], y_train[val_ind,:]
            train_datadict = {'x_data':x_train, 'y_data':y_train}
            val_datadict = {'x_data':x_val, 'y_data':y_val}
            return train_datadict, val_datadict, self.max_lag_order
        
    def get_test_data(self, df):
        # print(" --- for get_test_data function ---, origin shape = ", df.shape)
        df_t = self.maxmin_normalization(copy.deepcopy(df))
        # print(" ----- after normalization, shape = ", df_t.shape)
        df_t = self._deal_cov_lag_test(df_t)
        # print(" ----- after lag dealing, shape = ", df_t.shape)
        for i in range(self.pred_horizon):
            df_t[f'rate_y{i}'] = df_t['rate'].shift(-i)
        # print(" ----- after rate lag dealing, shape = ", df_t.shape)
        df_t = df_t.drop('rate', axis = 1).dropna()
        # print(" ----- after dropna, shape = ", df_t.shape)
        # print(df_t.shape)
        # x_test, y_test = df_t.iloc[:,0:-self.pred_horizon], df_t.iloc[:,-self.pred_horizon:]
        return df_t

class RFmodel():
    def __init__(self):
        self.best_param = {}
        self.model = None

    def _init_model(self, random_state):
        self.model = RandomForestRegressor(random_state=random_state)
    

    def CV_train_(self, x_train, y_train, fold_num = 5, param_dict = {}, random_state = 42,verbose = True):
        """
        the tunning parameter list : [n_estimators, max_depth]
        ---------------------------------------
        """
        model = RandomForestRegressor(random_state = random_state)
        model_cv = GridSearchCV(model, param_grid=param_dict, cv=fold_num, n_jobs = 3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.model = RandomForestRegressor(**best_params) # best model
        if verbose == True:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = RandomForestRegressor(**self.best_param) # best model
        self.model.fit(x_train,y_train)

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def output_model(self):
        return self.model
    

class XGBmodel():
    def __init__(self) -> None:
        self.model = None
        self.best_param = {}

    def _init_model(self, random_state):
        self.model = XGBRegressor(random_state = random_state)   
        
    def CV_train_(self, x_train, y_train, fold_num = 5, param_dict = {}, iter_num = 80, verbose = True):
        """
        the tunning parameter list : [n_estimators, max_depth]
        ---------------------------------------
        """
        model = XGBRegressor()
        model_cv = RandomizedSearchCV(model, param_distributions=param_dict, n_iter=iter_num, cv=fold_num, n_jobs = 3)
        model_cv.fit(x_train, y_train)
        best_params = model_cv.best_params_
        self.best_param = best_params
        self.model = XGBRegressor(**best_params) # best model
        if verbose == True:
            print("For this model, the best parameters are ", best_params)

    def fit_(self, x_train, y_train, random_state):
        # x_data, y_data = copy.deepcopy(x_train), copy.deepcopy(y_train)
        if self.model is None:
            self._init_model(random_state)
        else:
            self.best_param['random_state'] = random_state
            self.model = XGBRegressor(**self.best_param)
        # train_ind = np.random.choice(x_data.shape[0], size=int(x_data.shape[0]*0.9), replace=False)
        # val_ind = np.setdiff1d(np.array(range(x_data.shape[0])), train_ind)
        # x_train, y_train = x_data[train_ind,:],y_data[train_ind,:]
        # x_val, y_val = x_data[val_ind,:], y_data[val_ind,:]
        # print("train_size = ", x_train.shape[0], ", val_size = ", x_val.shape[0])
        # self.model.fit(x_train,y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', early_stopping_rounds=5)
        self.model.fit(x_train, y_train) 

    def predict_(self, x_test):
        y_test_hat = self.model.predict(x_test)
        return y_test_hat
    
    def output_model(self):
        return self.model
    
