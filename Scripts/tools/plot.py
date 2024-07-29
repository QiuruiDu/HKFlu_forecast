"""
    plot figure
    Richael-2023/6/16
    --------------------
    plot different figure
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import copy

class Plot_():
    def __init__(self):
        pass 

    def get_metric(self, df, pred_stamp = 4, log = True):
        df_plot1 = df.set_index('date', drop = True, inplace=False)
        mape_t, rmse_t = [], []
        for i in range(pred_stamp):
            df_t = df_plot1.loc[df_plot1['week_ahead'] == i,:][['true','pred']]
            if log == True:
                df_t = df_t.applymap(np.exp)
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            print(f"week{i} predict MAPE = ", smape_, "RMSE = ", rmse_)
            mape_t.append(smape_)
            rmse_t.append(rmse_)
        return mape_t, rmse_t


    def get_plot(self, df, pred_stamp = 4, log = True, figsize = None):
        # df_plot1 = df.set_index('date', drop = True, inplace=False)
        if figsize is None:
            plt.figure(figsize=(14,23))
        else:
            plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
        for i in range(pred_stamp):
            df_plot1 = df.loc[df['week_ahead'] == i,['true','pred','date']]
            df_t = df_plot1.set_index('date', drop = True, inplace=False)
            if log == True:
                df_t = df_t.applymap(np.exp)
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            print(f"week{i} predict SMAPE = ", smape_, "RMSE = ", rmse_)
            plt.subplot(pred_stamp,1,(i+1))
            plt.plot(df_t.index, df_t.true, color='blue')
            plt.plot(df_t.index, df_t.pred, color = 'orange')
            plt.legend(labels = ['true','pred'])
            plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}, RMSE = {rmse_}')

        plt.show()

    # def get_saved_plot(self, df, pred_stamp = 4, name = 'temp'):
    #     df_plot1 = df.set_index('date', drop = True, inplace=False)
    #     plt.figure(figsize=(14,23))
    #     plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
    #     for i in range(pred_stamp):
    #         df_t = df_plot1.loc[df_plot1['week_ahead'] == i,:][['true','pred']]
    #         df_t = df_t.applymap(np.exp)
    #         smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
    #         print(f"week{i} predict MAPE = ", smape_)
    #         plt.subplot(pred_stamp,1,(i+1))
    #         plt.plot(df_t.index, df_t.true, color='blue')
    #         plt.plot(df_t.index, df_t.pred, color = 'orange')
    #         plt.legend(labels = ['true','pred'])
    #         plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}')

    #     fig_path = f'/Users/hkuph/richael/flu/FluForecasting/figure/{name}.png'
    #     plt.savefig(fig_path, dpi=450, bbox_inches='tight', facecolor='w')
    #     print(f"the figure has been saved as {name}")
    #     plt.show()
    def get_saved_plot(self, df, pred_stamp = 4, log = True, figsize = None, path = None, show = False):
        # df_plot1 = df.set_index('date', drop = True, inplace=False)
        if figsize is None:
            plt.figure(figsize=(14,23))
        else:
            plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
        for i in range(pred_stamp):
            df_plot1 = df.loc[df['week_ahead'] == i,['true','pred','date']]
            df_t = df_plot1.set_index('date', drop = True, inplace=False)
            if log == True:
                df_t = df_t.applymap(np.exp)
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            print(f"week{i} predict SMAPE = ", smape_, "RMSE = ", rmse_)
            plt.subplot(pred_stamp,1,(i+1))
            plt.plot(df_t.index, df_t.true, color='blue')
            plt.plot(df_t.index, df_t.pred, color = 'orange')
            plt.legend(labels = ['true','pred'])
            plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}, RMSE = {rmse_}')

        if path is None:
            fig_path = f'/Users/hkuph/richael/flu/FluForecasting/figure/fig_tmp.png'
        else:
            fig_path = path
        plt.savefig(fig_path, dpi=450, bbox_inches='tight', facecolor='w')
        if show == True:
            plt.show()


