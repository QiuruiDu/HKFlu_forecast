########################################
# preparation
########################################

library(plyr)
library(dplyr)
library(tidyverse)
library(ISOweek)
library(locpol)
library(forecast)
library(scoringutils)
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
library(cowplot)
library(data.table)

source('./tools/MyPlot.R')

std_mode= 'ydiff'
models = c(
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
  'AWAE',
  'AWBE'
)
model_name_list = c()
for(st in models){
  model_name_list = c(model_name_list, strsplit(st, '_', fixed = F, perl = F, useBytes = F)[[1]][1])
}


models.fig2 = models
model_name.fig2 = model_name_list

mode = 'test8_2023'
if(mode == 'test4'){
  pred_horizon = 5
}else{
  pred_horizon = 9
}

################################################################################
# get interval result
################################################################################
# read interval windows length

dates_analysis = seq(as.Date("2022-11-06"),as.Date("2024-03-10"),by="week") 
interval.window.len = read.csv(paste0('../Results/Interval_',std_mode,'_raw/total_interval_window.csv'))
proj.interval <-tibble()
for(window_interval in c(5, seq(8, 50, 2))){
  for (i in c(1:length(models))){ #1:

    model = models[i]
    path_ = paste0('../Results/Interval_',std_mode,'_raw/interval',window_interval,'_',model,'_',mode,'.csv')
    if(file.exists(path_)){
      proj_tmp<-read.csv(file=path_,stringsAsFactors =F)
      
      proj_tmp = proj_tmp %>%
        mutate(date=as.Date(date)) %>%
        mutate(point = point_avg) %>%
        mutate(model = model_name_list[i]) %>%
        mutate(model_path_name = models[i]) %>%
        mutate(window_interval = window_interval)
      print(paste0("---- model = ", model, ", shape = ",length(proj_tmp)))
      proj.interval<-rbind.fill(proj.interval,proj_tmp)
    }
  }
}

# join + filter window interval
proj.interval.chosen = proj.interval %>%
  join(interval.window.len[,c(1:3)], type = 'inner',
       by = c('model' = 'model', 'week_ahead' = 'week_ahead')) %>%
  filter(window_interval == choose_window_interval) %>%
  dplyr::select(-c(window_interval, choose_window_interval))


if(dir.exists(paste0('../Results/Interval_',std_mode,'_pred/')) == FALSE){
  dir.create(paste0('../Results/Interval_',std_mode,'_pred/'), recursive = TRUE)
}

for(m_ in models){
  proj.interval.model = proj.interval.chosen %>%
    filter(model_path_name == m_) %>%
    dplyr::select(-model_path_name)
  print(paste0("-------- model = ", m_,", shape = ", dim(proj.interval.model)[1], " ------------"))
  path_ = paste0('../Results/Interval_',std_mode,'_pred/interval_',m_,'_', mode,'.csv') #_split_season
  write.csv(proj.interval.model, file = path_, row.names = FALSE)
}
