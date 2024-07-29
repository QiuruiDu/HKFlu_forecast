################################################################################
# preparation
################################################################################
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
library(patchwork)
library(data.table)
library(reshape2)
library(lemon)

source('./tools/MyPlot.R')

models = c(
  'Baseline', 
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'InTimePlus_v3_nontuning_rolling_v2',
  'TSTPlus_v3_nontuning_rolling',
  'LSTM_v3_nontuning_rolling_v2',
  'GRU_v3_nontuning_rolling_v2',
  
  'SAE_NC_R1',
  'NBE',
  'AWAE_NC_R1',
  'AWBE_NC_R1'
)
model_name_list = c()
for(st in models){
  model_name_list = c(model_name_list, strsplit(st, '_', fixed = F, perl = F, useBytes = F)[[1]][1])
}

index_level = c('RMSE','SMAPE','MAE','WIS','MAPE')
fig.name = 'Fig3_trend_and_season_analysis'

trend_split3= function(df){
  df1 = df %>%
    mutate(date = as.Date(date)) %>%
    mutate(true = as.numeric(true)) %>%
    dplyr::select(date, true) %>%
    group_by(date) %>%
    summarise(true = mean(true)) %>%
    ungroup() %>%
    mutate(trend = 'normal') %>%
    arrange(date) 
  df1$true_future_1d = c(df1$true[c(2:nrow(df1))], NA)
  df1$true_future_2d = c(df1$true[c(3:nrow(df1))], NA, NA)
  df1$true_past_1d = c(NA,df1$true[c(1:(nrow(df1)-1))])
  df1$true_past_2d = c(NA,NA,df1$true[c(1:(nrow(df1)-2))])
  
  df1$trend[which(((df1$true_future_2d-df1$true_future_1d)>0)&((df1$true_future_1d-df1$true)>0))] = 'Growth'
  df1$trend[which(((df1$true_future_1d-df1$true)>0)&((df1$true-df1$true_past_1d)>0))] = 'Growth'
  df1$trend[which(((df1$true-df1$true_past_1d)>0)&((df1$true_past_1d-df1$true_past_2d)>0))] = 'Growth'
  
  df1$trend[which(((df1$true_future_2d-df1$true_future_1d)<0)&((df1$true_future_1d-df1$true)<0))] = 'Decline'
  df1$trend[which(((df1$true_future_1d-df1$true)<0)&((df1$true-df1$true_past_1d)<0))] = 'Decline'
  df1$trend[which(((df1$true-df1$true_past_1d)<0)&((df1$true_past_1d-df1$true_past_2d)<0))] = 'Decline'
  
  df1$trend[which(((df1$true_future_1d-df1$true)<0)&((df1$true-df1$true_past_1d)>0)&(df1$trend == 'normal'))] = 'Plateau'
  df1$trend[which(((df1$true_future_1d-df1$true)>0)&((df1$true-df1$true_past_1d)<0)&(df1$trend == 'normal'))] = 'Plateau'
  
  df1$trend[which((df1$true/1000 < 0.005))] = 'normal'
  # df1%>%
  #   ggplot()+
  #   geom_line(aes(x = date, y = true, color = trend, group = year),linewidth=0.8)#+
  
  return(df1[,c('date','trend')])
}
get_relative = function(proj.t, plot.baseline = 'baseline'){
  res.t<-my.compute(proj.t)
  evual_result<-res.t %>%
    filter(inclusion==1) %>% 
    mutate(week_ahead=as.factor(week_ahead)) %>%
    group_by(model,week_ahead) %>%
    dplyr::summarize(RMSE = sqrt(mean(abs_error^2,na.rm=T)) ,
                     MAPE = (mean(abs_error/abs(true),na.rm=T)), 
                     SMAPE = (mean(abs_error/(abs(point)+abs(true))*2,na.rm=T)),
                     MAE = mean(abs_error, na.rm=T),
                     wis_avg = mean(wis, na.rm = T))%>%
    ungroup()
  evual_result = evual_result %>%
    mutate(WIS = wis_avg) %>%
    dplyr::select(-wis_avg) 
  evual.long.result = melt(evual_result, id.vars = c("model", 'week_ahead'))
  
  baseline_avg = evual.long.result %>%
    filter(model == 'Baseline') %>%
    group_by(model, variable) %>%
    dplyr::summarize(avg_baseline = mean(value, na.rm = TRUE))
  
  evual.relative = evual.long.result %>%
    group_by(model, variable) %>%
    dplyr::summarize(avg_value = mean(value, na.rm = TRUE)) %>%
    left_join(baseline_avg[,c('variable','avg_baseline')], by=c("variable" = 'variable')) %>%
    mutate(relative_index = avg_value/avg_baseline)
  
  if(is.null(plot.baseline) == TRUE){
    evual.relative.ranked = evual.relative %>%
      filter(model %in% c("Baseline","ARIMA","GARCH","RF","XGB", "InTimePlus","TSTPlus" , "LSTM", "GRU")) %>%
      mutate(rank = 0) %>%
      arrange(variable, relative_index)
    for(v in unique(evual.relative.ranked$variable)){
      evual.relative.ranked$rank[which(evual.relative.ranked$variable == v)] = rank(evual.relative.ranked$avg_value[which(evual.relative.ranked$variable == v)], ties.method = 'min')
    }
    evual.plot.baseline = evual.relative.ranked %>%
      filter(rank == 1) %>%
      mutate(avg_plot_baseline = relative_index) %>%
      dplyr::select(variable, model, avg_plot_baseline)
    evual.relative = evual.relative %>%
      left_join(evual.plot.baseline[,c('variable', 'avg_plot_baseline')], by =  c('variable' = 'variable')) %>%
      mutate(plot_baseline = avg_plot_baseline) %>%
      dplyr::select(-c(avg_plot_baseline))
  }else{
    evual.relative$plot_baseline = 0
    for(m in model_name_list1){
      evual.relative$plot_baseline[which(evual.relative$model == m)] = evual.relative$relative_index[which(evual.relative$model == baseline)]
    }
  }
  
  return(evual.relative)
}

################################################################################
# test
################################################################################
mode = 'test8'
dates_gap = seq(as.Date("2009-04-05"),as.Date("2010-03-21"),by="week")
if(mode == 'test4'){
  pred_horizon = 5
}else{
  pred_horizon = 9
}
dates_analysis <- seq(as.Date("2007-11-04"),as.Date("2019-07-14"),by="week")

models1 = models
model_name_list1 = model_name_list
proj<-tibble()
for (i in c(1:length(models1))){
  model = models1[i]
  path_ = paste0('../Results/Interval_ydiff_pred/interval_',model,'_',mode,'.csv')
  if(file.exists(path_)){
    proj_tmp<-read.csv(file=path_,stringsAsFactors =F)
    
    proj_tmp$model = model_name_list1[i]
    
    proj_tmp = proj_tmp %>%
      mutate(date=as.Date(date)) %>%
      mutate(point = point_avg) %>%
      mutate(date_origin=date-week_ahead*7) %>%
      filter(date >= min(dates_analysis)) %>%
      filter(date <= max(dates_analysis)) %>%
      filter(date_origin >= min(dates_analysis))
    
    proj.tmp.trend = trend_split3(proj_tmp)
    proj_tmp = proj_tmp %>%
      left_join(proj.tmp.trend, by = c('date' = 'date'))
    proj_tmp$trend[is.na(proj_tmp$trend)] <- 'normal'

    proj_tmp = proj_tmp[which((proj_tmp$date_origin < min(dates_gap))|(proj_tmp$date_origin > max(dates_gap))),]
    
    print(paste0('------model is', model,', data length = ',dim(proj_tmp)[1]))
    proj<-rbind.fill(proj,proj_tmp)
  }
}

proj$model<-factor(proj$model,levels=model_name_list1, ordered=T)

proj$inclusion<-1

proj.mode <- proj %>%
  mutate(date=as.Date(date)) %>%
  mutate(point = point_avg) %>%
  mutate(date_origin=date-week_ahead*7) %>%
  filter(date>=min(dates_analysis))  %>%
  filter(date<=max(dates_analysis)) %>%
  mutate(weekid_pred = as.numeric(strftime(date,format= "%V"))) 


pal<-c("black",brewer.pal(10,"Paired"),"turquoise",brewer.pal(6, "Dark2")[4:6])
val_size<-c(rep(0.5,length(models)))
val_linetype<-c("dashed", rep("solid",length(models)-1))
lab_mod<-levels(droplevels(proj$model))
val_linewidth = c(rep(0.5, length(models)-2), 1.5, 1.9)
val_point_size = c(rep(1.3, length(models)-1), 2.5, 2.9)

## ----- trend ----
trend_level = 'Growth'
proj.increase = proj.mode %>%
  filter(trend == trend_level)
res.increase = get_relative(proj.increase, NULL)
res.increase$type = 'Growth'

trend_level = 'Plateau'
proj.steady = proj.mode %>%
  filter(trend == trend_level)
res.steady = get_relative(proj.steady, NULL)
res.steady$type = 'Plateau'

trend_level = 'Decline'
proj.decrease = proj.mode %>%
  filter(trend == trend_level)
res.decrease = get_relative(proj.decrease, NULL)
res.decrease$type = 'Decline'

res.relative = rbind(res.increase, res.steady)
res.relative = rbind(res.relative, res.decrease)
res.relative$type = factor(res.relative$type, level = c('Growth','Plateau', 'Decline'))
res.relative$variable = factor(res.relative$variable, level = index_level)


p.trend <-res.relative %>%
  ggplot() +
  geom_point(aes(x = model, y = relative_index, color = model, group=type), size = 2) +
  geom_hline(data = res.relative, aes(yintercept = plot_baseline), linetype = "dashed") +
  facet_wrap( ~ type+variable, ncol = 5, scale = "free")+
  labs(y="Performance relative to baseline") + 
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_point_size)+
  labs(title = 'Performance in different epidemic trends',tag = "A") + #
  theme_bw()+
  theme(panel.grid = element_blank(),
        strip.text.y = element_text(size=10, angle = 0))+
  theme( legend.text=element_text(size=9),
         legend.key.size = unit(5, "mm"))+
  theme(title = element_text(size=9),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size=9),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=9))+
  theme(strip.text = element_text(face = "bold")) +
  guides(color=guide_legend(ncol=1))

################################################################################
# trend split and plot figure
################################################################################
data_rt<-readRDS(paste0("../Data/data_lograte.rds"))
data_rt$date <- as.Date(data_rt$date)
date_analysis <- seq(as.Date("1998-01-02"),as.Date("2019-07-14"),by="week")
fig_name = 'Fig_ILI_Trend_by_stage'


dat = data_rt %>%
  filter(date >= as.Date('1998-01-04')) %>%
  filter(date_analysis == max(as.Date(data_rt$date_analysis))) %>%
  mutate(year = as.numeric(year(date))) %>%
  mutate(monthid = as.numeric(strftime(date, format = "%m"))) 


trend_split3_truedata= function(df){
  df1 = df %>%
    mutate(date = as.Date(date)) %>%
    mutate(true = as.numeric(rate)) %>%
    dplyr::select(date, year, true) %>%
    group_by(date, year) %>%
    summarise(true = mean(true)) %>%
    ungroup() %>%
    mutate(trend = 'Normal') %>%
    arrange(date) 
  df1$true_future_1d = c(df1$true[c(2:nrow(df1))], NA)
  df1$true_future_2d = c(df1$true[c(3:nrow(df1))], NA, NA)
  df1$true_past_1d = c(NA,df1$true[c(1:(nrow(df1)-1))])
  df1$true_past_2d = c(NA,NA,df1$true[c(1:(nrow(df1)-2))])
  
  df1$trend[which(((df1$true_future_2d-df1$true_future_1d)>0)&((df1$true_future_1d-df1$true)>0))] = 'Growth'
  df1$trend[which(((df1$true_future_1d-df1$true)>0)&((df1$true-df1$true_past_1d)>0))] = 'Growth'
  df1$trend[which(((df1$true-df1$true_past_1d)>0)&((df1$true_past_1d-df1$true_past_2d)>0))] = 'Growth'
  
  df1$trend[which(((df1$true_future_2d-df1$true_future_1d)<0)&((df1$true_future_1d-df1$true)<0))] = 'Decline'
  df1$trend[which(((df1$true_future_1d-df1$true)<0)&((df1$true-df1$true_past_1d)<0))] = 'Decline'
  df1$trend[which(((df1$true-df1$true_past_1d)<0)&((df1$true_past_1d-df1$true_past_2d)<0))] = 'Decline'
  
  df1$trend[which(((df1$true_future_1d-df1$true)<0)&((df1$true-df1$true_past_1d)>0)&(df1$trend == 'Normal'))] = 'Plateau'
  df1$trend[which(((df1$true_future_1d-df1$true)>0)&((df1$true-df1$true_past_1d)<0)&(df1$trend == 'Normal'))] = 'Plateau'
  
  # df1$trend_yesterday1 = c(NA, df1$trend[c(0:(nrow(df1)-1))])
  # df1$trend_yesterday2 = c(NA,NA, df1$trend[c(0:(nrow(df1)-2))])
  # df1$trend_yesterday3 = c(NA,NA, NA, df1$trend[c(0:(nrow(df1)-3))])
  # df1$trend_yesterday4 = c(NA,NA, NA, NA, df1$trend[c(0:(nrow(df1)-4))])
  # df1$trend_tomorrow1 = c(df1$trend[c(2:nrow(df1))], NA)
  # df1$trend_tomorrow2 = c(df1$trend[c(3:nrow(df1))], NA, NA)
  # df1$trend_tomorrow3 = c(df1$trend[c(4:nrow(df1))], NA, NA, NA)
  # df1$trend_tomorrow4 = c(df1$trend[c(5:nrow(df1))], NA, NA, NA, NA)
  # 
  # df1$trend[which(((df1$trend_yesterday1 == 'Growth') | (df1$trend_yesterday2 == 'Growth')#| 
  #                    # (df1$trend_yesterday3 == 'Growth')#|(df1$trend_yesterday4 == 'growth')
  # )&(
  #   df1$trend_tomorrow1 == 'Decline' | df1$trend_tomorrow2 == 'Decline' #|
  #     # df1$trend_tomorrow3 == 'Decline' #| df1$trend_tomorrow4 == 'decline'
  # )&!(df1$trend %in% c('Growth','Decline')))] = 'Plateau'
  df1$trend[which((df1$true/1000 < 0.005))] = 'Normal'
  # df1%>%
  #   ggplot()+
  #   geom_line(aes(x = date, y = true, color = trend, group = year),linewidth=0.8)#+
  
  return(df1[,c('date','trend')])
}

dat.tmp.trend = trend_split3_truedata(dat)
dat.trend = dat %>%
  dplyr::select(date, year, rate) %>%
  left_join(dat.tmp.trend, by = c('date' = 'date'))
dat.trend$trend[is.na(dat.trend$trend)] <- 'Normal'

break_date = seq(as.Date("1998-11-01"),as.Date("2019-11-02"),by="year")
year_list = seq(1998, 2019)
title_name = c(rep(" ",10), as.character(year_list[1]))
for(i in c(2:length(year_list))){
  title_name = c(title_name, rep(" ",3), as.character(year_list[i]))
}

p.trend.ILI = 
  dat.trend%>%
  ggplot()+
  geom_line(aes(x = date, y = rate, color = trend, group = year),linewidth=0.8)+
  geom_hline(aes(yintercept=4.99), linetype = 'dashed', color = "#9C9C9C")+
  scale_y_continuous("Influenza activity (ILI+)") +
  labs(title = 'The split of different epidemic stages',tag = "B")+
  scale_x_date(breaks = break_date, label = date_format("%b"))+ 
  xlab(paste(title_name, collapse = "")) +
  coord_cartesian(ylim = c(0, 70))+
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size = 9, hjust=0.5), #, hjust=1.01
        axis.text.y = element_text(size= 10),
        axis.title.y = element_text(hjust=0.5),
        axis.title.x = element_text(size=9)) +
  theme(title = element_text(size=9))+
  guides(color=guide_legend(ncol=1)) +
  guides(alpha = FALSE, linetype = FALSE)



################################################################################
#season
################################################################################

proj.mode <- proj.mode %>%
  mutate(monthid = as.numeric(strftime(date, format = "%m")))%>%
  mutate(seasonid = 'Summer')
proj.mode$seasonid[which(proj.mode$monthid %in% c(11,12,1,2,3,4))] <- 'Winter'
proj.mode <- proj.mode %>%
  dplyr::select(-monthid)
proj.mode$seasonid = factor(proj.mode$seasonid, level = c('Summer','Winter'), ordered = TRUE)

res<-my.compute(proj.mode)
evual_result<-res %>%
  filter(inclusion==1) %>% 
  mutate(week_ahead=as.factor(week_ahead)) %>%
  group_by(model,week_ahead, seasonid) %>%
  dplyr::summarize(RMSE = sqrt(mean(abs_error^2,na.rm=T)) ,
                   MAPE = (mean(abs_error/abs(true),na.rm=T)), 
                   SMAPE = (mean(abs_error/(abs(point)+abs(true))*2,na.rm=T)),
                   MAE = mean(abs_error, na.rm=T),
                   avg_wis = mean(wis, na.rm = T))%>%
  ungroup() %>%
  mutate(WIS = avg_wis) %>%
  dplyr::select(-avg_wis)

evual.long.result = melt(evual_result, id.vars = c("model", 'week_ahead','seasonid'),
                         variable.name = "Metric", 
                         value.name = "Value")
index_level = c('RMSE','SMAPE','MAE','WIS','MAPE')
evual.long.result$Metric = factor(evual.long.result$Metric, level = index_level)

baseline_metric = evual.long.result %>%
  filter(model == 'Baseline') 

baseline_avg = baseline_metric%>%
  group_by(model, seasonid, Metric) %>%
  dplyr::summarize(avg_baseline = mean(Value, na.rm = TRUE))

evual.relative = evual.long.result %>%
  group_by(model, seasonid, Metric) %>%
  dplyr::summarize(avg_value = mean(Value, na.rm = TRUE)) %>%
  left_join(baseline_avg[,c('seasonid','Metric','avg_baseline')], 
            by=c("seasonid" = "seasonid","Metric" = 'Metric')) %>%
  mutate(relative_index = avg_value/avg_baseline)



shape_mod = levels(droplevels(proj.mode$seasonid))
val_shape = c(16,17)

season.relative <-evual.relative %>%
  ggplot() +
  geom_point(aes(x = model, y = relative_index, color = model,shape = seasonid, group=Metric), size = 2) +
  facet_wrap( ~ Metric, ncol = 5, scale = "free")+
  labs(y="Performance relative to baseline") + 
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_shape_manual("Season",labels=shape_mod,values = val_shape) +
  labs(title = 'Model performance in different season',tag = "A") + #
  theme_bw()+
  theme(panel.grid = element_blank(),
        strip.text.y = element_text(size=10, angle = 0))+
  theme( legend.text=element_text(size=9),
         legend.key.size = unit(5, "mm"))+
  theme(title = element_text(size=9),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size=9),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=9))+
  guides(color=guide_legend(ncol=1))


season.abs = baseline_metric %>%
  ggplot() + 
  geom_point(aes(x = week_ahead, y = Value, shape = seasonid, group=seasonid), 
             size = 2, color = 'black') +
  geom_line(aes(x = week_ahead, y = Value, group=seasonid),
            color = 'black', linetype = 'dashed') +
  facet_wrap( ~ Metric, ncol = 5, scale = "free")+
  scale_shape_manual("Season",labels=shape_mod,values = val_shape) +
  labs(y="Absolute metric value", x = 'Week') +
  labs(title = 'Baseline model performance in different seasons',tag = "B")+
  theme_bw()+
  theme(panel.grid = element_blank(),
        strip.text.y = element_text(size=10, angle = 0))+
  theme(title = element_text(size=9),
        axis.text.x = element_text(size=9),
        axis.text.y = element_text(size=9),
        axis.title.x = element_text(size=9),
        axis.title.y = element_text(size=9))+
  theme(legend.position="none")


## ----- plot combine ----
patchwork = p.trend | p.trend.ILI| season.relative
patchwork = patchwork + plot_annotation(tag_levels = 'A') + plot_layout(nrow = 3, heights = c(3,1.1,1.1), guides = "collect")

pdf(paste0('../Figures/',fig.name,".pdf"),width=10.8,height=11)

print(patchwork)
dev.off()

