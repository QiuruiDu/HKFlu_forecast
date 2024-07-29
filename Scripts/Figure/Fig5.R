# 进行结果的计???
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
library(rugarch)
library(scales)
library(patchwork)

# origin_path = '/Users/hkuph/richael/RA_academic/flu/SBEM_for_HKILI'
# setwd(origin_path)

source('./tools/MyPlot.R')

# source(paste0(origin_path,'/Scripts/tools/MyPlot.R'))

fig_type = 'Fig5_feature_importance'


models = c(
  'ARIMA','GARCH','RF','XGB','LSTM','GRU','InTimePlus','TSTPlus'
)
mode = 'test8'
model_name = models
dates_analysis_test <- seq(as.Date("2007-11-04"),as.Date("2019-07-14"),by="week")
dates_analysis_val <- seq(as.Date("2003-11-02"),as.Date("2007-10-28"),by="week")
dates_gap = seq(as.Date("2009-04-05"),as.Date("2010-03-21"),by="week")
if(mode == 'test4'){
  pred_horizon = 5
}else{
  pred_horizon = 9
}

origin_predictor = c('rate','weekid','monthid', 'relative.humidity', 'solar.radiation', 'temp.max','temp.min','total.rainfall')
final_predictor = c('Historical ILI+', 'Number of week in a year','Number of month in a year',
                    'Relative humidity', 'Solar radiation','Maximum of temperature', 'Minimum of temperature',
                    'Total rainfall')


proj<-tibble()
for (i in c(1:length(models))){
  model = models[i]
  date_origin_max = max(dates_analysis_test)-(pred_horizon-1)*7
  date_origin_min = min(dates_analysis_test)
  path_ = paste0('../Results/FI/fi_full_',model,'_', mode,'.csv')
  if(file.exists(path_)){

    proj_tmp<-read.csv(file=path_,stringsAsFactors =F)
    proj_tmp$model = model_name[i]
    proj_tmp = proj_tmp %>%
      mutate(date=as.Date(date)) %>%
      filter(date >= date_origin_min) %>%
      filter(date <= date_origin_max)
    proj_tmp = proj_tmp[which((proj_tmp$date < min(dates_gap))|(proj_tmp$date > max(dates_gap))),]
    proj<-rbind.fill(proj,proj_tmp)
  }
}

proj$model = factor(proj$model , level = models)

for(ip in c(1:length(origin_predictor))){
  proj$predictor[which(proj$predictor == origin_predictor[ip])] = final_predictor[ip]
}

lag_orders = c()
for(il in c(1:14)){
  lag_orders = c(lag_orders, paste0(il,'weeks'))
  proj$lag_order[which(proj$lag_order == paste0(il,'d'))] = paste0(il,'weeks')
}

proj$lag_order = factor(proj$lag_order , level = lag_orders)
proj$predictor = factor(proj$predictor , level = final_predictor)


######## -------
proj.avg =  proj %>%
  group_by(week, predictor, lag_order, model) %>%
  dplyr::summarize(avg_metric_change = mean(metric_change))%>%
  ungroup()

proj.max <- proj.avg %>%
  group_by(week,model) %>%
  dplyr::summarize(max_avg_metric_change = max(avg_metric_change))%>%
  ungroup()

proj.min <- proj.avg %>%
  group_by(week, model) %>%
  dplyr::summarize(min_avg_metric_change = min(avg_metric_change))%>%
  ungroup()


proj.norm = proj.avg %>%
  left_join(proj.max, by = c('week' = 'week', 'model' = 'model')) %>%
  left_join(proj.min, by = c('week' = 'week', 'model' = 'model')) %>%
  mutate(relative_metric_change = (avg_metric_change - min_avg_metric_change)/(max_avg_metric_change - min_avg_metric_change))


lag_list= paste0(c(2,4,6,8,10,12,14),'weeks')

fi.hot = proj.norm %>%
  filter(week %in% c('week0','week4','week8')) %>%
  ggplot(aes(y= predictor, fill=relative_metric_change, x= lag_order)) + 
  geom_tile() + ylab(NULL) + xlab(NULL) +
  scale_fill_gradient(low ="white",high = "#CD3700") +
  scale_x_discrete("Lag Order",breaks = lag_list)+
  scale_y_discrete("Predictors")+
  facet_grid(model~week, scale = "free") +

  theme()+
  theme(
    strip.text.x = element_text(
      size = 11, face = "bold"
    ), 
    strip.text.y = element_text(
      size = 11, face = "bold"
    ) 
  )+
  theme(axis.text.y = element_text(size=9,angle = 0),
        axis.text.x = element_text(size=10,angle = 45, vjust = 0.5))+
  guides(fill=guide_legend(title="Feature\nImportance"))+
  ggtitle("Feature Importance of predictors")


pdf(paste0('../Figures/',fig_type,".pdf"),width=8,height=10)
print(fi.hot)
dev.off()

