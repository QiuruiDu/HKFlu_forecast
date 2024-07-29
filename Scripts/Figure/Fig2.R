################################################################################
# preparation
################################################################################
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

source('./tools/MyPlot.R')

models = c(
  'baseline', 
  'ARIMA_rolling',
  'GARCH_rolling',
  'RF_rolling',
  'XGB_rolling',
  'InTimePlus_v3_nontuning_rolling_v2',
  'TSTPlus_v3_nontuning_rolling',
  'LSTM_v3_nontuning_rolling_v2',
  'GRU_v3_nontuning_rolling_v2',
  'SAE',
  'NBE',
  'AWAE',
  'AWBE'
)
model_name_list = c()
for(st in models){
  model_name_list = c(model_name_list, strsplit(st, '_', fixed = F, perl = F, useBytes = F)[[1]][1])
}
index_list = c('rmse','smape','mae','mape','wis')
fig_name = 'Fig2_metric_summary_add_WIS_plot'

################################################################################
# point result
################################################################################
models.fig2 = models
model_name.fig2 = model_name_list

mode = 'test8'
dates_gap = seq(as.Date("2009-04-05"),as.Date("2010-03-21"),by="week")
if(mode == 'test4'){
  pred_horizon = 5
}else{
  pred_horizon = 9
}
dates_analysis <- seq(as.Date("2007-11-04"),as.Date("2019-07-14"),by="week")


################################################################################
# interval result
################################################################################
# read interval windows length
std_mode = 'ydiff'
proj.interval <-tibble()
for (i in c(1:length(models))){
  # i = 12
  # print(model)
  model = models[i]
  path_ = paste0('../Results/Interval_',std_mode,'_pred/interval_',model,'_',mode,'.csv')
  if(file.exists(path_)){
    proj_tmp<-read.csv(file=path_,stringsAsFactors =F)
    
    proj_tmp$model = model_name_list[i]
    
    proj_tmp = proj_tmp %>%
      mutate(date=as.Date(date)) %>%
      mutate(point = point_avg) %>%
      mutate(date_origin=date-week_ahead*7) %>%
      filter(date >= min(dates_analysis)) %>%
      filter(date <= max(dates_analysis)) %>%
      filter(date_origin >= min(dates_analysis))
    
    proj_tmp = proj_tmp[which((proj_tmp$date_origin < min(dates_gap))|(proj_tmp$date_origin > max(dates_gap))),]

    proj.interval<-rbind.fill(proj.interval,proj_tmp)
  }
}


res.interval<-my.compute(proj.interval)
evual.result.interval <-res.interval %>%
  mutate(cov_true = 1) %>%
  group_by(model,week_ahead) %>%
  dplyr::summarize(rmse = sqrt(mean(abs_error^2,na.rm=T)) ,
                   smape = (mean(abs_error/(abs(point)+abs(true))*2,na.rm=T))
                   ,mape = (mean(abs_error/abs(true),na.rm=T))
                   ,mae = mean(abs_error, na.rm=T)
                   ,avg_wis = (mean(wis,na.rm=T)))%>%
  ungroup() 

evual.result.final = evual.result.interval %>% 
  dplyr::rename(wis = avg_wis)
evual.result.final$model = factor(evual.result.final$model,levels=model_name_list, ordered=T)

evual.long.result = melt(evual.result.final, id.vars = c("model", 'week_ahead'))


baseline_avg = evual.long.result %>%
  filter(model == 'baseline') %>%
  group_by(model, variable) %>%
  dplyr::summarize(avg_baseline = mean(value, na.rm = TRUE))


evual.relative = evual.long.result %>%
  group_by(model, variable) %>%
  dplyr::summarize(avg_value = mean(value, na.rm = TRUE)) %>%
  left_join(baseline_avg[,c('variable','avg_baseline')], by=c("variable" = 'variable')) %>%
  mutate(relative_index = avg_value/avg_baseline)

res.index.abs = evual.result.final

res.index.relative <- dcast(data.table(evual.relative), model  ~variable, value.var = 'relative_index')
res.index.relative$week_ahead = pred_horizon-1+0.05


evual.rank = evual.long.result %>%
  group_by(model, variable) %>%
  dplyr::summarize(avg_value = mean(value, na.rm = TRUE)) %>%
  left_join(baseline_avg[,c('variable','avg_baseline')], by=c("variable" = 'variable')) %>%
  mutate(relative_index = avg_value/avg_baseline) %>%
  modify_if(~is.numeric(.), ~round(., 2)) %>%
  mutate(rank = 0) %>%
  arrange(variable,relative_index)
for(v in unique(evual.rank$variable)){
  evual.rank$rank[which(evual.rank$variable == v)] = rank(evual.rank$relative_index[which(evual.rank$variable == v)], ties.method = 'min')
}
# evual.rank$rank <- ave(1:nrow(evual.rank), evual.rank$variable,FUN=rank)
evual.rank$model2<-factor(evual.rank$model,levels=model_name_list,ordered=T) 
evual.rank$variable = factor(evual.rank$variable,levels=index_list,ordered=T)

################################################################################
# plot
################################################################################
#### -- week
week_list = c(0,2,4,6,8,pred_horizon-1+0.05)
res.index.abs = res.index.abs %>%
  filter(week_ahead %in% week_list)
res.index.relative = res.index.relative %>%
  filter(week_ahead %in% week_list)

### --- plot

pal<-c("black",brewer.pal(10,"Paired"),"turquoise",brewer.pal(6, "Dark2")[4:6])
val_size<-c(rep(0.5,(length(model_name_list)-2)),c(0.8,0.8))
val_linetype<-c("dashed", rep("solid",length(model_name_list)-1))
lab_mod<-levels(droplevels(res.index.abs$model))
val_linewidth = c(rep(0.5, (length(model_name_list)-2)), c(0.8,0.8))
val_point_size = c(rep(1.3, (length(model_name_list)-2)), c(1.8,1.8))


rmse1 = ggplot() +
  geom_line(data = res.index.abs, aes(x = week_ahead, y = rmse, color = model ,
                                      linetype=model, size = model)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous("Weeks ahead", breaks = unique(res.index.abs$week_ahead)) +
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_linewidth) +
  scale_linetype_manual("Model",labels=lab_mod,values=val_linetype)



rmse.total = rmse1 +
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size=10, hjust = 0.5, vjust = .5, face = 'bold'),
        axis.text.y = element_text(size=10, face = 'bold'),
        axis.title.x = element_text(size=11, face = 'bold'),
        axis.title.y = element_text(size=11, face = 'bold'))+
  guides(color=guide_legend(title = "Model")) +
  labs(tag = "A")+
  ggtitle("RMSE")


mape1 = ggplot() +
  geom_line(data = res.index.abs, aes(x = week_ahead, y = mape, color = model ,
                                      linetype=model, size = model)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous("Weeks ahead", breaks = unique(res.index.abs$week_ahead)) +
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_linewidth) +
  scale_linetype_manual("Model",labels=lab_mod,values=val_linetype)



mape.total = mape1 + 
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size=10, hjust = 0.5, vjust = .5, face = 'bold'),
        axis.text.y = element_text(size=10, face = 'bold'),
        axis.title.x = element_text(size=11, face = 'bold'),
        axis.title.y = element_text(size=11, face = 'bold'))+
  guides(color=guide_legend(title = "Model")) +
  labs(tag = "A")+
  ggtitle("MAPE")

mae1 = ggplot() +
  geom_line(data = res.index.abs, aes(x = week_ahead, y = mae, color = model ,
                                      linetype=model, size = model)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous("Weeks ahead", breaks = unique(res.index.abs$week_ahead)) +
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_linewidth) +
  scale_linetype_manual("Model",labels=lab_mod,values=val_linetype)


mae.total = mae1 + 
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size=10, hjust = 0.5, vjust = .5, face = 'bold'),
        axis.text.y = element_text(size=10, face = 'bold'),
        axis.title.x = element_text(size=11, face = 'bold'),
        axis.title.y = element_text(size=11, face = 'bold'))+
  guides(color=guide_legend(ncol=1)) +
  labs(tag = "A")+
  theme(legend.position="none") +
  ggtitle("MAE")


smape1 = ggplot() +
  geom_line(data = res.index.abs, aes(x = week_ahead, y = smape, color = model ,
                                      linetype=model, size = model)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous("Weeks ahead", breaks = unique(res.index.abs$week_ahead)) +
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_linewidth) +
  scale_linetype_manual("Model",labels=lab_mod,values=val_linetype)



smape.total = smape1 + 
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size=10, hjust = 0.5, vjust = .5, face = 'bold'),
        axis.text.y = element_text(size=10, face = 'bold'),
        axis.title.x = element_text(size=11, face = 'bold'),
        axis.title.y = element_text(size=11, face = 'bold'))+
  guides(color=guide_legend(ncol=1)) +
  labs(tag = "A")+
  ggtitle("SMAPE")


wis1 = ggplot() +
  geom_line(data = res.index.abs, aes(x = week_ahead, y = wis, color = model ,
                                      linetype=model, size = model)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous("Weeks ahead", breaks = unique(res.index.abs$week_ahead)) +
  scale_color_manual("Model",labels=lab_mod,values=pal) +
  scale_size_manual("Model",labels=lab_mod,values=val_linewidth) +
  scale_linetype_manual("Model",labels=lab_mod,values=val_linetype)

wis.total = wis1 + 
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size=10, hjust = 0.5, vjust = .5, face = 'bold'),
        axis.text.y = element_text(size=10, face = 'bold'),
        axis.title.x = element_text(size=11, face = 'bold'),
        axis.title.y = element_text(size=11, face = 'bold'))+
  guides(color=guide_legend(ncol=1)) +
  labs(tag = "A")+
  ggtitle("WIS")



text.size = c(rep(10, (length(models.fig2)-2)), c(10,10))
test.color = c(rep('black', (length(models.fig2)-2)), rep('black',2))
val.rank<-ggplot(evual.rank, 
                 aes(y= model2, fill=rank, x= variable)) + 
  geom_tile() + ylab(NULL) + xlab(NULL) +
  geom_text(aes(label=round(relative_index, 2)), size=5) +
  scale_fill_gradient(name = "Rank of\nModel Performance",low ="dodgerblue4",high = "white", limits=c(1,length(models)+1)) +
  theme(axis.text.y = element_text(size=text.size, 
                                   face = 'bold', color = test.color, angle = 0),
        axis.text.x = element_text(size=11, face = 'bold',angle = 0))+
  ggtitle("Performance relative to baseline")


layout <- "
112233
445577
666677
"
patchwork = rmse.total+smape.total+wis.total+mape.total+mae.total+val.rank+ guide_area()
patchwork = patchwork + plot_annotation(tag_levels = 'A') + plot_layout(design = layout, nrow = 3, heights = c(1,1,2.6), guides = "collect")


pdf(paste0('../Figures/',fig_name,".pdf"),width=8.3,height=9.5)
print(patchwork)
dev.off()

