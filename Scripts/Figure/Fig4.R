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
  'SAE',
  'NBE',
  'AWAE',
  'AWBE'
)
model_name_list = c('Simple Average Ensemble (SAE) Model', 
                    'Normal Blending Ensemble (NBE) Model',
                    'Adaptive Weighted Average Ensemble (AWAE) Model',
                    'Adaptive Weighted Blending Ensemble (AWBE) Model'
                    )

fig_name = 'Fig4_interval_plot_only_ensemble'
mode = 'test8'
dates_gap = seq(as.Date("2009-04-05"),as.Date("2010-03-21"),by="week")
if(mode == 'test4'){
  pred_horizon = 5
}else{
  pred_horizon = 9
}
dates_analysis <- seq(as.Date("2007-11-04"),as.Date("2019-07-14"),by="week")
################################################################################
# read data
################################################################################
proj <-tibble()
for (i in c(1:length(models))){

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
    print(paste0('------model is', model,', data length = ',dim(proj_tmp)))
    proj<-rbind.fill(proj,proj_tmp)
  }
}
proj$inclusion<-1

proj.combine = proj 

proj.combine$model<-factor(proj.combine$model,levels=model_name_list, ordered=T)
proj.combine$week_ahead<-as.factor(proj.combine$week_ahead)
################################################################################
# plot single apart
################################################################################
lab_p <- c("0", "4",'8')
val_p<-c("powderblue","navajowhite",brewer.pal(10,"Paired")[5])
val_p2<-c("turquoise4","burlywood4",brewer.pal(10,"Paired")[6])

myvar<-"iHosp"
remove_last_n <- 1

max_prediction_horizon<-8+remove_last_n

setwd(origin_path)
data_rt<-readRDS(paste0("../Data/data_lograte.rds"))
data_rt$date <- as.Date(data_rt$date)
data_rt$date <- as.Date(data_rt$date)
data_rt$iHosp <- data_rt$rate
dat_true<-data_rt %>%
  filter(date_analysis==max(data_rt$date_analysis)) %>%
  filter(date>=min(dates_analysis) & date<=max(dates_analysis)) %>%
  dplyr::select(date,region,iHosp) %>%
  pivot_longer(cols=c("iHosp"),
               names_to="var",values_to="smooth_value") %>%
  mutate(var=gsub("_smooth","",var)) %>%
  filter(!is.element(region,c("COR","national")))


max.point = max(proj.combine$upper_10[which((proj.combine$date < min(dates_gap))|(proj.combine$date > max(dates_gap)))]) 
label_dates = seq(as.Date("2008-01-06"),as.Date("2020-01-06"),by="year")

p <-proj.combine %>%
  filter(week_ahead %in% c(0,4,8)) %>%
  mutate(prediction_horizon2 = factor(week_ahead,levels=c(8,4,0),ordered = T)) %>% 
  ggplot() +
  geom_ribbon(aes(x= date, ymin= lower_10, ymax= upper_10, 
                  fill=as.factor(prediction_horizon2)),alpha=0.7)+
  geom_line(aes(x = date, y = point, color = as.factor(prediction_horizon2))) +
  annotate(geom="rect",xmin=min(dates_gap),xmax=max(dates_gap),ymin=-Inf,ymax=Inf,fill="#F0F0F0",alpha=1)+
  geom_line(data = dat_true, aes(x = date, y = smooth_value)) +
  facet_wrap( ~ model, ncol = 1, scale = "free")+
  xlab("") + 
  theme_bw()+
  scale_x_date(breaks = label_dates, label = date_format("%Y %b"))+
  theme(axis.text.x = element_text(angle = 45, hjust=0.8))+
  scale_y_continuous("Influenza activity (ILI+)", limits = c(0, max.point)) +
  scale_color_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p2) +
  scale_fill_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p) +
  scale_size_manual("",labels="Data",values=0.8) + 
  coord_cartesian(xlim=c(min(dates_analysis),max(dates_analysis)))


pdf(paste0('../Figures/',fig_name,'.pdf'),width=7.8,height=11) 
print(p)
dev.off()
