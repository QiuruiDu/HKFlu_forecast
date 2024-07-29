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
library(rugarch)
library(scales)
library(patchwork)

source('./tools/MyPlot.R')

fig_name = 'Fig6_trajectory_2023_0248'

models = c(
  'AWAE_NC_R1',
  'AWBE_NC_R1'
)
model_name_list = c('Adaptive Weighted Average Ensemble (AWAE) Model',
                    'Adaptive Weighted Blending Ensemble (AWBE) Model')


index_list = c('rmse','mae','smape','mape','wis')
mode = 'test8_2023'
std_mode = 'ydiff'
dates_analysis = seq(as.Date("2023-03-05"),as.Date("2024-01-21"),by="week")
pred_horizon = 9

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
      filter(date <= max(dates_analysis)) 

    print(paste0('------model is', model,', data length = ',dim(proj_tmp)))
    proj<-rbind.fill(proj,proj_tmp)
  }
}
proj$inclusion<-1

proj$model<-factor(proj$model,levels=model_name_list, ordered=T)
proj$week_ahead<-as.factor(proj$week_ahead)
proj$date = as.Date(proj$date)

################################################################################
# plot single apart
################################################################################
lab_p <- c("0", '2','4','8')
proj.plot = proj %>%
  filter(week_ahead %in% c(0,2,4,8)) 
week_level = c(8,4,2,0)

val_p<-c("#FFBBFF","powderblue","navajowhite",brewer.pal(10,"Paired")[5])
val_p2<-c("#68228B","turquoise4","burlywood4",brewer.pal(10,"Paired")[6])

myvar<-"iHosp"
remove_last_n <- 1

max_prediction_horizon<-8+remove_last_n

data_rt<-readRDS(paste0("../Data/ILI_data_2023.rds"))
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



p.1 = proj.plot %>%
  filter(model == 'Adaptive Weighted Average Ensemble (AWAE) Model') %>%
  mutate(prediction_horizon2 = factor(week_ahead,levels=week_level,ordered = T)) %>% 
  ggplot() +
  ggnewscale::new_scale_fill()+
  geom_ribbon(aes(x= date, ymin= lower_10, ymax= upper_10, 
                  fill=as.factor(prediction_horizon2)),alpha=0.7)+
  geom_line(aes(x = date, y = point, color = as.factor(prediction_horizon2))) +
  geom_line(data = dat_true, aes(x = date, y = smooth_value)) +
  xlab(paste(c(rep(" ",32), "2023", rep(" ",75),"2024"), collapse = "")) + 
  ggtitle("Adaptive Weighted Average Ensemble (AWAE) Model")+
  theme_bw()+
  scale_y_continuous("Influenza activity (ILI+)", limits = c(0, NA)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b")+
  theme(axis.text.x = element_text(angle = 0, hjust=0.5),
        axis.title.x = element_text(size=10, vjust = 0.3))+
  scale_color_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p2) +
  scale_fill_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p) +
  scale_size_manual("",labels="Data",values=0.8) 

p.2 = proj.plot %>%
  filter(model == 'Adaptive Weighted Blending Ensemble (AWBE) Model') %>%
  mutate(prediction_horizon2 = factor(week_ahead,levels=week_level,ordered = T)) %>% 
  ggplot() +
  ggnewscale::new_scale_fill()+
  geom_ribbon(aes(x= date, ymin= lower_10, ymax= upper_10, 
                  fill=as.factor(prediction_horizon2)),alpha=0.7)+
  geom_line(aes(x = date, y = point, color = as.factor(prediction_horizon2))) +
  geom_line(data = dat_true, aes(x = date, y = smooth_value)) +
  xlab(paste(c(rep(" ",32), "2023", rep(" ",75),"2024"), collapse = "")) + 
  ggtitle("Adaptive Weighted Blending Ensemble (AWBE) Model")+
  theme_bw()+
  scale_y_continuous("Influenza activity (ILI+)", limits = c(0, NA)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b")+
  theme(axis.text.x = element_text(angle = 0, hjust=0.5),
        axis.title.x = element_text(size=10, vjust = 0.3))+
  scale_color_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p2) +
  scale_fill_manual("Prediction\nhorizon", labels = rev(lab_p), values = val_p) +
  scale_size_manual("",labels="Data",values=0.8) 

patchwork = p.1 + p.2 + plot_annotation(tag_levels = 'A') + plot_layout(nrow = 2, guides = "collect")  
pdf(paste0('../Figures/',fig_name,'_v2.pdf'),width=7,height=7) 
patchwork
dev.off()

