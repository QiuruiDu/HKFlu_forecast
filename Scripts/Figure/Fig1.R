##############################################################################
# 1. preparation
##############################################################################
library(plyr)
library(dplyr)
library(tidyverse)
library(ISOweek)
library(locpol)
library(forecast)
library(scoringutils)
library(tseries)
library(cowplot)
library(tidyr)
library(rugarch)
library(scales)
library(lubridate)
library(tidyverse)
library(patchwork)

data_rt<-read.csv("../Data/data_flu_year.csv")
data_rt$date <- as.Date(data_rt$date)
date_analysis <- seq(as.Date("1998-01-02"),as.Date("2019-07-14"),by="week")
fig_name = 'Fig1_ILI_Trend'


data_rt = data_rt %>%
  filter(date >= as.Date('1998-01-04')) %>%
  mutate(year = as.numeric(year(date))) %>%
  mutate(monthid = as.numeric(strftime(date, format = "%m"))) %>%
  mutate(season = 'summer') %>%
  mutate(train_mode = 'train')

data_rt$season[which(data_rt$monthid %in% c(11,12,1,2,3,4))] = 'winter'
month_ids = unique(data_rt$monthid)
data_rt$monthid = factor(data_rt$monthid,levels=month_ids, ordered=T)


data_rt$train_mode[which(data_rt$date >= as.Date('2007-11-01'))] = 'test'


library(zoo)
## smooth the ILI curve
temp <- matrix(NA,nrow(data_rt)*7,3)
temp[,1] <- 1:nrow(temp)
temp[1:nrow(data_rt)*7,2] <- data_rt$rate
temp[1,2] <- 0
temp[nrow(temp),2] <- 0
temp[,3] <- na.approx(zoo(temp[,2]))
daily.date = seq(as.Date("1998-01-04"),as.Date("2019-07-14"),by="day") #1998-01-04
dat.daily = data.frame(date = daily.date, rate = temp[c(1:length(daily.date)),3])
dat.daily = dat.daily %>%
  mutate(date = as.Date(date)) %>%
  mutate(year = strftime(date, format = "%y")) %>%
  mutate(monthid = as.numeric(strftime(date, format = "%m"))) %>%
  mutate(season = 'summer')
dat.daily$season[which(dat.daily$monthid %in% c(11,12,1,2,3,4))] = 'winter'

dat.daily = dat.daily %>%
  left_join(data_rt[c('date','flu_weekid','flu_year')], by = c('date' = 'date'))

#############################################################################
# epidemic calculate
##############################################################################
# start
dat.epidemic <- data_rt %>%
  mutate(today_more_than_5 = 0) %>%
  mutate(tomorrow_more_than_5 = 0) %>%
  mutate(could_start_today = 0) %>%
  mutate(could_start_yesterday = 0) %>%
  mutate(start = 0) %>%
  mutate(month_2_start = 0) %>%
  mutate(final_start = 0)
dat.epidemic$today_more_than_5[which(dat.epidemic$rate/1000 > 0.005)] = 1
dat.epidemic$tomorrow_more_than_5[c(1:(nrow(dat.epidemic)-1))] = dat.epidemic$today_more_than_5[2:nrow(dat.epidemic)]
dat.epidemic$could_start_today[which(dat.epidemic$today_more_than_5+dat.epidemic$tomorrow_more_than_5 == 2)] = 1
dat.epidemic$could_start_yesterday[c(2:nrow(dat.epidemic))] = dat.epidemic$could_start_today[1:(nrow(dat.epidemic)-1)]
dat.epidemic$start[which(dat.epidemic$could_start_today == 1 & dat.epidemic$could_start_yesterday == 0)] = 1
dat.epidemic$month_2_start = dat.epidemic$start
for(i in c(1:8)){
  
  ii = i+1
  dat.epidemic$month_2_start[c(ii:nrow(dat.epidemic))] = dat.epidemic$month_2_start[c(ii:nrow(dat.epidemic))] + dat.epidemic$start[1:(nrow(dat.epidemic)-i)]
}
dat.epidemic$final_start[which(dat.epidemic$start == 1 & dat.epidemic$month_2_start <= 1)] = 1

dat.start = dat.epidemic%>%
  dplyr::select(date, natural_weekid, season, final_start) %>%
  mutate(date = as.Date(date))%>%
  left_join(data_rt[c('date','flu_weekid','flu_year')], by = c('date' = 'date'))

# end
dat.epidemic.end <- data_rt %>%
  mutate(today_more_than_5 = 0) %>%
  mutate(tomorrow_more_than_5 = 0) %>%
  mutate(could_end_today = 0) %>%
  mutate(could_end_yesterday = 0) %>%
  mutate(end = 0) 
dat.epidemic.end$today_more_than_5[which(dat.epidemic.end$rate/1000 > 0.005)] = 1
dat.epidemic.end$tomorrow_more_than_5[c(1:(nrow(dat.epidemic.end)-1))] = dat.epidemic.end$today_more_than_5[2:nrow(dat.epidemic.end)]
dat.epidemic.end$could_end_today[which(dat.epidemic.end$today_more_than_5+dat.epidemic.end$tomorrow_more_than_5 == 0)] = 1
dat.period = dat.start %>%
  filter(final_start == 1) %>%
  rename(start_date = date) %>%
  mutate(end_date = as.Date('1000-10-10')) 
for(i in c(1:nrow(dat.period))){
  start_date = dat.period$start_date[i]
  end_date = dat.epidemic.end %>%
    filter(could_end_today == 1) %>%
    filter(date > start_date)
  end_date = min(end_date$date)
  if((i < nrow(dat.period))&&(end_date >= dat.period$start_date[i+1])){
    end_date = dat.period$start_date[i+1] - 7
  }
  dat.period$end_date[i] = end_date
}


dat.daily.epidemic = dat.daily %>%
  mutate(date = as.Date(date)) %>%
  left_join(dat.start[,c(1,2,4,5)], by = c('date' = 'date'))
dat.daily.epidemic$final_start[is.na(dat.daily.epidemic$final_start)] = 0

# the following data is generated to plot the segment line
dat_segment = dat.epidemic[which(dat.epidemic$final_start == 1),]
break_date = seq(as.Date("1998-11-01"),as.Date("2019-11-02"),by="year")

year_list = seq(1998, 2019)
title_name = c(rep(" ",10), as.character(year_list[1]))
for(i in c(2:length(year_list))){
  title_name = c(title_name, rep(" ",3), as.character(year_list[i]))
}

trend.date2.v2 = dat.daily.epidemic %>%
  ggplot()+
  annotate(geom="rect",xmin=as.Date("1998-01-04"),xmax=as.Date("2007-11-01"),ymin=-Inf,ymax=Inf,fill="#F7F7F7",alpha=0.6)+
  annotate(geom="rect",xmin=as.Date("2007-11-01"),xmax=as.Date("2019-07-14"),ymin=-Inf,ymax=Inf,fill="#FFFACD",alpha=0.7)+
  annotate("text", x=as.Date("2002-06-01"), y=44, label= "train", size=4.5)+
  annotate("text", x=as.Date("2013-12-17"), y=44, label= "test", size=4.5)+
  geom_line(aes(x = date, y = rate, color = season, group = year),linewidth=0.8)+
  geom_segment(aes(x = dat_segment$date, xend = dat_segment$date,
                   y = 0, yend = 43), color = '#6E7BBB', data = dat_segment, lty=2,lwd=0.6)+
  scale_y_continuous("Influenza activity (ILI+)") +
  scale_x_date(breaks = break_date, label = date_format("%b"))+  
  xlab(paste(title_name, collapse = "")) + 
  coord_cartesian(ylim = c(0, 44))+
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(5, "mm"))+
  theme(axis.text.x = element_text(size = 9, hjust=0.5), 
        axis.text.y = element_text(size= 10),
        axis.title.y = element_text(hjust=0.5),
        axis.title.x = element_text(size=9)) +
  ggtitle("")+ 
  theme(title=element_text(size=11,
                           hjust=0.2,lineheight=0.2))+
  guides(color=guide_legend(ncol=1)) +
  guides(alpha = FALSE, linetype = FALSE)


##############################################################################
# 2. plot of same week in different year
##############################################################################
trend.week <- data_rt %>%
  mutate(flu_year = as.factor(flu_year)) %>%
  mutate(season = as.factor(season)) %>%
  ggplot() +
  geom_line(aes(x = flu_weekid, y = rate, color = season, linetype = '-', group = flu_year, alpha = flu_year), linewidth=0.8) + 
  scale_y_continuous("Influenza activity (ILI+)") +
  coord_cartesian(ylim = c(0, 43), xlim = c(1,53))+
  labs(alpha = "Flu Year")+
  # scale_x_continuous("Weeks ID in the year") + 
  theme_bw()+
  theme( legend.text=element_text(size=10, face = 'bold'),
         legend.key.size = unit(4, "mm"))+
  xlab("Epidemic Week") +
  theme(axis.text.x = element_text(face = 'bold'),
        axis.text.y = element_text(face = 'bold'),
        axis.title.y = element_text(hjust=0.5),
        axis.title.x = element_text(hjust=0.5))+
  ggtitle("")+ # ILI+ Trend Plot by WeekID of the Year
  theme(title=element_text(size=11,
                           hjust=0.2,lineheight=0.2))+#face="bold",
  guides(color = FALSE) + #color=guide_legend(ncol=1), 
  guides(linetype = FALSE)



patchwork = trend.date2.v2/trend.week
patchwork = patchwork +plot_annotation(tag_levels = 'A')+ plot_layout(nrow = 2, heights = c(1.2, 1), guides = "collect")&theme(legend.position='right')

path2 = paste0("../Figures/",fig_name,"_v2.pdf")
pdf(path2,width=11.3,height=8) 
print(patchwork)
dev.off() 

