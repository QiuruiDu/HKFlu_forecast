####################
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
# run tim baseline
origin_path = '/Users/hkuph/richael/RA_academic/flu/SBEM_for_HKILI'
setwd(origin_path)

data_rt<-readRDS("./Data/ILI_data_2023.rds")
data_rt <- data_rt %>%
  mutate(date = as.Date(date)) %>%
  mutate(log_10rate = log(10*rate)) %>%
  mutate(iHosp = log_10rate) %>%
  mutate(iHosp_smooth = log_10rate) %>%

  dplyr::select(date,region,iHosp,date_analysis,rate,log_10rate,iHosp,iHosp_smooth,temp.max,temp.min, relative.humidity,
                total.rainfall, solar.radiation,
                monthid, weekid)
mode = 'test8_2023'
dates_analysis<- seq(as.Date("2022-11-06"),as.Date("2024-03-10"),by="week")

regions <-c("HK")

# number of removed data points (not consolidated) # for HK 1 day delay is OK
remove_last_n <- 1
# maximum prediction horizon
max_prediction_horizon<-8+remove_last_n
date_max<-max(dates_analysis)+max_prediction_horizon-remove_last_n
model_name = 'baseline'

library(msm)

null_model_forecast_quantiles <- function(values, horizon, truncation = 0)
{
  # values = dat_train$iHosp_smooth
  # horizon = max_prediction_horizon
  # truncation = 0
  if (truncation > 0) {
    values <- head(values, -truncation)
  }
  
  tnorm_unc_fit <- function(x, mean, true) {
    return(-sum(dtnorm(x = true, mean = mean, sd = x, lower = 0, log = TRUE)))
  }
  
  quantiles <- c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)
  ts <- tail(values, min(horizon, length(values)))
  
  if (length(ts) > 1) {
    ## null model
    null_ts <- rep(last(ts), horizon + truncation)
    
    interval <- c(0, max(abs(diff(ts))))
    
    if (diff(interval) > 0 ) {
      tnorm_sigma <-
        optimise(tnorm_unc_fit, interval = interval,
                 mean = head(ts, -1),
                 true = tail(ts, -1))
      
      quant <- qtnorm(p = quantiles, mean = last(ts),
                      sd = tnorm_sigma$minimum, lower = 0)
    } else {
      quant <- rep(last(ts), length(quantiles))
    }
  } else {
    quant <- rep(last(ts), length(quantiles))
  }
  quant = exp(quant)/10  #return the true inverse-log rate
  names(quant) <- quantiles
  
  return(quant)
}


library(tibble)
res <- tibble()
library(dplyr)
iRegion = 'HK'

for (iDate in seq_along(dates_analysis)) {
  # iDate = 1
  dat_train<-data_rt %>%
    filter(date_analysis==dates_analysis[iDate], region==iRegion)  %>%
    as.data.frame()
  
  quant <-
    null_model_forecast_quantiles(dat_train$iHosp_smooth, horizon = max_prediction_horizon,
                                  truncation = 0)
  
  names(quant)<-c(paste0("lower_",c(2,5,seq(10,90,by=10))),"point",paste0("upper_",rev(c(2,5,seq(10,90,by=10)))))
  
  
  res_tmp <- data.frame(
    var = "iHosp",
    region = iRegion, 
    date = dates_analysis[iDate],
    prediction_horizon = seq(1,max_prediction_horizon)-remove_last_n,
    t(quant)
    
  )
  
  res <- rbind(res, res_tmp)
  
  
}

model = 'baseline'
res1 <- res %>%
  mutate(date=as.Date(date)) %>%
  mutate(date_pred=date+prediction_horizon*7) %>%
  mutate(model=model) 

res1 = res1 %>%
  filter(date_pred>=min(dates_analysis))  %>%
  filter(date_pred<=max(dates_analysis))

dat_true<-data_rt %>%
  filter(date_analysis==max(data_rt$date_analysis)) %>%
  filter(date>=min(dates_analysis)-remove_last_n & date<=date_max) %>%
  dplyr::select(date,region,iHosp_smooth) %>%
  pivot_longer(cols=c("iHosp_smooth"),
               names_to="var",values_to="smooth_value") %>%
  mutate(var=gsub("_smooth","",var)) %>%
  filter(!is.element(region,c("COR","national")))

dat_true$smooth_value = exp(dat_true$smooth_value)/10


res1<-res1 %>%
  left_join(dat_true,by=c("date_pred"="date","region"="region","var"="var")) 

res1 = res1[ , -which(colnames(res1) %in% c("date"))]
res1 = res1 %>% 
  rename(date = date_pred) %>% 
  rename(true = smooth_value) %>% 
  rename(week_ahead = prediction_horizon) %>%
  mutate(point_avg = point)

col_list = c('date','true', 'week_ahead', 'point', 'point_avg')
res1 <- res1[, col_list]

######################### ------- save ---------#################################
path = paste0(origin_path, '/Results/Point/forecast_', model_name,'_',mode,'.csv')
write.csv(res1, path, row.names = FALSE)
