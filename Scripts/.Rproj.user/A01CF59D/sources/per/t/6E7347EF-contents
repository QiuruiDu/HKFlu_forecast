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
library(tseries)
library(cowplot)

data_ot<-readRDS("../Data/ILI_data_2023.rds")
data_rt <- data_ot %>%
  mutate(date = as.Date(date)) %>%
  mutate(log_10rate = log(10*rate)) %>%
  mutate(iHosp = log_10rate) %>%
  mutate(iHosp_smooth = log_10rate) %>%
  dplyr::select(date,region,iHosp,date_analysis,rate,log_10rate,iHosp,iHosp_smooth,temp.max,temp.min, relative.humidity,
                total.rainfall, solar.radiation,
                monthid, weekid)


max_lag = 14
model_name = 'arima_rolling'
covar_all<-c("temp.max", "temp.min", "relative.humidity", "total.rainfall",
             "solar.radiation")
# number of removed data points (not consolidated) # for HK 1 day delay is OK
remove_last_n <- 1
# maximum prediction horizon
max_prediction_horizon <- 8+remove_last_n
mode = 'test8_2023'

dates_analysis = seq(as.Date("2022-11-06"),as.Date("2024-03-03"),by="week")
regions <-c("HK")

library(tibble)
res<-tibble()

for (iDate in seq_along(dates_analysis)){
  set.seed(iDate)
  print(dates_analysis[iDate])
  dat_train<-data_rt %>%
    filter(date_analysis==dates_analysis[iDate]) %>%
    mutate(iHosp_r_covar=iHosp)
  
  # Estimate best lags for covariates
  res_lag<-tibble()
  list_col<-sort(covar_all)
  n_predictors <- length(list_col)
  
  if (n_predictors>0){
    for (iCol in seq_along(list_col)){
      # iCol = 1
      mycol<-list_col[iCol]
      
      for (lag in 1:max_lag){
        # lag = 1
        sub1<-dat_train
        sub2<-sub1
        sub2$date<-sub2$date+lag*7
        
        sub<-left_join(sub1[,c("date","region","iHosp")],sub2[,c("date","region",mycol)], by = c("date", "region"))
        sub<-sub %>% drop_na(iHosp,mycol) %>%
          as.data.frame()
        
        cor<-cor.test(y=sub$iHosp,x=sub[,mycol],method="pearson")$estimate
        
        res_lag_tmp<-data.frame(pearson=cor,region="regions",lag=lag,var=mycol)
        res_lag<-bind_rows(res_lag,res_lag_tmp)
        
      }
    }
    
    
    best_lag<-res_lag %>% 
      group_by(var) %>%
      dplyr::slice(which.max(abs(pearson)))
    if ("iHosp_r_covar" %in% list_col){
      best_lag$lag[best_lag$var=="iHosp_r_covar"]<-1
    }
    
    list_lag<-best_lag$lag
    print(list_lag)
  }
  
  for (iRegion in seq_along(regions)) {
    
    myreg<-regions[iRegion]
    
    
    dat_train_reg<-dat_train %>%
      filter(region==myreg)
    
    # Prepare lagged data
    sub<-dat_train_reg[,c("date","region","iHosp","weekid",'monthid')]
    sub1<-dat_train_reg[,c("date","region","iHosp","weekid",'monthid',list_col)]
    lag_list_col = c()
    if (n_predictors>0){
      for (i_col in 1:length(list_lag)){
        col_name = best_lag$var[i_col]
        max_lag = best_lag$lag[i_col]
        print(paste0(col_name,' max_lag - ', max_lag))
        for(i_lag in c(1:max_lag)){
          sub2<-sub1
          sub2$date<-sub2$date+i_lag*7
          
          sub<-left_join(sub,sub2[,c("date","region",col_name)],by=c("region"="region","date"="date"))
          colnames(sub)[ncol(sub)] = paste0(col_name,i_lag)
          lag_list_col = c(lag_list_col, paste0(col_name,i_lag))
        }
        
      }
    }
    sub<-sub %>% drop_na(iHosp,all_of(lag_list_col)) %>%
      dplyr::select(-date,-region)
    
    # Fit model 
    mod<-auto.arima(sub[,"iHosp"], xreg=as.matrix(sub[,c(2:ncol(sub))]),method="ML",allowdrift = F) # 
    
    # build new data to predict
    newdata<-data.frame(seq.Date(dates_analysis[iDate]-remove_last_n+1,dates_analysis[iDate]+(max_prediction_horizon-1)*7,by="week"),NA,NA,NA)
    names(newdata)<-c("date","iHosp","weekid",'monthid')
    sub1<-dat_train_reg[,c("date","iHosp","weekid",'monthid',list_col)]
    
    if (n_predictors>0){
      for (i_col in 1:length(list_lag)){
        col_name = best_lag$var[i_col]
        max_lag = best_lag$lag[i_col]
        print(paste0(col_name,' max_lag - ', max_lag))
        for(i_lag in c(1:max_lag)){
          sub2<-sub1
          sub2$date<-sub2$date+i_lag*7
          
          newdata<-left_join(newdata,sub2[,c("date",col_name)],by=c("date"="date"))
          # fill NA by last available value
          newdata[is.na(newdata[,ncol(newdata)]),ncol(newdata)]<- last(newdata[!is.na( newdata[,ncol(newdata)]),ncol(newdata)])
          
          colnames(newdata)[ncol(newdata)] = paste0(col_name,i_lag)
        }
      }
    }
    newdata$weekid <- (dat_train$weekid[nrow(dat_train)]+1:max_prediction_horizon-1)%%52+1
    newdata$monthid <- as.numeric(strftime(newdata$date, format = "%m"))
    
    # Predict growth rate
    out <- as.data.frame(forecast(mod,max_prediction_horizon,xreg=as.matrix(newdata[,-c(1,2)])))
    
    pred <- data.frame(point = out$`Point Forecast`, point_avg = out$`Point Forecast`)

    pred = exp(pred)/10 
    
    res_tmp<- data.frame(var="iHosp",
                         region=myreg,
                         date=dates_analysis[iDate],
                         prediction_horizon=seq(1,max_prediction_horizon)-remove_last_n,
                         pred) 
    
    
    res<-rbind(res,res_tmp)
    
  }
  
}
model = 'arima'
res1 <- res %>%
  mutate(date=as.Date(date)) %>%
  mutate(date_pred=date+prediction_horizon*7) %>%
  mutate(model=model) 
res1 = res1 %>%
  filter(date_pred>=min(dates_analysis))  %>%
  filter(date_pred<=max(dates_analysis))

dat_true<-data_rt %>%
  filter(date_analysis==max(data_rt$date_analysis)) %>%
  filter(date>=min(dates_analysis)-remove_last_n & date<=max(dates_analysis)) %>%
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
  rename(week_ahead = prediction_horizon) 

col_list = c('date','true','week_ahead', 'point','point_avg')
res1 <- res1[, col_list]


## save
if(dir.exists("../Results/Point/") == TRUE){
  write.csv(res1, paste0('../Results/Point/forecast_', model_name,'_',mode,'.csv'), row.names = FALSE)
}else{
  dir.create('../Results/Point/', recursive = TRUE)
  write.csv(res1, paste0('../Results/Point/forecast_', model_name,'_',mode,'.csv'), row.names = FALSE)
}