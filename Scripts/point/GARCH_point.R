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
library(tidyr)
library(rugarch)

data_rt<-readRDS("../Data/data_lograte.rds")
data_rt$date <- as.Date(data_rt$date)
data_rt$iHosp <- data_rt$log_10rate
data_rt$iHosp_smooth <- data_rt$log_10rate


max_lag = 14
model_name = 'GARCH_rolling'
covar_all<-c("temp.max", "temp.min", "relative.humidity", "total.rainfall",
             "solar.radiation")
dates_analysis = seq(as.Date("2003-11-02"),as.Date("2019-07-14"),by="week") #2017-10-29
regions <-c("HK")
mode = 'test8'


# number of removed data points (not consolidated) # for HK 1 day delay is OK
remove_last_n <- 1
max_prediction_horizon <- 8 + remove_last_n


res<-tibble()

for (iDate in seq_along(dates_analysis)){
  # 
  set.seed(iDate)
  print(dates_analysis[iDate])
  dat_train<-data_rt %>%
    filter(date_analysis==dates_analysis[iDate]) %>%
    mutate(iHosp_r_covar=iHosp)
  
  # Estimate best lags 
  res_lag<-tibble()
  list_col<-sort(covar_all)
  n_predictors <- length(list_col)
  
  if (n_predictors>0){
    for (iCol in seq_along(list_col)){
      
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
      
      for (i in 1:length(list_lag)){
        col_name = best_lag$var[i]
        lag_list_col = c(lag_list_col, col_name)
        print(paste0(col_name, ', lag - ', list_lag[i]))
        sub2<-sub1
        sub2$date<-sub2$date+list_lag[i]*7
        
        sub<-left_join(sub,sub2[,c("date","region",list_col[i])],by=c("region"="region","date"="date"))
        
      }
    }
    sub<-sub %>% drop_na(iHosp,all_of(lag_list_col)) %>%
      dplyr::select(-date,-region)
    
    # Fit model 
    
    modelspec <- ugarchspec(mean.model = list(armaOrder=c(max_lag,0) ,external.regressors  = as.matrix(sub[,c(2:ncol(sub))])), 
                            variance.model = list(model = "sGARCH",garchOrder= c(1,1)))
    mod <- ugarchfit(data=sub[,"iHosp"],spec=modelspec)
    
    
    # build new data to predict
    newdata<-data.frame(seq.Date(dates_analysis[iDate]-remove_last_n+1,dates_analysis[iDate]+(max_prediction_horizon-1)*7,by="week"),NA,NA,NA)
    names(newdata)<-c("date","iHosp","weekid",'monthid')
    
    sub1<-dat_train_reg[,c("date","iHosp","weekid",'monthid',list_col)]
    
    if (n_predictors>0){
      
      for (i in 1:length(list_lag)){
        
        sub2<-sub1
        sub2$date<-sub2$date+list_lag[i]*7
        
        newdata<-left_join(newdata,sub2[,c("date",list_col[i])],by=c("date"="date"))
        
        # fill NA by last available value
        newdata[is.na( newdata[,ncol(newdata)]),ncol(newdata)]<- last(newdata[!is.na( newdata[,ncol(newdata)]),ncol(newdata)])
        
      }
    }
    newdata$weekid <- (dat_train$weekid[nrow(dat_train)]+1:max_prediction_horizon-1)%%52+1
    newdata$monthid <- as.numeric(strftime(newdata$date, format = "%m"))
    
    # Predict growth rate
    pred_m <- ugarchforecast(mod,n.ahead = max_prediction_horizon,external.forecasts =  list(mregfor=newdata[,-c(1,2)]))
    out <- data.frame(point = pred_m@forecast$seriesFor, point_avg = pred_m@forecast$seriesFor)
    colnames(out) = c("point","point_avg")
    pred <- out
    pred = exp(pred)/10 # 得到真实的log预测
    
    res_tmp<- data.frame(var="iHosp",
                         region=myreg,
                         date=dates_analysis[iDate],
                         prediction_horizon=seq(1,max_prediction_horizon)-remove_last_n,
                         pred) 
    
    
    res<-rbind(res,res_tmp)
    
  }
  
}



res1 <- res %>%
  mutate(date=as.Date(date)) %>%
  mutate(date_pred=date+prediction_horizon*7) %>%
  mutate(model=model_name) 

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

col_list = c('date','true','week_ahead', 'point', "point_avg")
res1 <- res1[, col_list]


####save
if(dir.exists("../Results/Point/") == TRUE){
  write.csv(res1, paste0('../Results/Point/forecast_', model_name,'_',mode,'.csv'), row.names = FALSE)
}else{
  dir.create('../Results/Point/', recursive = TRUE)
  write.csv(res1, paste0('../Results/Point/forecast_', model_name,'_',mode,'.csv'), row.names = FALSE)
}

