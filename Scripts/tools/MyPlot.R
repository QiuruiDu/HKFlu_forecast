my.compute = function(proj) {
  res<-proj %>%
    mutate(in_interval95 = true >= lower_5 & true <= upper_5) %>%
    mutate(abs_error = abs(point - true)) #%>% 
    # mutate(bias = abs((point - true)/true))
  
  # WIS 
  res$IS_2<-interval_score(true_values = res$true,
                           lower=res$lower_2,
                           upper=res$upper_2,
                           interval_range=98,
                           weigh = TRUE,
                           separate_results = FALSE)
  
  res$IS_5<-interval_score(true_values = res$true,
                           lower=res$lower_5,
                           upper=res$upper_5,
                           interval_range=95,
                           weigh = TRUE,
                           separate_results = FALSE)
  
  res$IS_10<-interval_score(true_values = res$true,
                            lower=res$lower_10,
                            upper=res$upper_10,
                            interval_range=90,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_20<-interval_score(true_values = res$true,
                            lower=res$lower_20,
                            upper=res$upper_20,
                            interval_range=80,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_30<-interval_score(true_values = res$true,
                            lower=res$lower_30,
                            upper=res$upper_30,
                            interval_range=70,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_40<-interval_score(true_values = res$true,
                            lower=res$lower_40,
                            upper=res$upper_40,
                            interval_range=60,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_50<-interval_score(true_values = res$true,
                            lower=res$lower_50,
                            upper=res$upper_50,
                            interval_range=50,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_60<-interval_score(true_values = res$true,
                            lower=res$lower_60,
                            upper=res$upper_60,
                            interval_range=40,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_70<-interval_score(true_values = res$true,
                            lower=res$lower_70,
                            upper=res$upper_70,
                            interval_range=30,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_80<-interval_score(true_values = res$true,
                            lower=res$lower_80,
                            upper=res$upper_80,
                            interval_range=20,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_90<-interval_score(true_values = res$true,
                            lower=res$lower_90,
                            upper=res$upper_90,
                            interval_range=10,
                            weigh = TRUE,
                            separate_results = FALSE)
  
  res$IS_100<-abs(res$point-res$true)
  
  res <- res %>%
    mutate(wis = (IS_2+IS_5+IS_10+IS_20+IS_30+IS_40+IS_50+
                    IS_60+IS_70+IS_80+IS_90+0.5*IS_100)/11.5 )  
  
  return(res)
}

my.point.compute = function(proj){
  res<-proj %>%
    mutate(abs_error = abs(point - true)) #%>% 
  # mutate(bias = abs((point - true)/true))
}

my.plot = function(res, model_list){
  res$model<-factor(res$model,levels=model_list, ordered=T)
  res$inclusion<-1
  res$inclusion[which(res$date>=as.Date("2009-09-01") & res$date<=as.Date("2009-12-31") )]<-0
  # Plot settings  ------------------------------------------------
  pal<-c("black",brewer.pal(10,"Paired"),"turquoise",brewer.pal(6, "Dark2"))
  val_size<-c(rep(0.5,length(model_list)))
  val_linetype<-c("dashed", rep("solid",length(model_list)-1))
  lab_mod<-levels(droplevels(res$model))
  
  # wis trend
  # dev.new()
  res %>%
    mutate(weekid = as.numeric(strftime(date,format= "%V"))) %>%
    # mutate(weekid_pred = as.numeric(strftime(date+7*(week_ahead-1),format= "%V"))) %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,weekid) %>%
    dplyr::summarize(average_bias = mean(wis,na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = weekid, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("WIS") +
    scale_x_continuous("Predict WeekID", breaks = unique(res$weekid)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("WIS trend, HK")
  # dev.new()
  res %>%
    mutate(weekid = as.numeric(strftime(date,format= "%V"))) %>%
    # mutate(weekid_pred = as.numeric(strftime(date+7*(week_ahead-1),format= "%V"))) %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,weekid) %>%
    dplyr::summarize(average_bias = mean(abs(bias),na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = weekid, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("MAPE") +
    scale_x_continuous("Predict WeekID", breaks = unique(res$weekid)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    coord_cartesian(ylim=c(0,1.5)) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("MAPE trend, HK")
  # dev.new()
  res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = sqrt(mean(abs_error^2,na.rm=T))) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("RMSE") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    theme_bw()+
    coord_cartesian(ylim=c(1,5.5)) +
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("RMSE, HK")
  # dev.new()
  res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = mean(wis,na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("WIS") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    #coord_cartesian(ylim=c(0,0.45)) +
    theme_bw()+
    coord_cartesian(ylim=c(0.7,3)) +
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "B") + 
    ggtitle("WIS, HK")
  # dev.new()
  res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = mean(abs(bias),na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("MAPE") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    coord_cartesian(ylim=c(0.2,0.85)) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "B") + 
    ggtitle("MAPE, HK")
}

my.saved.plot = function(res, model_name){
  # ** Figure RMSE、 WIS and MAPE-----------------------------------------
  d0<-res %>%
    mutate(weekid = as.numeric(strftime(date,format= "%V"))) %>%
    group_by(model,weekid) %>%
    dplyr::summarize(average_bias = mean(wis,na.rm=T))%>%
    ungroup() 
  
  #### 1. trend plot -----
  wis.trend <- res %>%
    mutate(weekid = as.numeric(strftime(date,format= "%V"))) %>%
    # mutate(weekid_pred = as.numeric(strftime(date+7*(week_ahead-1),format= "%V"))) %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,weekid) %>%
    dplyr::summarize(average_bias = mean(wis,na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = weekid, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("WIS") +
    scale_x_continuous("Predict WeekID", breaks = unique(res$weekid)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("WIS, HK")
  
  ## - MAPE
  mape.trend <-res %>%
    mutate(weekid = as.numeric(strftime(date,format= "%V"))) %>%
    # mutate(weekid_pred = as.numeric(strftime(date+7*(week_ahead-1),format= "%V"))) %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,weekid) %>%
    dplyr::summarize(average_bias = mean(abs(bias),na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = weekid, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("MAPE") +
    scale_x_continuous("Predict WeekID", breaks = unique(res$weekid)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    coord_cartesian(ylim=c(0,1.5)) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("MAPE, HK")
  
  #### 2. model compare plot -----
  ## wis
  # d1<-res %>%
  #   filter(inclusion==1) %>% 
  #   filter(var=="iHosp") %>% 
  #   group_by(model,var) %>%
  #   dplyr::summarize(average_bias = mean(wis,na.rm=T))%>%
  #   ungroup() %>%
  #   ggplot() +
  #   geom_point(aes(x = model, y = average_bias/average_bias[1], color = model )) +
  #   theme_bw()+
  #   theme( legend.text=element_text(size=10),
  #          legend.key.size = unit(2, "mm"))+
  #   guides(color=guide_legend(ncol=1)) 
  
  ## rmse
  # d1.1<-res %>%
  #   filter(inclusion==1) %>% 
  #   filter(var=="iHosp") %>% 
  #   group_by(model,var) %>%
  #   dplyr::summarize(average_bias = sqrt(mean(abs_error^2,na.rm=T)))%>%
  #   ungroup() %>%
  #   ggplot() +
  #   geom_point(aes(x = model, y = average_bias/average_bias[1], color = model )) +
  #   theme_bw()+
  #   theme( legend.text=element_text(size=10),
  #          legend.key.size = unit(2, "mm"))+
  #   guides(color=guide_legend(ncol=1)) 
  
  
  # mae
  mae.compare <-res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = mean(abs(abs_error),na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("MAE") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    theme_bw()+
    coord_cartesian(ylim=c(1,3.5)) +
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("MAE, HK")
  
  
  # RMSE
  rmse.compare <-res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = sqrt(mean(abs_error^2,na.rm=T))) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("RMSE") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    theme_bw()+
    coord_cartesian(ylim=c(1,5.5)) +
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "A") + 
    ggtitle("RMSE, HK")
  
  
  # WIS 
  wis.compare <-res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = mean(wis,na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("WIS") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    #coord_cartesian(ylim=c(0,0.45)) +
    theme_bw()+
    coord_cartesian(ylim=c(0.7,3)) +
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "B") + 
    ggtitle("WIS, HK")
  
  # MAPE 
  mape.compare<-res %>%
    filter(inclusion==1) %>% 
    filter(var=="iHosp") %>% 
    group_by(model,var,week_ahead) %>%
    dplyr::summarize(average_bias = mean(abs(bias),na.rm=T)) %>%
    ungroup() %>%
    ggplot() +
    geom_line(aes(x = week_ahead, y = average_bias, color = model , 
                  size=model,linetype=model)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_y_continuous("MAPE") +
    scale_x_continuous("Days ahead", breaks = unique(res$week_ahead)) + 
    scale_color_manual("Model",labels=lab_mod,values=pal) +
    scale_size_manual("Model",labels=lab_mod,values=val_size) +
    scale_linetype_manual("Model",labels=lab_mod,values=val_linetype) +
    coord_cartesian(ylim=c(0.2,0.85)) +
    theme_bw()+
    theme( legend.text=element_text(size=10),
           legend.key.size = unit(2, "mm"))+
    guides(color=guide_legend(ncol=1)) +
    labs(tag = "B") + 
    ggtitle("MAPE, HK")
  
  # g1.1<-res %>%
  #   filter(inclusion==1) %>% 
  #   filter(var=="iHosp") %>% 
  #   group_by(model,var) %>%
  #   dplyr::summarize(average_bias = mean(abs(bias),na.rm=T))%>%
  #   ungroup() %>%
  #   ggplot() +
  #   geom_point(aes(x = model, y = average_bias/average_bias[1], color = model )) +
  #   theme_bw()+
  #   theme( legend.text=element_text(size=10),
  #          legend.key.size = unit(2, "mm"))+
  #   guides(color=guide_legend(ncol=1)) 
  
  
  #### 4. save plot -----
  png(paste0(origin_path,'/figure/',model_name,"_evaluation_index.png"),width=10,height=11,units="in",res=200)
  # p2 = rmse, d1.1 = rmse point, p3 = wis, d1 = wis point, g2 = mape, g1.1 = mape point
  print(plot_grid(mae.compare,rmse.compare,mape.compare,mape.trend,wis.compare,wis.trend,ncol=2,align="h",axis="lr"))
  
  dev.off()
  
}