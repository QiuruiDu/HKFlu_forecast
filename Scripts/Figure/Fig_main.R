if(dir.exists("../Figures/") == FALSE){
  dir.create('../Figures/', recursive = TRUE)
}
origin_path = getwd()

source(paste0(origin_path,'/Figure/',"Fig1.R"))
source(paste0(origin_path,'/Figure/',"Fig2.R"))
source(paste0(origin_path,'/Figure/',"Fig3.R"))
source(paste0(origin_path,'/Figure/',"Fig4.R"))
source(paste0(origin_path,'/Figure/',"Fig5.R"))
source(paste0(origin_path,'/Figure/',"Fig6.R"))