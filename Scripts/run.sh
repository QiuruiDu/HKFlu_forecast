# RUN CODE for 'An adaptive weight ensemble approach to forecast influenza activity in the context of irregular seasonality.'
# Author: QR Du

####### <1>, get the final forecasting result for train and test period
# 1. Individual Point Forecasting
Rscript point/GARCH_point.R
Rscript point/ARIMA_point.R

source activate

conda activate base
python point/RF.py
python point/XGB.py
conda activate torch_py39
python point/GRU.py
python point/LSTM.py
python point/TSTPlus.py
python point/InTimePlus.py

# 2.Ensemble forecasting
conda activate base
python point/Ensemble_point.py

# 3. get interval result, this step will take a lot of time
conda activate base
python Std_and_Interval/std_generate.py
Rscript Std_and_Interval/interval_result_generate.R

####### <2>, get the forecasting result for post-COVID period
# 1. individual models
Rscript post_COVID/GARCH_point.R
Rscript post_COVID/ARIMA_point.R

source activate

conda activate base
python post_COVID/RF.py
python post_COVID/XGB.py
conda activate torch_py39
python post_COVID/GRU.py
python post_COVID/LSTM.py
python post_COVID/TSTPlus.py
python post_COVID/InTimePlus.py

# 2. ensemble models
python post_COVID/Ensemble.py

# 3. get interval result
python post_COVID/std_generate.py
Rscript post_COVID/interval_result_generate.R

