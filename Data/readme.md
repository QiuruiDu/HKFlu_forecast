# Data Description
Different models are suitable for different data input formats. Here we provide an explanation of the data in this folder.
### 1. The data before COVID period
- 1) rolling_data_before_covid.parquet: used by RF and XGB models
- 2) data_no_absent.csv: used by deep learning models
- 3) data_lograte.rds: used by ARIMA and GARCH models

### 2. The data after COVID period
- 1) rolling_data_after_covid.parquet: used by RF and XGB models
- 2) ILI_data_2023.csv: used by deep learning models
- 3) ILI_data_2023.rds: used by ARIMA and GARCH models

# Columns
| column name | meaning |
| :----:| :---- | 
| weekid | The number of week in a year | 
| monthid | The number of month in a year |
| temp.max | The max temperature during this week |
| temp.mean | The average temperature during this week |
| temp.min | The min temperature during this week |
| relative.humidity| The average relative humidity (%) during this week|
| total.rainfall | The average of daily total rainfall during this week |
| solar.radiation | The average Global Solar Radiation (MJ/m²) during this week|
| wind.speed | The average Wind Speed (km/h) during this week |
| absolute.humidity | The average absolute humidity during this week  |
| pressure | The average Pressure (hPa) during this week |
| temp.range | The average of daily temperature range during this week |
| rate | The ILI+ indicator |