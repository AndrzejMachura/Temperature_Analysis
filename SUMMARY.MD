# Mean temperature growth analysis in Warsaw, Poland

This analysis is using historical temperature data to forecast future montly mean and yearly mean temperature in Warsaw, Poland using SARIMA model.

## Content
1. [Data Source](#data-source)
2. [Methodology](#methodology)
3. [Analysis](#analysis)
    - [Hyperparameters](#hyperparameters)
    - [Results](#results)
4. [Summary](#summary)
5. [Licence](#license)


## Data Source

Temperature data was utilised from [Open-Meteo.com](https://open-meteo.com/)

## Methodology

Analysis was performed in the following steps:

1. Main script ([temperature analysis](.\temperature_analysis.py)) is using function *get_data* to connect to [Open-Meteo.com](https://open-meteo.com/) API and receive requested temperature for each day in defined period. 
    - Location (Warsaw - Poland) is defined by geographical coordinates. 
    - Weather data is acquired for period from 1965-01-01 until today. 
    - *get_data* function is stored in [data downloader](.\data_downloader.py) script.
2. Before analysis can be performed some data engineering have to by done. 
    - Historical data published by Open-Meteo are delayed few days so empty rows for not existing dated have to be removed. 
    - Next step is to group data by monthly mean temperature.
3. As a forecasting method SARIMA model was used, which is Seasonal Autoregressive Integrated Moving Average method for time series forecasting with univariate data containing trends and seasonality. This step is divided into some sub-steps:
    - Monthly mean temperature table was divided into test and train data. Last 36 months are treated as test data to choose the best SARIMA hyperparameters.
    - As a measurement of correctness of model R-Squared (R<sup>2</sup>) coefficient was chosen.
    - Grid Search was used to get the best hyperparameters for machine learning model. Range <0;2> was used for each hyperperameter.
    - After choosing the best set of hyperparameters the analysis is redone for full set of historical data. Temperature is forecasted for next 10 years.
4. Basing on the historical and forecasted temperature the yearly mean temperature was calculated.
5. Yearly mean temperature was used to generate temperature change trendline, which is second order curve. *polyfit* and *poly1d* methods from *Numpy* library were used.
6. Analysis results are gathered in two charts:
    - First chart is showing convergence between test and train data.
    - Second chart is showing monthly mean temperature for historical data, 10 years forecast, yearly mean temperature and trendline. Trendline equation is noted in the right lower corner of chart.
7. Output:
    - *year_mean_temperature.csv* -  csv file containing yearly mean temperature for period from January 1965 until next ten years from today.
    - *params.csv* - csv file containing SARIMA hyperparameters and R<sup>2</sup> coeficien for each hyperparameters set.
    - *temperature_forecast.jpg* - second chart described in point 6 of [Methodology](#methodology)

 
## Analysis
Analysis was done 6th Aug 2023. Results may differ for time when analysis is performed because API is requested for data from Jan 1965 up today.

### Hyperparameters
Final SARIMA model was run with following parameters:
- p = 0
- i = 1
- q = 0
- P = 2
- D = 1
- Q = 0

It acquired R<sup>2</sup> score equal 0.9509. Test data and train data shows good convergence, what is shown in figure below:

<img src="./Test_train.png" width="400" height="350">

### Results

Analysis results are gathered in chart below. Historical monthly mean temperature is marked with blue line, the 10 years forecast is marked with orange one. Green line represents yearly mean temperature. Trendline was build on the yearly mean temperature with usage of polyline of second. Equation of the trendline is presented as following:<br>
f(x) = 0.001x<sup>2</sup> -0.028x +8.004

![plot](./temperature_forecast.jpg)

## Summary

In this analysis Seasonal Autoregressive Integrated Moving Average method was used to forecast monthly and yearly temperature in Warsaw, Poland. Period taken under consideration is Jan 1965 till current day (6th Aug 2023 - day of analysis). <br> 
SARIMA is univariate model which takes under consideration only one variable (in this case monthly mean temperature). It is not considering factors like city, population or population density growth. Excluding mentioned factors could have affected the projected rate of growth. Despite this the analysis clearly showed that the mean temperature in Warsaw is rising from year to year.

## License

Scripts and results are offered under [Creative Commons — CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). <br>
It would be nice to inform and mention me if you are using any part of my work.