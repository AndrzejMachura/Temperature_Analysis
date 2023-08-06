# import libraries
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score

# import data downloader script
import data_downloader as dd  

# localisation parameters
# Warsaw latitude and longnitutde
lat = 52.23
lon = 21.01

# date range for analysis
sd = "1965-01-01"
today = date.today()
ed =str(today)

# requested data
cd = "temperature_2m_mean,rain_sum,snowfall_sum"

# time zone
tz = "Europe%2FBerlin"

# get climate data
# Weather data by Open-Meteo.com (https://open-meteo.com/)
df = dd.get_data(lat, lon, sd, ed, cd, tz)
df = df.dropna()

# get beginning of next month for forecast 
start_n = df['time'].iloc[-1] + pd.offsets.MonthBegin(1)
start_n = str(start_n.date())

# use datetime.to_period() method to extract month and year
df['Year_Month'] = df['time'].dt.to_period('M')

# calculate month mean tepmerature
monthly_temp = df.groupby('Year_Month')['temperature_2m_mean'].mean().reset_index()

# divide mean month temperature data frame
# to test and train data frames
per = 36
train = monthly_temp['temperature_2m_mean'][:-per]
test = monthly_temp['temperature_2m_mean'][-per:]
test_dates = monthly_temp['Year_Month'][-per:]
"""
# select best hyperparameters for SARIMA model by Grid Search
# using test train data and comparing it to test data
scores = []
for p in range(3):
    for i in range(3):
        for q in range(3):
            for P in range(3):
                for D in range(3):
                    for Q in range(3):
                        try:
                            mod = sm.tsa.statespace.SARIMAX(train, order=(p,i,q), seasonal_order=(P,D,Q,12))
                            res = mod.fit(disp=False)
                            score = [p,i,q,P,D,Q,r2_score(test, res.forecast(steps=per))]
                            print(score)
                            scores.append(score)
                            del mod
                            del res
                        except:
                            print('errored')
res = pd.DataFrame(scores)
res.columns = ['p', 'i', 'q', 'P', 'D', 'Q', 'score']
res = res.sort_values(by = ['score'], ascending=False)

# export sorted list of all hyperparameters with r^2 score
res.to_csv("params.csv") 
res = res.iloc[0]

# hyperparameters definition
p= int(res.iloc[0])
i= int(res.iloc[1])
q= int(res.iloc[2])
P= int(res.iloc[3])
D= int(res.iloc[4])
Q= int(res.iloc[5])
r2 = res.iloc[6]
"""
#temporary parameters for quick test
p= 0
i= 1
q= 0
P= 2
D= 1
Q= 0
r2 = 0.9509028082227219

# set model for selected parameters and test data
t_mod = sm.tsa.statespace.SARIMAX(train, order=(p,i,q), seasonal_order=(P,D,Q,12))
t_fit = t_mod.fit(disp=False)
t_fcst = t_fit.forecast(steps=per)
t_fcst = pd.DataFrame(data= {"Year Month": test_dates, "test temperature": test, "train temperature": t_fcst})

bx = t_fcst.plot(x='Year Month', y='test temperature')
t_fcst.plot(ax=bx, x='Year Month', y='train temperature')
bx = plt.legend(['Test Data', 'Train Forecast'])
bx = plt.ylabel('Mean Temperature [\N{DEGREE SIGN}C]')
bx = plt.xlabel('Date')

print("\nparameters:\n")
print(f'p= {p}, i= {i}, q= {q}, P= {P}, D= {D}, Q= {Q}\n')
print(f'r2_score= {r2} \n')

# future periods
per = 120 

# create forecast with SARIMA model
model = sm.tsa.statespace.SARIMAX(monthly_temp['temperature_2m_mean'], order=(p,i,q),seasonal_order=(P,D,Q,12))
fit = model.fit(disp = False)
fcst = fit.forecast(steps=per)

# define date range for focasted values
f_date = pd.date_range(start=start_n, periods=per, freq='M').to_period("M")

# convert forecast Series object to DataFrame
fcst = pd.DataFrame(data= {"Year_Month": f_date, "temperature_2m_mean": fcst})

# add forecast to original dataframe
#mtwf = monthly_temp.append(fcst, ignore_index=False)
mtwf = pd.concat([fcst,monthly_temp.loc[:]]).reset_index(drop=True)

mtwf['Year_Month'] = mtwf['Year_Month'].astype(str)
mtwf['Year_Month'] = pd.to_datetime(mtwf['Year_Month'])

# use datetime.to_period() method to extract year
mtwf['Year'] = mtwf['Year_Month'].dt.to_period('Y')

# calculate year mean tepmerature
ymt = mtwf.groupby('Year')['temperature_2m_mean'].mean().reset_index()
ymt['Year'] = ymt['Year'].astype(str)
ymt['Year'] = pd.to_datetime(ymt['Year'])
ymt['Year_Month'] = ymt['Year'].dt.to_period('M')
ymt = ymt.drop('Year', axis=1)

#calculate equation for quadratic trendline
z = np.polyfit(ymt.index.tolist(), ymt['temperature_2m_mean'].values.tolist(), 2)
trend = np.poly1d(z)
ymt['Trend'] = trend(ymt.index.tolist())

#generate label with trend line equetion
txt =f'trendline equation:\ny=({z[0]:.3f})x\N{SUPERSCRIPT TWO}+({z[1]:.3f})x+({z[2]:.3f})' 

# export to csv - optional
ymt.to_csv('year_mean_temperature.csv') 

# generate plot of mean temperature with trendline 
plt.rcParams["figure.figsize"] = [15., 5.]
plt.rcParams["figure.autolayout"] = True

ax = monthly_temp.plot(x='Year_Month', y='temperature_2m_mean')
fcst.plot(ax=ax, x='Year_Month', y='temperature_2m_mean')
ymt.plot(ax=ax, x='Year_Month', y='temperature_2m_mean')
ymt.plot(ax=ax, x='Year_Month', y='Trend', linestyle='dotted')
ax = plt.text(0.67, 0.1, txt, transform = ax.transAxes)
ax = plt.legend(['Actual data - Month Mean', 'Forecast - Month Mean', 'Year Mean', 'Trendline'])
ax = plt.ylabel('Mean Temperature [\N{DEGREE SIGN}C]')
ax = plt.xlabel('Date')
ax = plt.savefig('temperature_forecast.jpg')

plt.show()