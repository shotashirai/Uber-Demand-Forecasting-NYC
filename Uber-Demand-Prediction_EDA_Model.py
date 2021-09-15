# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
from IPython.display import Image
Image(filename='images/uber-new-york2.jpeg') 

# + [markdown] tags=[]
# # Uber Ride Prediction in New York
# -

# ## Table of Contents

# ## Project Summary
# ---

# **Siuation**  
# The demand for rideshearing is drastically growing, especially in large cities. Uber is the first ride-hailing company and has operation in over 900 metropolitan areas worldwide. This project aims to investigate whether weather can make an impact on the demand for Uber rides in New York.  
#   
# **Action and  Goal**  
# Using the Uber trip data (the mid-6 months of 2014 and the first 6 months of 2015) and the weather data, data exploration and model deployment are implemented by using Python and jupyter notebook. The goal is to build a predictive model and test the hypothesis that ***"Weather makes an impact on the demand for the uber rides in New York"***.

# ARIMA: Auto Regressive Integrated Moving Average

# ## Assumptions
# ---
# - Since all boroughs are neighboring the same weather information in NY was used.
# - Since the available Uber trip data is for only 6 months of 2015, the result might have bias such as seasonal trend and anomary events.

# ## Comments(to be rivised)
# In a more optimized version we may use more localized weather stations but the area is relatively narrow for significant weather differences.
# Additionally, using information from different stations may enter noise by various factors (like missing values or small calibration differences).

# ---
# # Loading 
# ---

# ## Import Libraries 
# ---

# + tags=[]
import sys, os
sys.path.append(os.pardir)
import math
import warnings
warnings.filterwarnings('ignore')

from datetime import date
from datetime import datetime

# EDA
import numpy as np
import pandas as pd
from functools import reduce


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from boruta import  BorutaPy 


# -

# ## Functions 
# ---

# + tags=[]
def memory_usage(var, lower_limit=0):
    # input: var = dir()
    print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
    print(" ------------------------------------ ")
    for var_name in var:
        if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > lower_limit:
            print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
    return


# + tags=[]
def data_profile(df):
    
    # Datatype
    df_dtype = pd.DataFrame(df.dtypes)
    
    # Null count
    df_null = pd.DataFrame(df.isnull().sum())
    # Null ratio (%)
    null_ratio = df.isnull().sum() / df.shape[0] * 100 #Percentage
    null_ratio = null_ratio.apply(lambda x: math.floor(x * 10**2) / 10**2) #rounding
    df_null_ratio = pd.DataFrame(null_ratio)
    
    # Non-null count
    df_notnull = pd.DataFrame(df.notnull().sum())
    
    # Unique value/Unique ratio(%)
    df_unique = {}
    df_unique_ratio = {}
    
    for column in df.columns:
        df_unique[column] = len(df[column].value_counts())
        df_unique_ratio[column] = len(df[column].value_counts()) / df.shape[0] *100 #Percentage
        
    df_unique = pd.DataFrame.from_dict(df_unique, orient='index')
    df_unique_ratio = pd.DataFrame.from_dict(df_unique_ratio, orient='index')
    df_unique_ratio = df_unique_ratio[0].apply(lambda x: math.floor(x * 10**2) / 10**2) #rounding
    
    # Create a new dataframe showing the data profile 
    df_profile = pd.concat([df_dtype, df_null, df_null_ratio, df_notnull, df_unique, df_unique_ratio], axis=1).reset_index()
    df_profile.columns = ['Column', 'Data type', 'Null count', 'Null ratio (%)', 'Non-null count', 'Distinct', 'Distinct (%)']
    
    num_dup = df.duplicated().sum()
    if num_dup > 0:
        print(str(num_dup) + 'rows are duplicated')
    else: print('No duplicated row')
    return df_profile


# + [markdown] tags=[]
# ## Import data
# ---
# All data used in this project is stored in "data" directory.  
#   
# **Uber trip data from 2014** (April - September)  
# : separated by month, with detailed location information
# - uber-raw-data-apr14.csv
# - uber-raw-data-aug14.csv
# - uber-raw-data-jul14.csv
# - uber-raw-data-jun14.csv
# - uber-raw-data-may14.csv
# - uber-raw-data-sep14.csv
#   
# **Uber trip data from 2015** (January - June)
# : with less fine-grained location information
# - uber-raw-data-janjune-15.csv
#   
# **Weather data**  
# - humidity.csv
# - pressure.csv
# - temperature.csv
# - weather_description.csv
# - wind_direction.csv
# - wind_speed.csv
#   
# **(Data Source)**  
# The data shown above can be obtained from:  
# - Uber trip data: https://github.com/fivethirtyeight/uber-tlc-foil-response  
# - Histroical weather in NY: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=humidity.csv
#
#

# + tags=[]
# Uber data
uber_data_dir = 'data/uber-tlc-foil-response/uber-trip-data/uber-raw-data-'
# uber_raw_apr14 = pd.read_csv(uber_data_dir + 'apr14.csv')
# uber_raw_may14 = pd.read_csv(uber_data_dir + 'may14.csv')
# uber_raw_jun14 = pd.read_csv(uber_data_dir + 'jun14.csv')
# uber_raw_jul14 = pd.read_csv(uber_data_dir + 'jul14.csv')
# uber_raw_aug14 = pd.read_csv(uber_data_dir + 'aug14.csv')
# uber_raw_sep14 = pd.read_csv(uber_data_dir + 'sep14.csv')

# Concatenate raw data between April and September, 2014
# uber_raw_aprsep14 = pd.concat([uber_raw_apr14
#                              , uber_raw_may14
#                              , uber_raw_jun14
#                              , uber_raw_jul14
#                              , uber_raw_aug14
#                              , uber_raw_sep14]).reset_index()

# delete non-used variables
# del uber_raw_apr14, uber_raw_may14, uber_raw_jun14, uber_raw_jul14, uber_raw_aug14, uber_raw_sep14

# Raw data between January and June, 2015
uber_raw_janjun15 = pd.read_csv(uber_data_dir + 'janjune-15.csv').reset_index()
taxi_zone = pd.read_csv('data/uber-tlc-foil-response/uber-trip-data/taxi-zone-lookup.csv')

Base_code_ref = pd.DataFrame({
    'Base Code':['B02512', 'B02598', 'B02617', 'B02682', 'B02764', 'B02765', 'B02835', 'B02836'],
    'Base Name':['Unter', 'Hinter', 'Weiter', 'Schmecken', 'Danach-NY', 'Grun', 'Dreist', 'Drinnen']  
})

# Weather data
city_attrib = pd.read_csv('data/weather/city_attributes.csv')
humidity = pd.read_csv('data/weather/humidity.csv')
pressure = pd.read_csv('data/weather/pressure.csv')
temperature = pd.read_csv('data/weather/temperature.csv')
weather_description = pd.read_csv('data/weather/weather_description.csv')
wind_direction = pd.read_csv('data/weather/wind_direction.csv')
wind_speed = pd.read_csv('data/weather/wind_speed.csv')

# + tags=[] jupyter={"outputs_hidden": true}
# Check memory usage
# var = dir()
# memory_usage(var, lower_limit=1000)
# -

# # Exploring the Data 

# ## Data Overview
# ---

# ### Input variables 
# ---
# **Uber trip data from 2014**   
# - Date/Time : The date and time of the Uber pickup
# - Lat : The latitude of the Uber pickup
# - Lon : The longitude of the Uber pickup
# - Base : The TLC base company code affiliated with the Uber pickup
#   
# **Uber trip data from 2015**  
# - Dispatching_base_num : The TLC base company code of the base that dispatched the Uber
# - Pickup_date : The date and time of the Uber pickup
# - Affiliated_base_num : The TLC base company code affiliated with the Uber pickup
# - locationID : The pickup location ID affiliated with the Uber pickup
#
# **Weather data**  
# - datetime : The date and time of the weather record  
# ---

# ### Uber rides data 

uber_raw_aprsep14.shape

uber_raw_janjun15.shape

# + tags=[]
# Uber trip data from 2014
uber_raw_aprsep14.head()
# -

# Uber trip data from 2015
uber_raw_janjun15.head()

# Location reference
taxi_zone.head()

# Borough
uber_raw_janjun15['borough'] = uber_raw_janjun15['locationID']\
                            .map(taxi_zone.set_index('LocationID')['Borough'])
uber_raw_janjun15['borough'].unique()

uber_raw_janjun15.head()

# #### Format date and time

# convert string to timestamp object
# uber_raw_aprsep14['datetime'] = pd.to_datetime(uber_raw_aprsep14['Date/Time'])
uber_raw_janjun15['datetime'] = pd.to_datetime(uber_raw_janjun15['Pickup_date'])

# extract date and time (hour)
uber_raw_janjun15['datehour'] = uber_raw_janjun15['datetime'].dt.strftime('%Y-%m-%d %H')

# #### Time period of the uber trip data

# +
# print('Uber trip data from 2014')
# print('Min date: %s' % uber_raw_aprsep14['datetime'].min())
# print('Max date: %s' % uber_raw_aprsep14['datetime'].max())
# -

print('Uber trip data from 2015')
print('Min date: %s' % uber_raw_janjun15['datetime'].min())
print('Max date: %s' % uber_raw_janjun15['datetime'].max())

data_profile(uber_raw_janjun15)

# <div class='alert-success'>
#
# - The uber trip data for 2015 has no duplicated data.  
# - 1.1% of Affliated_base_num is NaN values.
#
# </div>

# + [markdown] tags=[]
# ---
# ### Weathre data 
# -

# #### Extract and concatenate the weather data for New York 

# +
# Extract weather in NY and merge weather data

# merged weather data
dataframes = [
    humidity[['datetime','New York']]
    , pressure[['datetime','New York']]
    , temperature[['datetime','New York']]
    , weather_description[['datetime','New York']]
    , wind_direction[['datetime','New York']]
    , wind_speed[['datetime','New York']]
]

# merged data frame for weather data in NY
weather_NY = reduce(lambda left, right: pd.merge(left, right , on='datetime',
                   how='outer'), dataframes)

weather_NY.columns = ['datetime', 'humidity', 'pressure', 'temperature', 'weather_description', 'wind_direction', 'wind_speed']

# delete the unnecessary variables
del humidity, pressure, temperature, weather_description, wind_direction, wind_speed
# -

# Weather data (01/01/2015 - 31/06/2015)
weather_NY_15 = weather_NY[weather_NY['datetime'] < '2015-07'][weather_NY['datetime'] > '2015']

# + tags=[]
# Check memory usage
# var = dir()
# memory_usage(var, lower_limit=1000)
# del var
# -



# #### Weather descriptions

weather_NY['weather_description'].unique()

# #### Temperature 

#Convert temperature (K) into temperature (C: degree Celsius)
weather_NY['temperature'] = weather_NY['temperature'] - 273.15

# #### Format date and time  

weather_NY_15['datetime'] = pd.to_datetime(weather_NY_15['datetime'])

# extract date and time (hour)
weather_NY_15['datehour'] = weather_NY_15['datetime'].dt.strftime('%Y-%m-%d %H')

print('Weather data (2015)')
print(weather_NY_15['datetime'].min())
print(weather_NY_15['datetime'].max())

# #### Basic summary of the weather data 

# + tags=[]
data_profile(weather_NY_15)
# -

# <div class='alert-success'>
#
# The weather data for the first 6 months in 2015 does not have missing data and duplicated data 
#
# </div>

weather_state = weather_NY_15['weather_description'].unique()
print(weather_state)
print('\n The number of the weather states is', len(weather_state))

# ## Data preparation for analysis

# ### Hourly demand for rides

# hourly rides from all borough
hourly_Uber_rides = uber_raw_janjun15[['datehour','datetime']]\
                    .groupby('datehour')\
                    .count()\
                    .reset_index()
hourly_Uber_rides.columns  = ['datehour', 'count']

hourly_Uber_rides.head()

# ### Hourly demand for rides by borough

# hourly rides from all borough
hourly_rides_borough = uber_raw_janjun15[['datehour','datetime','borough']]\
                        .groupby(['datehour','borough'])\
                        .count()\
                        .reset_index()
hourly_rides_borough.columns  = ['datehour','borough', 'count']

hourly_rides_borough.head()

# ### Merge Uber rides data and weather data

df_hourly_rides = pd.merge(hourly_Uber_rides, weather_NY_15, on='datehour')
df_hourly_rides_borough = pd.merge(hourly_rides_borough, weather_NY_15, on='datehour')

# add day of week(0:Sun, 1:Mon,...,6)
df_hourly_rides['day_of_week'] = df_hourly_rides['datetime'].dt.strftime('%w')
df_hourly_rides_borough['day_of_week'] = df_hourly_rides_borough['datetime'].dt.strftime('%w')

df_hourly_rides.head()

# +
fig, ax = plt.subplots(figsize=(7,5))

sns.heatmap(df_hourly_rides.corr(), ax=ax,vmax=1, vmin=-1, cmap='coolwarm')

# +
fig, ax = plt.subplots(figsize =(20,5))
corr_matt_sub = df.corr().drop('count', axis=0)

color_cor = []
for i in corr_matt_sub['count']:
    if i >= 0:
        color_cor.append('red')
    else:
        color_cor.append('blue')

plt.bar(x=corr_matt_sub['count'][:].index, height=corr_matt_sub['count'][:].values
        , color = color_cor, align='center', width=0.9)
# sns.barplot(x=corr_matt_sub['count'][:].index, y=corr_matt_sub['count'][:].values
#             , hue = color_cor)
# plt.xticks(rotation=90);
plt.xlim([-1, len(corr_matt_sub['count'][:].index)])
ax.set_ylabel('Correltaion');
ax.set_title("Correlation strength for 'subscribed'");
plt.show()

# +
df_corr = df_hourly_rides_borough.copy()
df_corr = df_corr.drop(['datehour','datetime'], axis=1)

# sc = StandardScaler()
# df_corr['humidity'] = sc.fit_transform(df_corr[['humidity']])
# df_corr['temperature'] = sc.fit_transform(df_corr[['temperature']])
# df_corr['pressure'] = sc.fit_transform(df_corr[['pressure']])
# df_corr['wind_direction'] = sc.fit_transform(df_corr[['wind_direction']])
# df_corr['wind_speed'] = sc.fit_transform(df_corr[['wind_speed']])

df_corr = pd.get_dummies(df_corr)

df_corr.head()

# + tags=[]
fig, ax = plt.subplots(figsize=(7,5))

sns.heatmap(df_corr.corr(),vmax=1, vmin=-1, cmap='bwr'
            , square=True, ax=ax, linewidths=0.1, center=0)

# +
fig, ax = plt.subplots(figsize =(20,5))
corr_matt_sub = df_corr.corr().drop('count', axis=0)

color_cor = []
for i in corr_matt_sub['count']:
    if i >= 0:
        color_cor.append('red')
    else:
        color_cor.append('blue')

plt.bar(x=corr_matt_sub['count'][:].index, height=corr_matt_sub['count'][:].values
        , color = color_cor, align='center', width=0.9)
# sns.barplot(x=corr_matt_sub['count'][:].index, y=corr_matt_sub['count'][:].values
#             , hue = color_cor)
plt.xticks(rotation=90);
plt.xlim([-1, len(corr_matt_sub['count'][:].index)])
plt.ylim([-1, 1])
ax.set_ylabel('Correltaion');
ax.set_title("Correlation strength for 'subscribed'");
plt.show()
# -

# Correlation 
corr_matt_sub['count'][:]

# + [markdown] tags=[]
# ## Visualization 
# -

# ### The number of rides by borough 

# + tags=[]
total_rides_by_borough = df_hourly_rides_borough.groupby('borough')['count'].agg(sum).reset_index()

total_rides_by_borough.columns = ['borough','count']
fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(x='borough', y = 'count', data = total_rides_by_borough)
# -

# <div class='alert-info'>
#
# Manhattan has the most populous county followed by Brooklyn and Queens.  
#
# </div>

# ### Average number of rides by event 

# +
avg_rides_weather = df_hourly_rides.groupby('weather_description')['count'].mean().reset_index()
avg_rides_weather.columns = ['weather_description', 'avg_rides']

fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(x='weather_description', y = 'avg_rides', data = avg_rides_weather)
plt.xticks(rotation=90);
# -

# ### Average Number of Rides by Event and Boroguh

# +
avg_rides_weather_borough = df_hourly_rides_borough.groupby(['weather_description','borough'])['count'].mean().reset_index()
avg_rides_weather_borough.columns = ['weather_description', 'borough', 'avg_rides']

fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(x='weather_description', y = 'avg_rides', hue = 'borough', data = avg_rides_weather_borough)
plt.xticks(rotation=90);
# -

# ### Number of rides per hour by month

# +
df_month = df.copy()
df_month['month'] = df_month['datetime'].dt.month
df_month['dayhour'] = df_month['datetime'].dt.strftime('%d %H')
df_month = df_month.pivot('dayhour','month','count')

fig, ax = plt.subplots(figsize=(20,10))

sns.lineplot(data=df_month)
ax.set_ylabel('Counts of Uber rides');
ax.set_xlabel('Day-Hour');

# + [markdown] tags=[]
# ### Histogram of the Uber rides by the weather decription 
# -

df_hourly_rides.hist(column="count", by='weather_description'
        , sharex=True, sharey=False
        , figsize=(15,10), layout=(4,5)
        , color='b'
       )
plt.show()

df.groupby('weather_description').count()['count']

# <div class = 'alert-danger'>
#
# - Some weather conditions (drizzle, fog, haze, heavy intensity rain, light intensity drizzle, proximity thunderstorm, snow, thunderstorm, thunderstorm with light rain, very heavy rain) are not frequent events (< 50 events)  
# - The histogram is left-skewed
#
# </div>

# ### Avrerage demand through the week

# +
df_hour = df_hourly_rides[['datetime','datehour','day_of_week','count']]
df_hour['hour']=df['datetime'].dt.strftime('%H')

hourly_rides_day_mean = df_hour.groupby(['hour','day_of_week']).mean().unstack()
hourly_rides_day_mean.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

fig, axes = plt.subplots(1,2, figsize=(12,5))

sns.lineplot(data=hourly_rides_day_mean, ax=axes[0])
axes[0].set_ylabel('Average rides')

sns.heatmap(hourly_rides_day_mean, ax=axes[1]
            , cmap='coolwarm', cbar_kws={'label':' Pickups per hour'})
axes[1].invert_yaxis()

# -

# ### Distribution of demands per day 

df_rides_day = df_hourly_rides[['datetime','day_of_week','count']]
df_rides_day['date'] = df_rides_day['datetime'].dt.strftime('%Y-%m-%d')
daily_rides = df_rides_day.groupby(['date','day_of_week']).agg(sum).reset_index()
daily_rides.head()

fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='day_of_week', y='count', data=daily_rides)

# ### Average Demand through the week by Borough

# +
df_hour_borough = df_hourly_rides_borough[['datetime','datehour','day_of_week','count','borough']]
df_hour_borough['hour'] = df_hour_borough['datetime'].dt.strftime('%H')

list_borough = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR']
fig, axes = plt.subplots(6, 1, figsize=(10,40))

for borough, ax in zip(list_borough, axes):
    rides_borough = (
        df_hour_borough.loc[df_hour_borough['borough']==borough]
        .groupby(['hour','day_of_week']).mean().unstack()
    )
    rides_borough.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
    rides_borough=rides_borough.fillna(0)
    sns.heatmap(rides_borough 
                ,ax=ax,cmap='coolwarm',cbar_kws={'label':' Pickups per hour'})
    ax.invert_yaxis()
    ax.set_title(borough);
    ax.set_xlabel('Day');
# -

# ## Key findings from data explorartion
# ---
# <div class='alert-info'>
#
# - It seems that **time variables (time, day of week)** have much stronger effect on demand for the rides than **weather variables**.
#
# </div>

# + [markdown] tags=[]
# ## Feature Engineering

# + [markdown] tags=[]
# ### Create Lag Features 
# ---
# In time series data, at any point in time, the model needs information about the past. Here, to pass the past (most recent) information, lag feature is created.
# -

df_hourly_rides.head()

df_hour_borough

# +
df_model_rides = df_hourly_rides.copy()
# df_model_rides['day_of_week'] = df_model_rides['datetime'].dt.strftime('%a') 
df_model_rides = df_model_rides.drop(['datehour','day_of_week'], axis=1)
df_model_rides['time_block_num'] = df_model_rides.index+1

# Standardize the variables
sc = StandardScaler()
df_model_rides['humidity'] = sc.fit_transform(df_model_rides[['humidity']])
df_model_rides['temperature'] = sc.fit_transform(df_model_rides[['temperature']])
df_model_rides['pressure'] = sc.fit_transform(df_model_rides[['pressure']])
df_model_rides['wind_direction'] = sc.fit_transform(df_model_rides[['wind_direction']])
df_model_rides['wind_speed'] = sc.fit_transform(df_model_rides[['wind_speed']])
df_model_rides['day_of_week'] = df_model_rides['datetime'].dt.strftime('%a')
# Get one-hot label data for weather description 
df_model_rides = pd.get_dummies(df_model_rides)



borough_list_model = ['Manhattan', 'Brooklyn', 'Queens']

df_borough_model = df_hour_borough[['datetime', 'count', 'borough']]
for borough in borough_list_model:
    df_borough = df_borough_model.loc[df_borough_model['borough']==borough]
    df_borough=df_borough.drop(['borough'], axis=1)
    df_borough.columns = ['datetime']+['rides_'+borough]
    df_model_rides = pd.merge(df_model_rides, df_borough, on='datetime', how='left')

df_model_rides.columns

# +
lag_variables = ['count'
                 , 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens'
       ,'humidity', 'pressure', 'temperature',
       'wind_direction', 'wind_speed', 'weather_description_broken clouds',
       'weather_description_drizzle', 'weather_description_few clouds',
       'weather_description_fog', 'weather_description_haze',
       'weather_description_heavy intensity rain',
       'weather_description_light intensity drizzle',
       'weather_description_light rain', 'weather_description_light snow',
       'weather_description_mist', 'weather_description_moderate rain',
       'weather_description_overcast clouds',
       'weather_description_proximity thunderstorm',
       'weather_description_scattered clouds',
       'weather_description_sky is clear', 'weather_description_snow',
       'weather_description_thunderstorm',
       'weather_description_thunderstorm with light rain',
       'weather_description_very heavy rain',
       'day_of_week_Fri',
       'day_of_week_Mon', 'day_of_week_Sat', 'day_of_week_Sun',
       'day_of_week_Thu', 'day_of_week_Tue', 'day_of_week_Wed']

# lags (1hr, 2hr, 3hr, 6hr, 12hr, 1day, 1week)
lags = [1, 2, 3, 6, 12, 24, 168]

# +
df_model = df_model_rides.copy()

for lag in lags:
    
    df_lag = df_model.copy()
    df_lag.time_block_num+=lag
    # subset only the lag variable required
    df_lag = df_lag[['time_block_num']+lag_variables]
    df_lag.columns = ['time_block_num']+[lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    
    df_model = pd.merge(df_model, df_lag, on=['time_block_num'], how='left')
    
df_model = df_model.drop(range(0,168)).reset_index()
df_model = df_model.drop(['index'], axis=1)
# -

df_model

# # Modelling

# +
time_block = df_model[['datetime', 'time_block_num']]

df_model = df_model.drop('datetime', axis=1)

X_train = df_model[df_model['time_block_num'] < 3663]
X_test = df_model[df_model['time_block_num'] >= 3663]

Y_train = X_train['count']
Y_test = X_test['count']

del X_train['count'],X_train['rides_Manhattan'], X_train['rides_Brooklyn'], X_train['rides_Queens']
del X_test['count'], X_test['rides_Manhattan'], X_test['rides_Brooklyn'], X_test['rides_Queens']
# -



# + tags=[]
model = XGBRegressor(
    max_depth=20,
    n_estimators=1000,
#     min_child_weight=300, 
#     colsample_bytree=0.8, 
#     subsample=0.8, 
#     eta=0.3,    
    seed=42)
# -

model.fit(
    X_train, 
    Y_train, 
    eval_set=[(X_train, Y_train), (X_test, Y_test)], 
    verbose=True
#     early_stopping_rounds = 10
)

feature_importances = pd.DataFrame({'col': X_train.columns,'imp':model.feature_importances_})
feature_importances = feature_importances.sort_values(by='imp',ascending=False)
px.bar(feature_importances,x='col',y='imp')

Y_predict = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
rmse

fig = go.Figure()
fig.add_trace(go.Scatter(y=yyy['count'],
                    mode='lines',
                    name='Actual'))
fig.add_trace(go.Scatter(y=Y_predict,
                    mode='lines',
                    name='Predict'))

# ## Model Optimization

# + tags=[]
feature_selector =BorutaPy(model, 
                         n_estimators='auto', 
                         verbose=2, # 0: no output,1: displays iteration number,2: which features have been selected already
                         alpha=0.1,
                         max_iter=100,
                         random_state=42
                        )
feature_selector.fit(X_train.values, Y_train.values)
# -

X_train_selected.columns

X_train_selected = X_train.iloc[:,feature_selector.support_]
X_train_selected
X_test_selected = X_test.iloc[:,feature_selector.support_]

# + tags=[]
model.fit(
    X_train_selected, 
    Y_train, 
#     eval_set=[(X_train_selected, Y_train), (X_test_selected, Y_test)], 
    verbose=True
#     early_stopping_rounds = 10
)
# -

feature_importances = pd.DataFrame({'col': X_train_selected.columns,'imp':model.feature_importances_})
feature_importances = feature_importances.sort_values(by='imp',ascending=False)
px.bar(feature_importances,x='col',y='imp')

Y_predict = model.predict(X_test_selected)

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
rmse

yyy = Y_test.reset_index()
plt.plot(yyy['count'], label='original')
plt.plot(Y_predict, label='predict')
plt.legend()


fig = go.Figure()
fig.add_trace(go.Scatter(y=yyy['count'],
                    mode='lines',
                    name='Actual'))
fig.add_trace(go.Scatter(y=Y_predict,
                    mode='lines',
                    name='Predict'))












