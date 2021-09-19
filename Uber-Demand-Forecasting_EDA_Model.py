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
# # Forecasting Uber Demand in New York

# + [markdown] tags=[]
# ## Table of Contents
# 1 [ProjectSummary](#1)  
#     1.1 [Situation](#1.1)  
#     1.2 [Action and Goal](#1.2)  
# 2 [Loading](#2)  
#     2.1 [Import Libraries](#2.1)  
#     2.2 [Functions](#2.2)  
#     2.3 [Import Data](#2.3)  
# 3 [Exploring Data Analysis (EDA)](#3)  
#     3.1 [Data Overview](#3.1)  
#         3.1.1 [Input Variables](#3.1.1)  
#         3.1.2 [Uber Trip Data](#3.1.2)  
#         3.1.3 [Weather Data](#3.1.3)  
#     3.2 [Data Cleaning](#3.2)  
#     3.3 [Visualisation](#3.3)  
#     3.4 [Summary of EDA](#3.4)  
# 4 [Preprocessing](#4)  
#     4.1 [Data Preparation](#4.1)  
#     4.2 [Fearture Engineering](#4.2)  
# 5 [Model Deployment](#5)  
#     5.1 [Model Development](#5.1)  
#     5.2 [Model Tuning](#5.2)  
#     5.3 [Model Validation](#5.3)  
# 6 [Deploy](#6)  

# + [markdown] tags=[]
# <a id="1"></a>
# ***
# ***
# # Project Summary
# ***
# -

# ### Situation
# The demand for rideshearing is drastically growing, especially in large cities. Uber is the first ride-hailing company and has operation in over 900 metropolitan areas worldwide. This project aims to investigate whether weather can make an impact on the demand for Uber rides in New York.  
#  

#  
# ### Action and Goal
# Using the Uber trip data (the mid-6 months of 2014 and the first 6 months of 2015) and the weather data, data exploration and model deployment are implemented by using Python and jupyter notebook. The goal is to build a predictive model and test the hypothesis that ***"Weather makes an impact on the demand for the uber rides in New York"***.

# ARIMA: Auto Regressive Integrated Moving Average

# + [markdown] tags=[]
# ## Assumptions
# ---
# - Since all boroughs are neighboring the same weather information in NY was used.
# - Since the available Uber trip data is for only 6 months of 2015, the result might have bias such as seasonal trend and anomary events.
# -

# ## Comments(to be rivised)
# In a more optimized version we may use more localized weather stations but the area is relatively narrow for significant weather differences.
# Additionally, using information from different stations may enter noise by various factors (like missing values or small calibration differences).

# + [markdown] tags=[]
# <a id="2"></a>
# ***
# ***
# # Loading 
# ***
# This section loads required libraries, data as well as custom-made functions.
# -

# <a id="2.1"></a>
# ***
# ***
# ## Import Libraries
# ***
# These libraries can be installed from **Pipfile** and **Pipfile.lock** using pipenv (**Python version: 3.8.10**)

# + tags=[]
# Import reqired libraries -----

import os, sys
sys.path.append(os.pardir)

import math
import warnings
warnings.filterwarnings('ignore')
from itertools import compress
from datetime import date, datetime
from functools import reduce

# EDA
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew

# Visualisation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
plt.rcParams['font.size'] = '16'# Set general font size

# Modelling
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from boruta import BorutaPy
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)


# -

# <a id="2.2"></a>
# ***
# ***
# ## Functions 
# ***
# Defined custom functions are shown in this section.

# ***
# **Memory Usage**
# ***
# Check the memory usage. If a variable not used occupy a large memory, it can be deleted. (var = dir() as input)

# + tags=[]
def memory_usage(var, lower_limit=0):
    ''' Memory Usage
    This code provides information about the memory usage
    
    Parameters
    ----------
    var = dir()
    lower_limit (default:0, optional): define the minimam value of the memory usage displayed
    
    Return
    ------
    print memory usage
    '''
    
    # input: var = dir()
    print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
    print(" ------------------------------------ ")
    for var_name in var:
        if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > lower_limit:
            print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
    return


# + [markdown] tags=[]
# ***
# **Data Profile**
# ***
# This function provides a summary of data structure in dataframe such as data types, count of the null data and count of the unique values. (See [Profile](#profile))

# + tags=[] jupyter={"source_hidden": true}
def data_profile(df):
    ''' Data Profile
    This code provides data profile for the inpiut dataframe
    
    Parameters
    ----------
    df (dataframe)
    
    Return
    ------
    df_profile (dataframe)
    '''
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
# ***
# ***
# ## Import Data
# <a id="2.3"></a>
# ***
# All data used in this project is stored in "data" directory.  
#   
# **Uber trip data from 2015** (January - June)  
# : with less fine-grained location information  
# This data contains 14.3 million more Uber pickups from January to June 2015.
# - uber-raw-data-janjune-15.csv
#   
# **Weather data**  
# This dataset contains ~5 years of high temporal resolution (hourly measurements) data of various weather attributes shown as below:
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

# Raw data between January and June, 2015
uber_raw_janjun15 = pd.read_csv(uber_data_dir + 'janjune-15.csv', parse_dates=['Pickup_date'])
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

# + tags=[]
# Check memory usage
# var = dir()
# memory_usage(var, lower_limit=1000)

# + [markdown] tags=[]
# ***
# ***
# # Exploring Data Analysis (EDA) 
# <a id="3"></a>
# ***

# + [markdown] tags=[]
# <a id="3.1"></a>
# ***
# ***
# ## Data Overview
# ***
# This section provides the overview of the Uber trip data and the weather data.

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# <a id="3.1.1"></a>
# ***
# ### Input Variables 
# ***
# **Uber trip data (2015)**  
# - Dispatching_base_num : The TLC base company code of the base that dispatched the Uber
# - Pickup_date : The date and time of the Uber pickup
# - Affiliated_base_num : The TLC base company code affiliated with the Uber pickup
# - locationID : The pickup location ID affiliated with the Uber pickup
#
# **Weather data**  
# - datetime : The date and time of the weather record  
#   
# **taxi zone: reference of the borough**
# - Location ID
# - Borough
# - Zone
# ***

# + [markdown] tags=[]
# <a id="3.1.1"></a>
# ***
# ### Uber rides data 
# ***
# -

# Number of rows and columns
uber_raw_janjun15.shape

# Uber trip data from 2015
uber_raw_janjun15.head()

# Location reference
taxi_zone.head()

# #### Add borough information based on location ID

# Borough
uber_raw_janjun15['borough'] = uber_raw_janjun15['locationID']\
                            .map(taxi_zone.set_index('LocationID')['Borough'])
uber_raw_janjun15['borough'].unique()

uber_raw_janjun15.head()

# #### Format date and time

# convert string to timestamp object
uber_raw_janjun15['datetime'] = pd.to_datetime(uber_raw_janjun15['Pickup_date'])

# extract date and time (hour)
uber_raw_janjun15['datehour'] = uber_raw_janjun15['datetime'].dt.floor('1h')

# #### Time period of the uber trip data

print('Uber trip data from 2015')
print('Min date: %s' % uber_raw_janjun15['datetime'].min())
print('Max date: %s' % uber_raw_janjun15['datetime'].max())

# #### Data Profile 

data_profile(uber_raw_janjun15)

# <div class='alert-info'>
#
# - The uber trip data for 2015 has a lot of duplicated rows. This just indicates there were rides at the same time in the same pick up area.  
# - 1.1% of Affliated_base_num is NaN values.
#
# </div>

# <div class='alert-success'>
#
# There is no problem with the raw data of the 2015 Uber rides at this stage. No further action required.
#
# </div>

# + [markdown] tags=[]
# <a id="3.1.2"></a>
# ***
# ### Weather data 
# ***
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

# +
# Weather data (01/01/2015 - 31/06/2015)
weather_NY_15 = weather_NY.loc[(weather_NY['datetime'] < '2015-07')&(weather_NY['datetime'] > '2015')]

del weather_NY

# + tags=[]
# Check memory usage
var = dir()
memory_usage(var, lower_limit=1000)
del var
# -
# #### Weather descriptions

weather_state = weather_NY_15['weather_description'].unique()
print(weather_state)
print('\n The number of the weather states is', len(weather_state))

# #### Temperature 

#Convert temperature (K) into temperature (C: degree Celsius)
weather_NY_15['temperature'] = weather_NY_15['temperature'] - 273.15

# #### Format date and time  

weather_NY_15['datetime'] = pd.to_datetime(weather_NY_15['datetime'])

# extract date and time (hour)
weather_NY_15['datehour'] = weather_NY_15['datetime'].dt.floor('1h')

print('Weather data (2015)')
print(weather_NY_15['datetime'].min())
print(weather_NY_15['datetime'].max())

# #### Basic summary of the weather data 

# + tags=[]
data_profile(weather_NY_15)
# -

weather_NY_15.describe().T\
    .style.bar(subset=['mean'], color =px.colors.qualitative.G10[0])\
    .background_gradient(subset=['std'], cmap='Greens')\
    .background_gradient(subset=['50%'], cmap='coolwarm')

# <div class='alert-success'>
#
# The weather data for the first 6 months in 2015 does not have missing data and duplicated data 
#
# </div>

# + [markdown] tags=[]
# <a id="3.2"></a>
# ***
# ***
# ## Data preparation for analysis
# ***
# This section provides required variables and dataframe for further analysis in the later section. The created data is shown below:  
# - df_hourly_rides:  
# contains the number of rides per hour and the weather information for each time stamp.
# - df_hourly_rides_borough:  
# contains the number of rides per hour by borough and the weather information for each time stamp.

# + [markdown] tags=[]
# ### Hourly demand for rides

# +
# hourly rides from all borough
# hourly_Uber_rides = uber_raw_janjun15.Pickup_date.value_counts().resample('H').sum().reset_index()
# hourly_Uber_rides.columns  = ['datetime', 'count']
# hourly_Uber_rides.head()
# -

# hourly rides from all borough
hourly_total_rides = uber_raw_janjun15[['datehour','datetime']]\
                    .groupby('datehour')\
                    .count()\
                    .reset_index()
hourly_total_rides.columns  = ['datehour', 'count']

hourly_total_rides.head()

# + [markdown] tags=[]
# ### Hourly demand for rides by borough
# -

# hourly rides from all borough
hourly_rides_borough = uber_raw_janjun15[['datehour','datetime','borough']]\
                        .groupby(['datehour','borough'])\
                        .count()\
                        .reset_index()
hourly_rides_borough.columns  = ['datehour','borough', 'count']

hourly_rides_borough.head()

# + [markdown] tags=[]
# ### Merge Uber rides data and weather data
# -

df_hourly_rides = pd.merge(hourly_total_rides, weather_NY_15, on='datehour')
df_hourly_rides_borough = pd.merge(hourly_rides_borough, weather_NY_15, on='datehour')

# add day of week(0:Sun, 1:Mon,...,6)
df_hourly_rides['day_of_week'] = df_hourly_rides['datetime'].dt.strftime('%w')
df_hourly_rides_borough['day_of_week'] = df_hourly_rides_borough['datetime'].dt.strftime('%w')

df_hourly_rides.head()

# + [markdown] tags=[]
# <a id="3.3"></a>
# ***
# ***
# ## Visualization 
# ***
# In this section, some key plots are shown.

# + [markdown] tags=[]
# ### The Number of Rides by Borough 

# + tags=[]
# The number of rides by borough
total_rides_by_borough = df_hourly_rides_borough.groupby('borough')['count'].agg(sum).reset_index()
total_rides_by_borough.columns = ['borough','count']
total_rides_by_borough

# + tags=[]
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x='borough', y = 'count', data = total_rides_by_borough)
ax.set_title('Number of Rides by Borough');
ax.set_xlabel('Borough', fontsize=16);
ax.set_ylabel('Number of Rides', fontsize=16);

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)
    
plt.savefig('images/rides_borough_barchart.png')

# + tags=[]
fig, ax = plt.subplots(figsize=(15,10))
borough = total_rides_by_borough['borough'].unique()
size = total_rides_by_borough['count']
#create pie chart
patches, texts = plt.pie(size, startangle=90, shadow=False)

# Legend setting
percent = 100.*size/size.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(borough, percent)]
sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, size),
                                          key=lambda x: x[2],
                                          reverse=True))
plt.legend(patches, labels, loc='upper right', bbox_to_anchor=(0.1, 1.), fontsize=16)

# Savefig
plt.savefig('images/rides_borough_piechart.png', bbox_inches='tight')

plt.show()
# -

# <div class='alert-info'>
#
# Manhattan has the most populous county followed by Brooklyn and Queens.  
#
# </div>

# + [markdown] tags=[]
# ### Average Number of Rides by Weather Event

# +
avg_rides_weather = df_hourly_rides.groupby('weather_description')['count'].mean().reset_index()
avg_rides_weather.columns = ['weather_description', 'avg_rides']
avg_rides_weather=avg_rides_weather.sort_values(by='avg_rides')

fig, ax = plt.subplots(figsize=(15,7))
# sns.barplot(x='weather_description', y = 'avg_rides', data = avg_rides_weather)
sns.barplot(y='weather_description', x='avg_rides', data = avg_rides_weather, palette='winter')
plt.xticks(rotation=90);
ax.set_title('Average Number of Rides by Weather Event')
ax.set_ylabel('Weather Events', fontsize=20)
ax.set_xlabel('Average Number of Rides', fontsize=20)
ax.invert_yaxis()
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

# Savefig
figname = 'Avg_rides_by_weather'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')
# -

# <div class='alert-info'>
#
# 'Proximity thunderstorm', 'thunderstorm with light rain' and 'very heavy rain' have more rides. However, these weather events were probably not often occured. So need to be careful to investigate the correlation.
#
# </div>

# + [markdown] tags=[]
# ### Average Number of Rides by Event and Borough

# +
avg_rides_weather_borough = df_hourly_rides_borough.groupby(['weather_description','borough'])['count'].mean().reset_index()
avg_rides_weather_borough.columns = ['weather_description', 'borough', 'avg_rides']

fig, ax = plt.subplots(figsize=(25,10))
sns.barplot(x='weather_description', y = 'avg_rides', hue = 'borough', data = avg_rides_weather_borough)
plt.xticks(rotation=90);
ax.set_title('Average Number of Rides by Event and Borough');
ax.set_xlabel('Weather Events', fontsize=16);
ax.set_ylabel('Average Number of Rides', fontsize=16);
plt.legend(loc='upper left')

figname = 'Avg_rides_by_weather_borough'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')

# + [markdown] tags=[]
# ### Number of Rides per hour by Month
# -

df_hour = df_hourly_rides[['datetime','datehour','day_of_week','count']]
hourly_rides_by_month = df_hour.copy()
hourly_rides_by_month['month'] = hourly_rides_by_month['datetime'].dt.month
hourly_rides_by_month['dayhour'] = hourly_rides_by_month['datetime'].dt.strftime('%d %H')
hourly_rides_by_month = hourly_rides_by_month.pivot('dayhour','month','count')

hourly_rides_by_month

# +
fig, ax = plt.subplots(figsize=(20,10))

sns.lineplot(data=hourly_rides_by_month)
ax.set_ylabel('Counts of Uber rides');
ax.set_xlabel('Day-Hour');
# hours = mdates.HourLocator(interval = 10000)
# ax.xaxis.set_major_locator(hours)

# + [markdown] tags=[]
# ### Average Number of the Rides by Month 
# -

monthly_rides = df_hourly_rides.copy()
monthly_rides['month'] = df_hourly_rides['datetime'].dt.month
monthly_rides = monthly_rides[['count','month']].groupby(['month']).agg(sum).reset_index()
monthly_rides

# +
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x='month', y = 'count', data = monthly_rides)
ax.set_title('Number of Rides by Month');
ax.set_xlabel('Month', fontsize=16);
ax.set_ylabel('Number of Rides', fontsize=16);

figname = 'N_rides_per_month'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')
# -

# <div class='alert-info'>
#
# - January has the lowest number of rides.
# - June has the highest number of the rides.
#
# </div>

# + [markdown] tags=[]
# ### Histogram of Number of Rides per hour
# -

sns.distplot(df_hourly_rides['count'], fit=norm);

# Log of the rides counts
sns.distplot(np.log1p(df_hourly_rides['count']), fit=norm);

# <div class='alert-danger'>
#
# Distribution of the hourly rides is skewed.
#
# </div>

# + [markdown] tags=[]
# ### Histogram of the Uber Rides by the Weather Events 

# +
# df_hourly_rides.hist(column="count", by='weather_description'
#         , sharex=True, sharey=False
#         , figsize=(20,15), layout=(4,5)
#         , color='b'
#        )

# plt.show()

# +
fig, axes = plt.subplots(4, 5, figsize=(25, 15))
weather_list = df_hourly_rides['weather_description'].unique()
for weather, ax in zip(weather_list, axes.flat):
    df_hourly_rides_weather = df_hourly_rides[df_hourly_rides['weather_description']==weather]
    sns.histplot(data=df_hourly_rides_weather, x='count', ax=ax
                ,bins=20, common_bins=True, binrange=[0,12000], color='#1f77b4')
  
    ax.set_ylabel('');
    ax.set_xlabel('');
    ax.set_title(weather);
for i in range(0,15):
    axes.flat[i].set_xticklabels([])

figname = 'N_rides_vs_WeatherEvents_hist'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')
# -

EventCount_by_weather = df_hourly_rides.groupby('weather_description').count()['count'].reset_index()
EventCount_by_weather.columns = ['Weather Event', 'Number of Occurence']
EventCount_by_weather.to_csv('results/WeatherEventOccurenceCount.csv', index=False)

# <div class = 'alert-danger'>
#
# - Some weather conditions (drizzle, fog, haze, heavy intensity rain, light intensity drizzle, proximity thunderstorm, snow, thunderstorm, thunderstorm with light rain, very heavy rain) are not frequent events (< 50 events)  
# - The histogram is right-skewed
#
# </div>

# + [markdown] tags=[]
# ### Avrerage Demand through the Week

# +
df_hour = df_hourly_rides[['datetime','datehour','day_of_week','count']]
df_hour['hour']=df_hour['datetime'].dt.strftime('%H')

hourly_rides_day_mean = df_hour.groupby(['hour','day_of_week']).mean().unstack()
hourly_rides_day_mean.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

fig, axes = plt.subplots(1,2, figsize=(25,10))

sns.lineplot(data=hourly_rides_day_mean, ax=axes[0])
axes[0].set_ylabel('Average Number of Rides')
axes[0].set_xlabel('Hour of Day')
axes[0].set_title('Average Demand through the Week')
axes[0].set_xticks(range(0,24,2));

sns.heatmap(hourly_rides_day_mean, ax=axes[1]
            , cmap='coolwarm', cbar_kws={'label':' Average Rides per Hour'})
axes[1].invert_yaxis()
axes[1].set_xlabel('Day of Week', fontsize=16);
axes[1].set_ylabel('Hour of Day', fontsize=16)
axes[1].set_title('Average Demand through the Week (Heatmap)')
axes[1].set_yticklabels(['0','','','3','',''
                         ,'6','','','9','',''
                         ,'12','','','15','',''
                         ,'18','','','21','','']);

figname = 'avgN_rides_per_Hour_heatmap'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')

# + [markdown] tags=[]
# ### Distribution of Demands per Day
# -

df_rides_day = df_hourly_rides[['datetime','day_of_week','count']]
df_rides_day['day_of_week'] = df_rides_day['datetime'].dt.strftime('%a')
df_rides_day['date'] = df_rides_day['datetime'].dt.strftime('%Y-%m-%d')
daily_rides = df_rides_day.groupby(['date','day_of_week']).agg(sum).reset_index()
daily_rides.head()

# +
fig, ax = plt.subplots(figsize=(20,15))

order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
# sns.boxplot(x='day_of_week', y='count', data=daily_rides, order=order)
sns.violinplot(x='day_of_week', y='count', data=daily_rides, order=order)
sns.swarmplot(x='day_of_week', y='count', data=daily_rides, order=order)
plt.gcf().autofmt_xdate()
ax.set_xlabel('Day of Week', fontsize=16);
ax.set_ylabel('Number of Rides per Hour', fontsize=16);
figname='Dist_N_rides_DayOfWeek'
plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
# ax.legend()

# + [markdown] tags=[]
# ### Average Demand through the week by Borough
# -

df_hour_borough = df_hourly_rides_borough[['datetime','datehour','day_of_week','count','borough']]
df_hour_borough['hour'] = df_hour_borough['datetime'].dt.strftime('%H')

# +
list_borough = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR']
fig, axes = plt.subplots(3, 2, figsize=(25,30))

for borough, ax in zip(list_borough, axes.flat):
    rides_borough = (
        df_hour_borough.loc[df_hour_borough['borough']==borough]
        .groupby(['hour','day_of_week']).mean().unstack()
    )
    rides_borough.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
    rides_borough=rides_borough.fillna(0)
    sns.heatmap(rides_borough 
                ,ax=ax,cmap='coolwarm',cbar_kws={'label':' Number of Rides per hour'})
    ax.invert_yaxis()
    ax.set_title(borough);
    ax.set_xlabel('Day of Week');
    ax.set_ylabel('Hour')
    if borough is not 'EWR':
        ax.set_yticklabels(['0','','','3','',''
                         ,'6','','','9','',''
                         ,'12','','','15','',''
                         ,'18','','','21','','']);
        
figname = 'avgN_rides_per_Hour_heatmap_by_Borough'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')

# + [markdown] tags=[]
# ### Correlation map ------------------ (Need to write Comment)

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
fig, ax = plt.subplots(figsize=(30,30))

sns.heatmap(df_corr.corr(),vmax=1, vmin=-1, cmap='bwr'
            , square=True, ax=ax, linewidths=0.1, center=0
            , cbar_kws={'label':' Correlation'})
ax.set_xlabel('Features', fontsize=30);
ax.set_ylabel('Features', fontsize=30);
ax.set_xticklabels(df_corr.columns,fontsize=20);
ax.set_yticklabels(df_corr.columns,fontsize=20);

figname = 'Correlation_map'
plt.savefig(('images/'+figname+'.png'), bbox_inches='tight')

# +
fig, ax = plt.subplots(figsize =(20,10))
corr_matt_sub = df_corr.corr().drop('count', axis=0)

corr_values = pd.DataFrame({'x':corr_matt_sub['count'][:].values, 'y':corr_matt_sub['count'][:].index})
corr_values['abs_x'] = np.abs(corr_values['x'])
corr_values = corr_values.sort_values(by='abs_x', ascending=False)
corr_values = corr_values.drop('abs_x', axis=1)


color_cor = []
for i in corr_values['x']:
    if i >= 0:
        color_cor.append('red')
    else:
        color_cor.append('blue')

plt.barh(corr_values['y'][:15], corr_values['x'][:15]
        , color = color_cor)

plt.xlim([-1, 1])
ax.set_xlabel('Correltaion strength');
ax.set_title("Correlation strength for total number of rides");
ax.set_ylabel('Features')
ax.invert_yaxis();
figname='CorrelationStrength_TotalRides'
plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
plt.show()
# -

# Correlation 
corr_matt_sub['count'][:]

# ## Key findings from data explorartion
# ---
# <div class='alert-info'>
#
# - It seems that **time variables (time, day of week)** have much stronger effect on demand for the rides than **weather variables**.
#
# </div>

# + [markdown] tags=[]
# ***
# ***
# # Preprocessing 
# <a id="4"></a>
# ***
# Based on the insights obtained from [EDA](#3) and [Visualisation](#4.3), Preprocess the data used in modelling.

# + [markdown] tags=[]
# ## Feature Engineering
# -

df_hourly_rides.head()

df_hour_borough

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Data preparation for modelling
# ---

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
    
# log transformation
# df_model_rides['count'] = np.log1p(df_model_rides['count'])
# df_model_rides['rides_Brooklyn'] = np.log1p(df_model_rides['rides_Brooklyn'])
# df_model_rides['rides_Manhattan'] = np.log1p(df_model_rides['rides_Manhattan'])
# df_model_rides['rides_Queens'] =np.log1p(df_model_rides['rides_Queens'])

df_model_rides.columns

# + [markdown] tags=[]
# ### Create Lag Features 
# ---
# In time series data, at any point in time, the model needs information about the past. Here, to pass the past (most recent) information, lag feature is created.

# +
lag_variables = [
       'count'
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

# + tags=[]
#Create Lag Features
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

df_model

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ***
# ***
# # Modelling
# <a id="5"></a>
# ***
#

# +
# Split the date into test data and Validation data

# Length of test data (days)
test_length = 30

df_train = df_model[df_model['time_block_num'] <= (df_model['time_block_num'].max()-24*test_length)]
df_test = df_model[df_model['time_block_num'] > (df_model['time_block_num'].max()-24*test_length)]

print('Time period of Train data')
print('Min time: %s' % df_train['datetime'].min())
print('Min time: %s' % df_train['datetime'].max())
print('\nTime period of Test data')
print('Min time: %s' % df_test['datetime'].min())
print('Min time: %s' % df_test['datetime'].max())
# -
# Keep the datetime info with 'time_block_num'
time_block = df_model[['datetime', 'time_block_num']]
df_model = df_model.drop('datetime', axis=1)
df_train = df_train.drop('datetime', axis=1)
df_test = df_test.drop('datetime', axis=1)


# + tags=[]
def prediction_XGBoost(df_train, df_test, model, target_list
                       , feature_select=False, log_transform=False, include_borough=False):
    
    def _mape(true, pred): 
        true, pred = np.array(true), np.array(pred)
        return np.mean(np.abs((true - pred) / true)) * 100
    
    results = {}
    for target in target_list:
#         print(model)
        print('Forecasting:'+target)
        new_df_train = df_train.copy()
        new_df_test = df_test.copy()
        
        # Define Y_train and Y_test (Target label)
        Y_train = new_df_train[target]
        Y_test = new_df_test[target]
        
        
        drop_list = target_list.copy() # list of dropped variables

        ##----- When rides variables for majour borough are not included ------------------##
        if include_borough:
            del_borough_list = target_list.copy()
            del_borough_list.remove(target)
            for del_borough in del_borough_list:
                all_columns_df_model = list(df_model.columns) # all columns of df_train
                drop_borough_col = df_train.columns.str.contains(del_borough+'_') #boolean
                drop_borough_list = list(compress(col_name_df_model, drop_borough_col))
                # Add dropped columns on the drop_list
                drop_list = drop_list + drop_borough_list
        ##---------------------------------------------------------------------------------##
        
        # Define X_train and X_test (Features)
        X_train = new_df_train.drop(drop_list, axis=1)
        X_test = new_df_test.drop(drop_list, axis=1)
        
        
        
        # Feature selection----------------------------------------------------------------##
        if feature_select:
            feature_selector =BorutaPy(model
                                   ,n_estimators='auto' 
                                   ,verbose=0 # 0: no output,1: displays iteration number,2: which features have been selected already
                                   ,alpha=0.1
                                   ,max_iter=100
                                   ,random_state=42
                                  )
            feature_selector.fit(X_train.values, Y_train.values)
            # Select only selected feature
            X_train = X_train.iloc[:,feature_selector.support_]
            X_test = X_test.iloc[:,feature_selector.support_]
        
        # ---------------------------------------------------------------------------------##
        
        
        # Fitting
        model.fit(
            X_train, Y_train, 
            eval_set=[(X_train, Y_train), (X_test, Y_test)], 
            verbose=0
#             early_stopping_rounds = 10
        )
        
        # Prediction
        Y_predict = model.predict(X_test)
        if log_transform:
            # Accuracy (RMSE)
            mse = mean_squared_error(Y_test,Y_predict)
            rmse = np.sqrt(mse)
            print('RMLSE: ' + str(np.round(rmse, 2)))

            # Accuracy (MAPE)
            mape = _mape(np.expm1(Y_test), np.expm1(Y_predict))
            print('MAPE: ' + str(np.round(mape, 2)) +'%')
        
            # Results
            results[target] = {'feature_importances': model.feature_importances_
                               ,'mape': mape
                               ,'rmse': rmse
                               ,'X_train': X_train
                               ,'X_test': X_test
                               ,'Y_train': np.expm1(Y_train)
                               ,'Y_test': np.expm1(Y_test)
                               ,'Y_predict': np.expm1(Y_predict)
                          }
            
            
            
            ## Plot feature importances
            fig, ax = plt.subplot()
            
            
            
            
        else:
            
            # Accuracy (RMSE)
            mse = mean_squared_error(Y_test, Y_predict)
            rmse = np.sqrt(mse)
            print('RMSE: ' + str(np.round(rmse, 2)))

            # Accuracy (MAPE)
            mape = _mape(Y_test, Y_predict)
            print('MAPE: ' + str(np.round(mape, 2)) +'%')
            
            # Results
            results[target] = {'feature_importances': model.feature_importances_
                               ,'mape': mape
                               ,'rmse': rmse
                               ,'X_train': X_train
                               ,'X_test': X_test
                               ,'Y_train': Y_train
                               ,'Y_test': Y_test
                               ,'Y_predict': Y_predict
                              }
            
    return results
# -

target_list = ['count', 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens']

# ## Create Baseline 


# Basic Model (Without parameter tuning)
model = XGBRegressor(seed=42)

results1 = prediction_XGBoost(df_train, df_test, model, target_list)

# ## Model Optimization

# ### Parameter Tuning

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

results2 = prediction_XGBoost(df_train, df_test, model, target_list)

# ### Feature Selection by Boruta 

# + tags=[]
model = XGBRegressor(
    max_depth=20,
    n_estimators=1000,
#     min_child_weight=300, 
#     colsample_bytree=0.8, 
#     subsample=0.8, 
#     eta=0.3,    
    seed=42)

# + tags=[]
results3 = prediction_XGBoost(df_train, df_test, model, target_list, feature_select=True)
# -



# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# <a id="6"></a>
# ***
# ***
# # Deploy 
# ***

# +
def plot_feat_imp(results, target_list, num_feat=10, label_str=''):
    df_res = results.copy()
    for target in target_list:
        print(target)
        #Extract data
        feat_imp_target = results[target]['feature_importances']
        X_train_target = results[target]['X_train']
        
        df_feat_imp = pd.DataFrame({'col': X_train_target.columns,'imp':feat_imp_target})
        df_feat_imp = df_feat_imp.sort_values(by='imp',ascending=False)
        if df_feat_imp.shape[0] > num_feat:
            df_feat_imp = df_feat_imp.iloc[:10]
            
        # Plot
        fig, ax = plt.subplots(figsize=(20,5))
        
        bar = ax.barh(df_feat_imp.col, df_feat_imp.imp, 0.6
                     , color=mcolors.TABLEAU_COLORS)
        ax.set_ylabel('Feature', fontsize=16);
        ax.set_xlabel('Feature Importances');
        ax.set_title('Feature Importances - '+target+label_str);
        ax.bar_label(bar, fmt='%.02f');
        
        ax.invert_yaxis();
        
        figname='Feature Importances - '+target+label_str
        plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
        fig.tight_layout() 
        plt.show()
        
        
# -

def plot_feat_imp_multi(results, target_list, num_feat=10, label_str=''):
    df_res = results.copy()
    fig, axes = plt.subplots(len(target_list), 1, figsize=(20,8*len(target_list)))
    for target, ax in zip(target_list, axes):
        print(target)
        #Extract data
        feat_imp_target = results[target]['feature_importances']
        X_train_target = results[target]['X_train']
        
        df_feat_imp = pd.DataFrame({'col': X_train_target.columns,'imp':feat_imp_target})
        df_feat_imp = df_feat_imp.sort_values(by='imp',ascending=False)
        if df_feat_imp.shape[0] > num_feat:
            df_feat_imp = df_feat_imp.iloc[:10]
            
        # Plot
        
        
        bar = ax.barh(df_feat_imp.col, df_feat_imp.imp, 0.6
                     , color=mcolors.TABLEAU_COLORS)
        ax.set_ylabel('Feature', fontsize=16);
        ax.set_xlabel('Feature Importances');
        ax.set_title('Feature Importances - '+target+label_str);
        ax.bar_label(bar, fmt='%.02f');
        
        ax.invert_yaxis();
        
    figname='Feature Importances - multi'+label_str
    plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
    plt.show()

# ## Feature Importance

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Base Model
# -

#Base Model
plot_feat_imp_multi(results1, target_list, label_str='-BaseModel')

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Tuned Model1 
# -

plot_feat_imp_multi(results2, target_list, label_str='-TunedModel1')

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Tuned Model2 
# -

plot_feat_imp_multi(results3, target_list, label_str='-FeatSelectModel')


# ## Predicted values vs True Values

# +
def plot_true_pred(results, target_list, time_block, label_str=''):
    
    
    for target in target_list:
        df_res = results.copy()
        print(target)
        #Extract data
        df_plot = df_res[target]['Y_test']
        df_plot = pd.merge(df_plot, time_block, left_index=True, right_index=True)
        df_plot = df_plot.drop('time_block_num', axis=1)
        
        df_plot['predict'] = df_res[target]['Y_predict']
        # Plot
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df_plot['datetime'], y = df_plot[target],
                    mode='lines', 
                    name='True',
                    line=dict(color='blue')
                                ))
        fig.add_trace(go.Scatter(x = df_plot['datetime'], y = df_plot['predict'],
                    mode='lines',
                    name='Predict',
                    line=dict(color='red')
                                ))
        fig.update_layout(
            autosize=False
            , width=1600
            , height=500
            , yaxis=dict(title_text="Number of Rides"
                        ,titlefont=dict(size=18)
                        )
            , xaxis=dict(title_text="Date/Time"
                        ,titlefont=dict(size=18)
                        )
        )
        fig.update_yaxes(automargin=True, tickfont=dict(size=14)
                        , showline=True, linewidth=1, linecolor='black', mirror=True
                        , showgrid=True, gridwidth=1, gridcolor='#D0D0D0'
                        )
        fig.update_xaxes(automargin=True, tickfont=dict(size=14)
                        , showline=True, linewidth=1, linecolor='black', mirror=True
                        , showgrid=True, gridwidth=1, gridcolor='#D0D0D0'
                        )
        fig.layout.plot_bgcolor = '#fff'
        fig.layout.paper_bgcolor = '#fff'
        fig.show()
        
        

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Base Model
# -

# Base Model
plot_true_pred(results1, target_list, time_block)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Tuned Model 1 
# -

# Tuned Model1
plot_true_pred(results2, target_list, time_block)

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### Tuned Model 2
# -

# Tuned Model2
plot_true_pred(results3, target_list, time_block)
