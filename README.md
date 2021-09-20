# Uber Demand Forecasting in NY
![UberNewYork](images/uber_NY2.jpg)
## Project Summary
**Situation**  
The demand for ride-sharing is drastically increasing, especially in large cities. Uber is the first ride-sharing company and has operations in over 900 metropolitan areas worldwide.  As rapidly growing demand and shortage of the vehicle, surge pricing for the rides is concerned these days. It is difficult to forecast the demand with various factors such as weather or public events. 

**Action and Goal**  
Time-series data for the Uber trip (January - June 2015) and the weather was used for exploratory data analysis, visualisation and forecasting model to test the hypothesis. All analysis was implemented on Python and Jupyter Notebook and a forecasting model was developed using a machine learning framework, called **XGBoost**. Using insights obtained from the analyses, this project aims to test the hypothesis **"Weather is a predictor of demand for Uber rides in New York"**.
  
**Results**  
The exploration and the build forecasting model shows that:  
- The demand follows specific daily and weekly patterns of hourly Uber rides.
- Weather variables did not have any or had very weak impacts on the forecasting model. i.e. the hypothesis was rejected. 

## Data
All data used in this project is stored in "data" directory.  
  
**Uber trip data from 2015** (January - June)  
: with less fine-grained location information  
This data contains 14.3 million more Uber pickups from January to June 2015.
- uber-raw-data-janjune-15.csv
  
**Weather data**  
This dataset contains ~5 years of high temporal resolution (hourly measurements) data of various weather attributes shown as below:
- humidity.csv
- pressure.csv
- temperature.csv
- weather_description.csv
- wind_direction.csv
- wind_speed.csv
  
**(Data Source)**  
The data shown above can be obtained from:  
- Uber trip data: https://github.com/fivethirtyeight/uber-tlc-foil-response  
- Histroical weather in NY: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=humidity.csv  

## Assumptions
- Since the available data is only for the first 6-month of the year 2015, the forecasting model may not accurately fit the true data.
- The seasonality of the data is ignored due to the limited time period of the data.
- Since all boroughs are neighbours, the weather information from a single source is used for all boroughs.
- The Forecasting is implemented for hourly demand for the rides because the weather data is hourly recorded.

## Results of EDA/Modelling
https://github.com/shotashirai/Uber-Demand-Forecasting-NYC/blob/main/Uber-Demand-Forecasting_EDA_Model.ipynb
