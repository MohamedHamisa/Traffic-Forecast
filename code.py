import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet

df = pd.read_csv('Traffic data.csv')
df.head()

# check null values
df.isnull().sum()

df.info()

# convert object to datetime datatype
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.info()

# EDA
plt.figure(figsize=(10,7))
plt.plot(df['Datetime'], df['Count'])
plt.show()

#Format data for the model
df.index = df['Datetime']
df['y'] = df['Count']
df.drop(columns=['ID', 'Datetime', 'Count'], axis=1, inplace=True)
df = df.resample('D').sum()
df.head()

df['ds'] = df.index
df.head()

#Input Split
size = 60
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=size/len(df), shuffle=False)
train.tail()

test.head()

model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(train)

future = model.make_future_dataframe(periods=60)
future

forecast = model.predict(future)
forecast.head()

model.plot_components(forecast)
pred = forecast.iloc[-60:, :]

len(pred)

# test results
plt.figure(figsize=(10,7))
plt.plot(test['ds'], test['y'])
plt.plot(pred['ds'], pred['yhat'], color='red')
plt.plot(pred['ds'], pred['yhat_lower'], color='green')
plt.plot(pred['ds'], pred['yhat_upper'], color='orange')
plt.show()

# input data
plt.plot(df['ds'], df['y'])
plt.show()


# forecast data
plt.plot(forecast['ds'], forecast['yhat'])
plt.show()model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(df)
future = model.make_future_dataframe(periods=200)
forecast = model.predict(future)
forecast.head()

# forecast data
plt.plot(forecast['ds'], forecast['yhat'])
plt.show()

