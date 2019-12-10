from textblob import TextBlob
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tweepy
import math
import csv
import datetime
import numpy as np
import pandas as pd
import time
import pandas_datareader as dr
import matplotlib.pyplot as plt
import yfinance as yf
#from scale_data import preprocess_data


#Step 1 - Insert your API keys
consumer_key = open("../consumer_key.txt","r").read()
consumer_secret = open("../consumer_secret.txt","r").read()
access_token = open("../access_token.txt","r").read()
access_token_secret = open("../access_token_secret.txt","r").read()

#Step 1.1 Authentication(?)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Import company stock data
data1 = dr.get_data_yahoo('AAPL', '2010-01-02', '2019-12-05', interval="d") # Apple
data2 = dr.get_data_yahoo('NSRGY', '2010-01-02', '2019-12-05',interval="d") # Nestlé

#Step 3 - Simple visualization using the Adj Close --> Adj close is better than close because it has updates that occur overnight
fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(data1.index, data1['Adj Close'], c='black')
plt.plot(data2.index, data2['Adj Close'], c='red')
plt.ylabel("USD/share")
plt.grid()
plt.show()

#Volume
# plt.figure()
# plt.plot(data1["Volume"])
# plt.title('GE stock volume history')
# plt.ylabel('Volume')
# plt.xlabel('Days')
# plt.show()


# Step 4 - Preprocessing of data
training_set = data1.iloc[:,2:3].values
print(training_set)

#Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
 
print(X_train)

#Step 5 - Search for your company name on Twitter
#public_tweets = api.search('apple')

#Step 6 - LSTM recurrent neural network model

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017

test = data1.copy()
real_stock_price = test.iloc[:, 2:3].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((data1['Open'], test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 2499):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()