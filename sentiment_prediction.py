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
import sys
import jsonpickle
import os
import io
import numpy as np
<<<<<<< HEAD
import pandas as pd
import time
import pandas_datareader as dr
import matplotlib.pyplot as plt
import yfinance as yf
#from scale_data import preprocess_data

=======
import pandas as pd 
import pandas_datareader as dr 
import matplotlib.pyplot as plt
import yfinance as yf
#from textblob import TextBlob
from datetime import date
from pandas.plotting import register_matplotlib_converters
#from keras.models import Sequential
#from keras.layers import Dense

# 7250759366 is The first tweet 01-01-10 with the hashtag #newyear2010
def download_tweets(query="*", max_tweets = 500, geocode="49.895077,-97.138451,2000mi", count=100, since_id=7250759366):
    public_tweets = []
    last_id = -1
    fName = 'tweets.txt' # We'll store the tweets in a text file.
    max_id = -1
    tweetCount = 0
    print("Downloading max {0} tweets".format(max_tweets))
    with io.open(fName, 'w', encoding="utf-8") as f:
        while tweetCount < max_tweets:
            try:
                if (max_id <= 0):
                    new_tweets = api.search(q=query, geocode="49.895077,-97.138451,2000mi", count=count, since_id=since_id)
                else:
                    new_tweets = api.search(q=query, count=count, geocode="49.895077,-97.138451,2000mi", max_id=str(max_id - 1), since_id=since_id)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n\n')
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
    return
>>>>>>> 497cc064df9c64300d60011fafe0c87f9d490f76

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
data1 = dr.get_data_yahoo('AAPL', '2010-01-01', date.today(), interval="d") # Apple
data2 = dr.get_data_yahoo('NSRGY', '2010-01-01', date.today(), interval="d") # Nestlé

#Step 3 - Simple visualization using the Adj Close --> Adj close is better than close because it has updates that occur overnight
register_matplotlib_converters() #Removes a warning
fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(data1.index, data1['Adj Close'], c='black', label='Apple')
plt.plot(data2.index, data2['Adj Close'], c='red', label='Nestlé')
plt.ylabel("USD/share")
plt.legend()
plt.grid()
#plt.show() #Comment if not used

#download_tweets('nestle') #UNCOMMENT TO DOWNLOAD TWEETS

<<<<<<< HEAD
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
=======

#contributors  -
#coordinates  - 
#created_at  + 
#entities:
#   hashtags  ?
#   symbols  -?
#   urls  -?
#   user_mentions  -
#favorite_count  ?
#favorited  -
#geo  -
#id  -
#id_str  -
#in_reply_to_screen_name  -
#in_reply_to_status_id  -
#in_reply_to_status_id_str  -
#in_reply_to_user_id  -
#in_reply_to_user_id_str  -
#is_quote_status  -?
#lang  ?
#metadata  -
#possibly_sensitive  -?
#retweet_count  -
#retweeted  -
#source  -
#text  +
#truncated  -
#user:
#   contributors_enabled  -
#   created_at  -?
#   default_profile  -
#   default_profile_image  -
#   description  -
#   entities:
#       name  +?
#       massa annan skit  -
#source  -
#text  +
>>>>>>> 497cc064df9c64300d60011fafe0c87f9d490f76
