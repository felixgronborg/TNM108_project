from __future__ import unicode_literals
import tweepy
import csv
import datetime
import numpy as np
import pandas as pd
import pandas_datareader as dr

import matplotlib.pyplot as plt
import yfinance as yf
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense

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

#Step 4 - Search for your company name on Twitter
#public_tweets = api.search('apple')

