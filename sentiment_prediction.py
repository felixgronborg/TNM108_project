import tweepy
import math
import csv
import datetime
import sys
import jsonpickle
import os
import io
import time
import numpy as np
import pandas as pd 
import pandas_datareader as dr 
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from pandas.plotting import register_matplotlib_converters
from textblob import TextBlob
from bs4 import BeautifulSoup
from html import unescape
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler

import twitter_credentials

# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_api()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client
    
    def get_user_timeline_tweets(self, max_num_tweets=5000, since='2010-01-01 00:00:01'):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items():
            tweets.append(tweet)
        return tweets
    def get_api_search_tweets(self, query='*', max_num_tweets=500, startDate='2010-01-01 00:00:01'):
        tweets= []
        for tweet in Cursor(self.twitter_client.search, q=query, count=20,lang="en",since=startDate, tweet_mode='extended').items():
            tweets.append(tweet)
        return tweets

    def download_tweets(self, query="*", max_tweets=500, geocode="49.895077,-97.138451,2000mi", count=100, since_id=7250759366):  
        # 7250759366 is The first tweet 01-01-10 with the hashtag #newyear2010
        tweets = []
        last_id = -1
        filename = 'tweets.txt' # We'll store the tweets in a text file.
        max_id = -1 #Latest tweets
        tweetCount = 0
        print("Downloading max {0} tweets".format(max_tweets))
        with io.open(filename, 'w', encoding="utf-8") as f:
            while tweetCount < max_tweets:
                try:
                    if (max_id <= 0):
                        new_tweets = self.twitter_client.search(q=query, geocode="49.895077,-97.138451,2000mi", count=count, since_id=since_id)
                    else:
                        new_tweets = self.twitter_client.search(q=query, geocode="49.895077,-97.138451,2000mi", count=count, max_id=str(max_id - 1), since_id=since_id)
                    if not new_tweets:
                        print("No more tweets found")
                        break
                    for tweet in new_tweets:
                        if (not tweet.retweeted) and ('RT @' not in tweet.text):
                            tweets.append(tweet)
                            f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n\n')
                    tweetCount = len(tweets)
                    print("Downloaded {0} tweets".format(tweetCount))
                    max_id = new_tweets[-1].id
                except tweepy.TweepError as e:
                    # Just exit if any error
                    print("some error : " + str(e))
                    break

        print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, filename))
        return tweets

# # # # TWITTER AUTHENTICATOR # # # #
class TwitterAuthenticator():
    """
    Class for handling authentication
    """
    def authenticate_twitter_api(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth
        
# # # # TWITTER STREAMER # # # # 
class TwitterStreamer():
    """
    Class for streaming and processing live tweets
    """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, filename, keyword):
        # This handles Twitter authentication and the connection to Twitter Streaming API
        listener = TwitterListener(filename)
        auth = self.twitter_authenticator.authenticate_twitter_api()
        stream = Stream(auth, listener)

        # api = API(auth)

        stream.filter(track=keyword)

# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout
    """
    def __init__(self, filename):
        self.filename = filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.filename, 'w') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
    
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs
            return False
        print(status)

class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets
    """
    def clean_text(self, text):
        a = text
        b = ",.!?;"
        c = "&"

        for char in b:
            if char==c:
                a = a.replace(char,"and")
            else:
                a = a.replace(char,"")
        return a

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['id'] = np.array([tweet.id_str for tweet in tweets])
        #df['likes'] = np.array([tweet.likes for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        df['favorites'] = np.array([tweet.favorite_count for tweet in tweets])

        return df

keyword = "*"

filename = "tweets.txt"

twitter_client = TwitterClient(twitter_user='realDonaldTrump')
tweet_analyzer = TweetAnalyzer()
api = twitter_client.get_twitter_client_api()

#tweets = twitter_client.get_user_timeline_tweets()
tweets = twitter_client.get_api_search_tweets()
#tweets = twitter_client.download_tweets(query=keyword, max_tweets=500)


for tweet in tweets:
    tweet.text = tweet_analyzer.clean_text(tweet.text)
tweets_df = tweet_analyzer.tweets_to_data_frame(tweets)

tweets_df.to_csv('tweets_df.csv', sep='\t', encoding='utf-8', index=False)

# OR

#tweets_df = pd.read_csv("tweets_df.csv", error_bad_lines=False, sep='\t')


print(tweets_df.info())

"""
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
"""
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
