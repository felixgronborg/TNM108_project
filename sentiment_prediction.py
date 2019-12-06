from __future__ import unicode_literals
import tweepy
import csv
import datetime
import sys
import jsonpickle
import os
import io
import numpy as np
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
