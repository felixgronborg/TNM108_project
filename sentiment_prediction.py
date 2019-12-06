from __future__ import unicode_literals
import tweepy
import csv
import numpy as np
import pandas as pd
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense

#Step 1 - Insert your API keys
consumer_key = open("../consumer_key.txt","r").read()
consumer_secret = open("../consumer_secret.txt","r").read()
access_token = open("../access_token.txt","r").read()
access_token_secret = open("../access_token_secret.txt","r").read()


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Import company stock data
data = pd.read_csv("stock_data/AAPL.csv") 
print(data.head)


#Step 3 - Search for your company name on Twitter
public_tweets = api.search('apple')

