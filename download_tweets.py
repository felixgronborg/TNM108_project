import tweepy
import csv
import datetime
import numpy as np
import pandas as pd 
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from datetime import date

import twitter_credentials


# # # # TWITTER AUTHENTICATOR # # # #
class TwitterAuthenticator():
    """
    Class for handling authentication
    """
    def authenticate_twitter_api(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_api()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.twitter_user = twitter_user

    def get_user_timeline_tweets(self, startDate=datetime.datetime(1,1,1,0,0), endDate=datetime.datetime(9999,1,1,0,0)):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user, tweet_mode='extended').items():
            if (not tweet.retweeted) and ('RT @' not in tweet.full_text) and (tweet.created_at < endDate) and (tweet.created_at > startDate):
                            tweets.append(tweet)
        return tweets

class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets
    """
    def clean_text(self, text):
        a = text
        b = ",.!?;'"
        c = "&"

        for char in b:
            if char==c:
                a = a.replace(char,"and")
            else:
                a = a.replace(char,"")
        return a

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['Tweets'])

        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['id'] = np.array([tweet.id_str for tweet in tweets])
        #df['likes'] = np.array([tweet.likes for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        df['favorites'] = np.array([tweet.favorite_count for tweet in tweets])

        return df

tweet_analyzer = TweetAnalyzer()

##### Donald Trump #####
startDate = datetime.datetime(2017, 1, 20, 0, 0 ,0)
endDate = datetime.datetime(2021, 1, 1, 0, 0 ,0)
twitter_client = TwitterClient(twitter_user='realDonaldTrump')
tweets = twitter_client.get_user_timeline_tweets(startDate)
for tweet in tweets:
    tweet.full_text = tweet_analyzer.clean_text(tweet.full_text)
tweets_df = tweet_analyzer.tweets_to_data_frame(tweets)
tweets_df.to_csv('dtrump.csv', sep='\t', encoding='utf-8', index=False)
##### Donald Trump #####


##### Barack Obama #####
startDate = datetime.datetime(2009, 1, 20, 0, 0 ,0)
endDate = datetime.datetime(2017, 1, 20, 0, 0 ,0)
twitter_client = TwitterClient(twitter_user='BarackObama')
tweets = twitter_client.get_user_timeline_tweets(startDate,endDate)
for tweet in tweets:
    tweet.full_text = tweet_analyzer.clean_text(tweet.full_text)
tweets_df = tweet_analyzer.tweets_to_data_frame(tweets)
tweets_df.to_csv('bobama.csv', sep='\t', encoding='utf-8', index=False)
##### Barack Obama #####


