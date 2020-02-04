import csv
import pandas as pd
import nltk
import pickle
import random
from nltk.corpus import stopwords
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

#import tweets
trump_df = pd.read_csv("trump.csv", error_bad_lines=False, sep='\t') 
obama_df = pd.read_csv("obama.csv", error_bad_lines=False, sep='\t')

#clean the dataframe
obama_df = obama_df.iloc[:,[0]]
trump_df = trump_df.iloc[:2586,[0]]

#add column
obama_df['author'] = 0
trump_df['author'] = 1

# print(obama_df.info)
# print(trump_df.info)

#split data
obama_train = obama_df[:1000]
obama_test = obama_df[1001:2001]
trump_train = trump_df[:1000]
trump_test = trump_df[1001:2001]

data_train = pd.concat((obama_train[:], trump_train[:]), sort=False)
data_test = pd.concat((obama_test[:], trump_test[:]), sort=False)

#Shuffle rows in dataframe and reset index
data_train = data_train.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)

################################### CLASSIFIER #######################################

cv = CountVectorizer()
X_train_counts = cv.fit_transform(data_train.text)

tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB().fit(X_train_tfidf, data_train.author)

X_test_counts = cv.transform(data_test.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = model.predict(X_test_tfidf)

for doc, category in zip(data_test.text, predicted):
    author = data_test.author[category]
    if author==0:
        author = 'obama'
    elif author==1:
        author = 'trump'
    else:
        pass

###########################################################

####### CHECK ACCURACY #######

r_counter = 0
w_counter = 0
# if (data_test[i].author == predicted)
# print(data_test.author)
wrong = []
for i in range(len(predicted)):
    if (data_test.author[i] == predicted[i]):
        r_counter = r_counter + 1
    elif (data_test.author[i] != predicted[i]):
        w_counter = w_counter + 1
        wrong.append(data_test.text[i])

    else:
        print("what")
print("right:", r_counter)
print("wrong:", w_counter)


##############      #############

###### MAKE A GAME OF IT ######
while True:
    num = random.randint(0, len(predicted)-1)
    the_tweet = data_train.text[num]

    print('The tweet:')
    print(the_tweet)

    print('Please type your guess. 1 for Trump or 0 for Obama. Type anything else to quit')
    the_guess = int(input())
    if(the_guess != 1 and the_guess != 0):
        print("Exiting!")
        break

    prediction = predicted[num]
    truth = data_test.author[num]

    defined_guess = 'Trump' if the_guess == 1 else 'Obama'
    defined_pred = 'Trump' if the_guess == 1 else 'Obama'
    defined_truth = 'Trump' if the_guess == 1 else 'Obama'
    
    print('Your guess was: ', defined_guess)
    print('My guess is: ', defined_pred)
    print('The answer is: ', defined_truth)
