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



target = [1 if i < 12500 else 0 for i in range(25000)]

# Verkar fungera som det ska förutom 2 tweets från Trump som vi får ta bort
#import tweets
trump_df = pd.read_csv("trump.csv", error_bad_lines=False, sep='\t') 
obama_df = pd.read_csv("obama.csv", error_bad_lines=False, sep=',')

# print("TRUMP TWEETS INFO:")
# print(trump_df.info(), "\n")
# print("OBAMA TWEETS INFO:")
# print(obama_df.info())

#clean the dataframe
obama_df = obama_df.iloc[:,[0]]
trump_df = trump_df.iloc[:2586,[0]]

#add column
obama_df['author'] = 0
trump_df['author'] = 1

# print(obama_df.info)
# print(trump_df.info)

#split data
obama_train = obama_df[:500]
obama_test = obama_df[501:601]
trump_train = trump_df[:500]
trump_test = trump_df[501:601]

data_train = pd.concat((obama_train[:], trump_train[:]))
data_test = pd.concat((obama_test[:], trump_test[:]))




#Shuffle rows in dataframe and reset index
data_train = data_train.sample(frac=1).reset_index(drop=True)
data_test = data_train.sample(frac=1).reset_index(drop=True)

################################### CLASSIFIER #######################################

cv = CountVectorizer()
X_train_counts = cv.fit_transform(data_train.Tweets)

tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, data_train.author)

X_test_counts = cv.transform(data_test.Tweets)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = model.predict(X_test_tfidf)



for doc, category in zip(data_train.Tweets, predicted):
	author = data_train.author[category]

	if(author==0):
		author = 'obama'

	elif(author==1):
		author = 'trump'

	else:
		pass

print('%r => %s' % (doc, author))


# temp_list = []
# for i in range(len(y_test)):
#     temp_list.append(np.argmax(predictions[[i]]))

cm = confusion_matrix(data_test.author, predicted)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(len(data_test.columns)),
            yticks=np.arange((len(data_test.columns))),
            xticklabels=data_test, 
            yticklabels=data_test,
            title='Confusion Matrix',
            xlabel= 'Predicted tweeter',
            ylabel='True tweeter')

#Rotate the x-tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.show()




# for doc, category in zip(docs_new, predicted):
# 	print('%r => %s' % (doc, train.target_names[category]))

#get sentiment
	# create TextBlob object of passed tweet text 

# for i in range (len(obamatweets)):

#     analysis = TextBlob(obamatweets[i])

# 	    # set sentiment 
#     if analysis.sentiment.polarity > 0:
# 	    print('positive')

#     elif analysis.sentiment.polarity == 0: 
# 	    print('neutral')

#     else: 
# 	    print('negative')


# # Check percentage
#     # picking positive tweets from tweets 
# 	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
# 	# percentage of negative tweets 
# 	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
# 	# percentage of neutral tweets 
# 	print("Neutral tweets percentage: {} % \ 
# 		".format(100*len(tweets - ntweets - ptweets)/len(tweets))) 