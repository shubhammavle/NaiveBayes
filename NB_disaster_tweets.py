# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:25:39 2024

@author: HP
"""

#libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#loading data
tweets_data = pd.read_csv('E:/datasets/Disaster_tweets_NB.csv', encoding='ISO-8859-1')
tweets_data.head()

"""
Business Problem: To predict salaries based on naive bayes
Constraits: finding best features to improve accuracy of the model, cleaning the data
"""

#cleaning data
tweets_data.head()
tweets_data.dtypes
tweets_data.info()

import re

def cleaning_text(i):
    W=[]
    i=re.sub("[^A-Za-z]+", " ", i).lower()
    for word in i.split(' '):
       if len(word)>3:
            W.append(word)
    return ' '.join(W)

tweets_data.text = tweets_data.text.apply(cleaning_text)
tweets_data.text

#model 
from sklearn.model_selection import train_test_split
tweets_train, tweets_test = train_test_split(tweets_data, test_size=0.2)

#applying it on naive bayes
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(tweets_data.iloc[:,:-1], tweets_data.iloc[:,-1])
#evaluation on test
pred = classifier_mb.predict(tweets_test.iloc[:,:-1])

np.mean(pred==np.array(tweets_test.iloc[:,-1]))

#testing on test data
test = pd.read_csv('E:/datasets/tweetsData_Test.csv', encoding='ISO-8859-1')
test.head()

def clean_tweets(i):
    return float(i[3:-1])

test.tweets = test.tweets.apply(clean_tweets)
test.hoursperweek = test.hoursperweek.apply(lambda x: float(x))
test = test.drop(columns=['workclass','education','educationno','maritalstatus', 'occupation','relationship','race','sex','native'])
test.dtypes

pred = classifier_mb.predict(test.iloc[:,:-1])