# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:42:30 2024

@author: HP
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#loading data
email_data = pd.read_csv('E:/datasets/sms_raw_NB.csv', encoding='ISO-8859-1')
email_data.head()
#cleaning data
import re

def cleaning_text(i):
    W=[]
    i=re.sub("[^A-Za-z]+", " ", i).lower()
    for word in i.split(' '):
        if len(word)>3:
            W.append(word)
    return ' '.join(W)

#tresting above function
cleaning_text("hope you are having good week just checking")
cleaning_text('hi how are you i am sad')

email_data.text = email_data.text.apply(cleaning_text)
email_data.head()

len(email_data)
email_data = email_data.loc[email_data.text != "", :]
len(email_data)

from sklearn.model_selection import train_test_split
email_train, email_test = train_test_split(email_data, test_size=0.2)

#creating matrix of tokens
def split_into_words(i):
    return [word for word in i.split(" ")]

emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)
all_emails_matrix = emails_bow.transform(email_data.text)
#for traing messages
train_emails_matrix = emails_bow.transform(email_train.text)
#for testing messages
test_emails_matrix = emails_bow.transform(email_test.text)

#applying tfidf
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
#preparing tfidf
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape

#applying it on naive bayes
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)
#evaluation on test
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==email_test.type)
print(accuracy_test_m)













