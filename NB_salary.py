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
salary_data = pd.read_csv('E:/datasets/SalaryData_Train.csv', encoding='ISO-8859-1')
salary_data.head()

"""
Business Problem: To predict salaries based on naive bayes
Constraits: finding best features to improve accuracy of the model, cleaning the data
"""

#cleaning data
salary_data.head()
salary_data.dtypes

def clean_salary(i):
    return float(i[3:-1])

salary_data.Salary = salary_data.Salary.apply(clean_salary)
salary_data.hoursperweek = salary_data.hoursperweek.apply(lambda x: float(x))

salary_data.columns
salary_data = salary_data.drop(columns=['workclass','education','educationno','maritalstatus', 'occupation','relationship','race','sex','native'])
salary_data.columns

from sklearn.model_selection import train_test_split
salary_train, salary_test = train_test_split(salary_data, test_size=0.2)


#applying it on naive bayes
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(salary_data.iloc[:,:-1], salary_data.iloc[:,-1])
#evaluation on test
pred = classifier_mb.predict(salary_test.iloc[:,:-1])

np.mean(pred==np.array(salary_test.iloc[:,-1]))

#testing on test data
test = pd.read_csv('E:/datasets/SalaryData_Test.csv', encoding='ISO-8859-1')
test.head()

def clean_salary(i):
    return float(i[3:-1])

test.Salary = test.Salary.apply(clean_salary)
test.hoursperweek = test.hoursperweek.apply(lambda x: float(x))
test = test.drop(columns=['workclass','education','educationno','maritalstatus', 'occupation','relationship','race','sex','native'])
test.dtypes

pred = classifier_mb.predict(test.iloc[:,:-1])
np.mean(pred==np.array(test.iloc[:,-1]))