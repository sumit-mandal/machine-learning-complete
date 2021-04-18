# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = "\t",quoting=3) #by inputing quoting =  3 we are ignoring double quotes

#cleaning the texts
#getting rid of words like a,the,an, any punctuation,capitals, etc.

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #corpus in-general means collection of text of same type

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #we don't want to remove all the letters from a to z. i will take all the values from our dataset and change them
    review = review.lower()  #converting upper case to lower cases
    
    #splitting the review in different words
    # it is string before the splitting of the data after splitting it becomes a list
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #list comprehension. FIlter all the words that are not in stopwords "English" list of dictionary
    #steming - removing past or future or othe format of word for example removing loved with love
    review =' '.join(review) #it will join all the strings in a word seperated by space.
    corpus.append(review)

#creating the bag of words model
# we take all different but unique reviews and create 1 column for each word. And then we'll put all this columns in a table where the rows are 1000 review.

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
#X has 1500 words i.e. 1500 columns and 1000 rows

y = dataset.iloc[:,1].values

#using naive bayes
#splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# #feature scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

#fitting Naive bais into training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the result
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

