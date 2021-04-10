import pandas as pd
import numpy as np

import re
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

import pickle

def preprocess_and_tokenize(data):    

    #remove html markup
    data = re.sub("(<.*?>)", "", data)

    #remove urls
    data = re.sub(r'http\S+', '', data)
    
    #remove hashtags and @names
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)

    #remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)
    
    #remove whitespace
    data = data.strip()
    
    # tokenization with nltk
    data = word_tokenize(data)
    
    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
        
    return stem_data


df_train = pd.read_csv('./emotions_data/data_train.csv')
df_test = pd.read_csv('./emotions_data/data_test.csv')

X_train = df_train.Text
X_test = df_test.Text

y_train = df_train.Emotion
y_test = df_test.Emotion

class_names = ['joy', 'sadness', 'anger', 'neutral', 'fear']
data = pd.concat([df_train, df_test])

# TFIDF, unigrams and bigrams
vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))

# fit on our complete corpus
vect.fit_transform(data.Text)

# transform testing and training datasets to vectors
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)

svc = LinearSVC(tol=1e-05)
svc.fit(X_train_vect, y_train)

svm_pred = svc.predict(X_test_vect)

svm_model = Pipeline([('tfidf', vect),('clf', svc),])

emotions_clf_filename = 'tfidf_svm.sav'
pickle.dump(svm_model, open(emotions_clf_filename, 'wb'))

emotions_clf = pickle.load(open('tfidf_svm.sav', 'rb'))

message = 'delivery was hour late and my pizza is cold!' 
# print(emotions_clf.predict([message]))