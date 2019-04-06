#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[2]:


# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2


# In[3]:


def ngram_vectorize_train(df1 = pd.read_csv('./data/train_lyrics_1000.csv'), df2 = pd.read_csv('./data/valid_lyrics_200.csv')):
    df1 = df1[['lyrics','mood']]
    df2 = df2[['lyrics','mood']]
    df1['lyrics'] = df1['lyrics'].apply(lambda x : x.lower())
    df2['lyrics'] = df2['lyrics'].apply(lambda y : y.lower())
    df1.loc[:,'lyrics']= df1.loc[:,'lyrics'].apply(lambda x : re.sub('[^a-zA-z0-9\s]','',x))
    df2.loc[:,'lyrics']= df2.loc[:,'lyrics'].apply(lambda x : re.sub('[^a-zA-z0-9\s]','',x))
    df1['lyrics'] = df1['lyrics'].apply(lambda x : re.sub('[\n]',' ',x))
    df2['lyrics'] = df2['lyrics'].apply(lambda x : re.sub('[\n]',' ',x))
    df1['lyrics'] = df1['lyrics'].apply(lambda x : x.lstrip(' '))
    df2['lyrics'] = df2['lyrics'].apply(lambda x : x.lstrip(' '))

    #Removing StopWords
    stop = stopwords.words('english')
    df1['lyrics'] = df1['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df2['lyrics'] = df2['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #Lemmatize
    lem = WordNetLemmatizer()
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word) for word in x.split()]))
    df2['lyrics'] = df2['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word) for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='a') for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='n') for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='v') for word in x.split()]))
    df2['lyrics'] = df2['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='a') for word in x.split()]))
    df2['lyrics'] = df2['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='n') for word in x.split()]))
    df2['lyrics'] = df2['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='v') for word in x.split()]))

    # Label encoding
    le = LabelEncoder()
    Y_train = le.fit_transform(df1['mood'])
    Y_test = le.fit_transform(df2['mood'])
    
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    
     # Learn vocabulary from training texts and vectorize training texts.
    X_train = vectorizer.fit_transform(df1['lyrics'])

    # Vectorize validation texts.
    X_test = vectorizer.transform(df2['lyrics'])

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, X_train.shape[1]))
    selector.fit(X_train, Y_train)
    X_train = selector.transform(X_train).astype('float32').toarray()
    X_test = selector.transform(X_test).astype('float32').toarray()
    print(X_test) 
    return X_train, Y_train, X_test, Y_test


# In[4]:


def ngram_vectorize_pred(df3):
    df1 = pd.read_csv('./data/train_lyrics_1000.csv')
    df1 = df1[['lyrics','mood']]
    #df3 = df3['lyrics']
    df1['lyrics'] = df1['lyrics'].apply(lambda x : x.lower())
    df3['lyrics'] = df3['lyrics'].apply(lambda y : y.lower())
    df1.loc[:,'lyrics']= df1.loc[:,'lyrics'].apply(lambda x : re.sub('[^a-zA-z0-9\s]','',x))
    df3.loc[:,'lyrics']= df3.loc[:,'lyrics'].apply(lambda x : re.sub('[^a-zA-z0-9\s]','',x))
    df1['lyrics'] = df1['lyrics'].apply(lambda x : re.sub('[\n]',' ',x))
    df3['lyrics'] = df3['lyrics'].apply(lambda x : re.sub('[\n]',' ',x))
    df1['lyrics'] = df1['lyrics'].apply(lambda x : x.lstrip(' '))
    df3['lyrics'] = df3['lyrics'].apply(lambda x : x.lstrip(' '))

    #Removing StopWords
    stop = stopwords.words('english')
    df1['lyrics'] = df1['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df3['lyrics'] = df3['lyrics'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #Lemmatize
    lem = WordNetLemmatizer()
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word) for word in x.split()]))
    df3['lyrics'] = df3['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word) for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='a') for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='n') for word in x.split()]))
    df1['lyrics'] = df1['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='v') for word in x.split()]))
    df3['lyrics'] = df3['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='a') for word in x.split()]))
    df3['lyrics'] = df3['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='n') for word in x.split()]))
    df3['lyrics'] = df3['lyrics'].apply(lambda x :" ".join([lem.lemmatize(word, pos='v') for word in x.split()]))

    # Label encoding
    le = LabelEncoder()
    Y_train = le.fit_transform(df1['mood'])
    
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    X_train = vectorizer.fit_transform(df1['lyrics'])

    # Vectorize validation texts.
    X_test = vectorizer.transform(df3['lyrics'])

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, X_train.shape[1]))
    selector.fit(X_train, Y_train)
    X_train = selector.transform(X_train).astype('float32').toarray()
    X_pred = selector.transform(X_test).astype('float32').toarray()
    
    return X_pred


# In[ ]:




