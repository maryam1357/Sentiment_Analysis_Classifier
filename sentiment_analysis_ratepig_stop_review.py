#Python 2.7
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt #Only plotting will not work
from collections import defaultdict
import nltk
import re


def our_tokenizer(line, stops=None, stemmer=None):
    vocab = defaultdict(int)
    word_list = [word.decode("utf8", errors='ignore') for word in line.lower().split()]
    if stops:
      word_list = [word for word in word_list if (word not in stops)]
    if stemmer:
      wnl = WordNetLemmatizer()
      word_list = [wnl.lemmatize(word) for word in word_list]
    for word in word_list:
      vocab[word] += 1
    return vocab


def str_join(df, sep, *cols):
    return reduce(lambda x, y: x.astype(str).str.cat(y.astype(str), sep=sep), [df[col] for col in cols])

def vote(vocab,pol_words):
    vote_count = 0
    for word in vocab.keys():
        if word in pol_words.keys():
            polarity = pol_words[word]
        else:
            polarity = 0
        value = vocab[word]
        vote_count += polarity*value
    return vote_count


def sentiment_analysis (vocab,pol_words):
    result = vote(vocab,pol_words)
    if result>0:
        return("Positive review")
    elif result == 0:
        return("Neutral review")
    else:
        return("Negative review")


def polarity_column(review,pol_words, stop_words=None):
    vocab_review = our_tokenizer(review, stops=stop_words, stemmer=WordNetLemmatizer())
    polarity_results = sentiment_analysis(vocab_review, pol_words)
    return polarity_results


def vote_column(review,pol_words,stop_words=None):
    vocab_review = our_tokenizer(review, stops=stop_words, stemmer=WordNetLemmatizer())
    vote_results = vote(vocab_review,pol_words)
    return vote_results

def new_df(df2,column):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    df2['polarity']=df2[column].map(lambda x: polarity_column(x,pol_words,stopwords))
    df2['vote']=df2[column].map(lambda y: vote_column(y,pol_words,stopwords))
    df2.to_csv("rating_sentiment3.csv")



# Load polarity words, adapted from http://sentiwordnet.isti.cnr.it/
f = open('polarity_words_uniq.csv','r')
i = 0
pol_words = {} #Dictionary of polarity words: {pol_word1:polarity1, pol_word2:polarity2,...}
next(f) #Skip header
for line in f:
    line = line.strip()
    line = line.split(',')
    pol_words[line[0]] = np.sign(float(line[1])) #+1 is for positive words, -1 is for negative words


reviews = pd.read_csv('reviews.csv')
products = pd.read_csv('products.csv')
join_table= pd.merge(reviews,products, on='Product ID', how='outer')

reviews_nonull = join_table[pd.notnull(join_table['content'])]
df_content_rating=pd.DataFrame(reviews_nonull[['content','Rating_x','Sitename','Category','Brand']])
new_df(df_content_rating,'content')
