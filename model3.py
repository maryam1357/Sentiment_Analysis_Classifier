from __future__ import division
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
from nltk.classify.util import apply_features
import nltk

processed_record = 0
words_features = None

def getFeatureVector(line):
    stops = set(nltk.corpus.stopwords.words('english'))
    stemmer=WordNetLemmatizer()
    featureVector = []
    word_list = [word.decode("utf8", errors='ignore') for word in line.lower().split()]
    if stops:
      word_list = [word for word in word_list if (word not in stops)]
    if stemmer:
      wnl = WordNetLemmatizer()
      word_list = [wnl.lemmatize(word) for word in word_list]
    for word in word_list:
        if(word in stops):
            continue
        else:
            featureVector.append(word)
    return featureVector

def featureExtraction(df):
    #Here I am reading the surveys one by one and process it
    dataset = []
    for index, row in df.iterrows():
        survey = getFeatureVector(row['Content'])
        sentiment = row['polarity']
        dataset.append((survey,sentiment))
    return dataset


def bag_of_words(dataset):
    all_words = []
    for (text, sentiment) in dataset:
        all_words.extend(text)
    return all_words

def word_frequency(all_words):

    words_freq = nltk.FreqDist(all_words)
    return words_freq


def get_words_features(words_freq):

# This line calculates the frequency distrubtion of all words in dataset

    return words_freq.keys()

    # This prints out the list of all distinct words in the text in order
    # of their number of occurrences.



def find_word_feature(review):
    global processed_record
    global words_features
    processed_record += 1
    print '{},{}'.format('processed record', processed_record)
    word_feature = {}
    for w in words_features:
        word_feature[w] = (w in review)
    return word_feature

def naive_bayes(training_set,test_set):

    # build the trained set and save it in naive_classifier.p
    naive_classifier = 'naive_classifier.p'
    if not os.path.exists(naive_classifier):
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        fileobject = open(naive_classifier, 'wb')
        pickle.dump(classifier, fileobject)
        fileobject.close()
    # load the ctrained data set in the previous step.
    fileobject = open(naive_classifier, 'rb')
    classifier = pickle.load(fileobject)
    fileobject.close()

#######
    print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
    classifier.show_most_informative_features(15)


df = pd.read_csv("rating_sentiment3.csv")
dataset = featureExtraction(df)
all_words = bag_of_words(dataset)
word_freq = word_frequency(all_words)
words_features = get_words_features(word_freq)
processed_record = 0
training_set = apply_features(find_word_feature, dataset[:-750])
test_set = apply_features(find_word_feature, dataset[-750:])
naive_bayes(training_set,test_set)
