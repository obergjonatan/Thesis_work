# -*- coding: utf-8 -*-
from tarfile import ENCODING
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import csv
import math
from pandas.core import base
import tweepy
import random
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import preprocessor


# Autherize twitter api


def twitter_auth(cons_key, cons_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(cons_key, cons_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

# Get twitter data from given path


def get_covid_tweet_data(path, lang):
    data = pd.read_csv(path, header=0, sep='\t')
    data = data.fillna(0)
    if lang == 'english':
        data = data[data.lang == 'en']
        data = data.reset_index(drop=True)
    return data['tweet_id']


# Takes sample_size number of samples from the given data and returns list of new data
def sample_tweets(data, sample_size):
    nmbr_tweets = len(data.index)
    sampled_tweets = []
    previously_sampled = []
    for j in range(sample_size):
        random_index = random.randrange(1, nmbr_tweets)
        while str(random_index) in previously_sampled:
            random_index = random.randrange(1, nmbr_tweets)
        sampled_tweets.append(str(data.loc[random_index]))
        previously_sampled.append(str(random_index))
    return sampled_tweets

# Takes list of tweetids and returns a list of texts related to those tweetids

# Takes list of tweet_ids and an authorized twitter api and returns a list containing the texts from the given tweet_ids.


def get_tweet_texts(tweet_ids):
    tweet_texts = []
    for i in range(math.ceil(len(tweet_ids)/100)):
        lower_limit = i*100
        upper_limit = min(len(tweet_ids)-1, ((i+1)*100-1))
        sliced_ids = tweet_ids[lower_limit:upper_limit]
        tweet_data = api.lookup_statuses(sliced_ids)
        for tweet in tweet_data:
            tweet_texts.append(tweet.text)
    return tweet_texts

# Gathers tweets from list of tweetids and then does sentiment analysis on tweets and calculates the average sentiment of all tweets


def get_average_sentiment(texts):
    total_sentiment = 0
    for text in texts:
        pre_processed_text = pre_process_text(text)
        total_sentiment += sia.polarity_scores(pre_processed_text)["compound"]
    return total_sentiment/len(texts)

# Returns the calculated VADER sentiment for the given string.


def pre_process_text(text: str):
    cleaned_text = preprocessor.clean(text)
    tokenized_text = nltk.word_tokenize(cleaned_text)
    preprocessed_text = [
        w for w in tokenized_text if not w.lower() in stopwords]
    full_text = ' '.join(preprocessed_text)
    return full_text


# Takes the dictionary of daily sentiment values and creates a csv file and fills it with the data.


def export_sentiment_data(daily_sentiment):
    csv_file = open("daily_sentiment" + str(start_date) + "-" +
                    str(start_date + timedelta(days=days_to_check-1)) + ".csv",
                    "w",
                    newline='')
    writer = csv.writer(csv_file)
    header = ['date', 'sentiment']
    writer.writerow(header)
    for key, value in daily_sentiment:
        writer.writerow([key, value])
    csv_file.close()


cons_key = 'GWcaRFYChJVAzzhs5iIA3SgQ5'
cons_secret = 'W1Imhgxr9jGDLHi6GUFASf4pcACaBKqNVt7BhZFetaGRlCQLBd'

access_token = '1450852367616856077-g1OpvMn6sWmRTfK9bQtPZAybHHkJ2F'
access_secret = 'qPsEW3VLWrrV9kmFTXy2cCxJteLnvxNQvStOIlGuyGSxE'

base_path = 'Covid_tweet_data/covid19_twitter/dailies/'

nltk.download([
    "names",
    "stopwords",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
print(stopwords)
sia = SentimentIntensityAnalyzer()

start_date = dt.date(2021, 1, 20)
days_to_check = 6
date = start_date
daily_sentiment = {}
sample_size = 1000

api = twitter_auth(cons_key, cons_secret, access_token, access_secret)

for i in range(days_to_check):
    path = base_path + str(date) + '/' + str(date) + '_clean-dataset.tsv'
    if (date >= dt.date(2020, 7, 26)):
        data = get_covid_tweet_data(path, 'english')
    else:
        data = get_covid_tweet_data(path, 'all')
    sampleids = sample_tweets(data, sample_size)
    texts = get_tweet_texts(sampleids)
    average_sentiment = get_average_sentiment(texts)
    daily_sentiment[str(date)] = average_sentiment
    date = date + timedelta(days=1)
    print('Analyzed %d/%d days' % ((i+1), int(days_to_check)))
export_sentiment_data(daily_sentiment.items())
