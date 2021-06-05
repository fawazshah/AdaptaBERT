import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import re

data_dir = 'data/'

def _read_pkl(input_file):
    """Reads a tab separated value file."""
    data = pickle.load(open(input_file, 'rb'))
    return data

# ARTICLES

train_data = _read_pkl(os.path.join(data_dir, "conll_train.pkl"))
test_data = _read_pkl(os.path.join(data_dir, "conll_test.pkl"))
all_data = train_data + test_data

num_sentences_articles = []

for elem in all_data:
    text = elem[0]
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_articles.append(len(sentences))

print('Twitter num sentences information:')

num_sentences_articles = pd.Series(num_sentences_articles)
print(num_sentences_articles.describe())

num_sentences_articles_dist = num_sentences_articles.value_counts()
num_sentences_articles_dist.sort_index(inplace=True)
num_sentences_articles_dist = num_sentences_articles_dist.to_frame('frequency')
# Remove small outliers for easier visualisation
num_sentences_articles_dist.drop(num_sentences_articles_dist[num_sentences_articles_dist.index > 6].index, inplace=True)
# Make index 'continuous'
num_sentences_articles_dist = num_sentences_articles_dist.reindex(range(num_sentences_articles_dist.index[-1]+1)).fillna(0)
num_sentences_articles_dist.reset_index(inplace=True)

num_sentences_articles_dist.plot(x='index', y='frequency', kind='bar', fontsize=14, figsize=(10, 7), legend=None)
plt.xlabel('no. sentences in text', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.xticks(np.arange(1, 7), rotation=0)
plt.title('articles', fontsize=24)
plt.show()

# TWEETS

train_data = _read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl"))
test_data = _read_pkl(os.path.join(data_dir, "sep_twitter_test.pkl"))
all_data = train_data + test_data

num_sentences_tweets = []

for elem in all_data:
    text = elem[0]
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_tweets.append(len(sentences))

print('Twitter num sentences information:')

num_sentences_tweets = pd.Series(num_sentences_tweets)
print(num_sentences_tweets.describe())

num_sentences_tweets_dist = num_sentences_tweets.value_counts()
num_sentences_tweets_dist.sort_index(inplace=True)
num_sentences_tweets_dist = num_sentences_tweets_dist.to_frame('frequency')
# Remove small outliers for easier visualisation
num_sentences_tweets_dist.drop(num_sentences_tweets_dist[num_sentences_tweets_dist.index > 15].index, inplace=True)
# Make index 'continuous'
num_sentences_tweets_dist = num_sentences_tweets_dist.reindex(range(num_sentences_tweets_dist.index[-1]+1)).fillna(0)
num_sentences_tweets_dist.reset_index(inplace=True)

num_sentences_tweets_dist.plot(x='index', y='frequency', kind='bar', fontsize=14, figsize=(10, 7), legend=None)
plt.xlabel('no. sentences in text', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.xticks(np.arange(1, 9), rotation=0)
plt.title('tweets', fontsize=24)
plt.show()