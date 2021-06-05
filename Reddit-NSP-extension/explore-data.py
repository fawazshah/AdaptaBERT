import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

data_dir = 'data/train-test-split'

# ARTICLES

submissions_train = pd.read_csv(os.path.join(data_dir, 'submissions_train.tsv'), sep='\t')
submissions_test = pd.read_csv(os.path.join(data_dir, 'submissions_test.tsv'), sep='\t')
all_submissions = pd.concat([submissions_train, submissions_test])

num_sentences_submissions = []

for _, row in all_submissions.iterrows():
    text = row['article body']
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_submissions.append(len(sentences))

print('Articles num sentences information:')

num_sentences_submissions = pd.Series(num_sentences_submissions)
print(num_sentences_submissions.describe())

num_sentences_submissions_dist = num_sentences_submissions.value_counts()
num_sentences_submissions_dist.sort_index(inplace=True)
num_sentences_submissions_dist = num_sentences_submissions_dist.to_frame('frequency')
# Remove small outliers for easier visualisation
num_sentences_submissions_dist.drop(num_sentences_submissions_dist[num_sentences_submissions_dist.index > 100].index, inplace=True)
# Make index 'continuous'
num_sentences_submissions_dist = num_sentences_submissions_dist.reindex(range(num_sentences_submissions_dist.index[-1]+1)).fillna(0)
num_sentences_submissions_dist.reset_index(inplace=True)

num_sentences_submissions_dist.plot(x='index', y='frequency', kind='bar', fontsize=14, figsize=(10, 7), legend=None)
plt.xlabel('no. sentences in text', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.xticks(np.arange(0, 101, 20), rotation=0)
plt.title('articles', fontsize=24)
plt.show()

# COMMENTS

comments_train = pd.read_csv(os.path.join(data_dir, 'comments_train.tsv'), sep='\t')
comments_test = pd.read_csv(os.path.join(data_dir, 'comments_test.tsv'), sep='\t')
all_comments = pd.concat([comments_train, comments_test])

num_sentences_comments = []

for _, row in all_comments.iterrows():
    text = row['comment body']
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_comments.append(len(sentences))

print('Comments num sentences information:')

num_sentences_comments = pd.Series(num_sentences_comments)
print(num_sentences_comments.describe())

num_sentences_comments_dist = num_sentences_comments.value_counts()
num_sentences_comments_dist.sort_index(inplace=True)
num_sentences_comments_dist = num_sentences_comments_dist.to_frame('frequency')
# Remove small outliers for easier visualisation
num_sentences_comments_dist.drop(num_sentences_comments_dist[num_sentences_comments_dist.index > 15].index, inplace=True)
# Make index 'continuous'
num_sentences_comments_dist = num_sentences_comments_dist.reindex(range(num_sentences_comments_dist.index[-1]+1)).fillna(0)
num_sentences_comments_dist.reset_index(inplace=True)

num_sentences_comments_dist.plot(x='index', y='frequency', kind='bar', fontsize=14, figsize=(10, 7), legend=None)
plt.xlabel('no. sentences in text', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.xticks(np.arange(0, 16, 5), rotation=0)
plt.title('comments', fontsize=24)
plt.show()