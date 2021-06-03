import matplotlib.pyplot as plt
import os
import pandas as pd
import re

data_dir = 'data/train-test-split'

# SUBMISSIONS

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
num_sentences_submissions.sort_values(inplace=True)
num_sentences_submissions.reset_index(drop=True, inplace=True)
num_sentences_submissions.plot(fontsize=14, figsize=(10, 7))
plt.xlabel('article number', fontsize=16)
plt.ylabel('no. sentences', fontsize=16)
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
num_sentences_comments.sort_values(inplace=True)
num_sentences_comments.reset_index(drop=True, inplace=True)
num_sentences_comments.plot(fontsize=14, figsize=(10, 7))
plt.xlabel('comment number', fontsize=16)
plt.ylabel('no. sentences', fontsize=16)
plt.show()