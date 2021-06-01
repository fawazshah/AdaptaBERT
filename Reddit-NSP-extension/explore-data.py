import os
import pandas as pd
import re

data_dir = 'data/train-test-split'

submissions_train = pd.read_csv(os.path.join(data_dir, 'submissions_train.tsv'), sep='\t')
submissions_test = pd.read_csv(os.path.join(data_dir, 'submissions_test.tsv'), sep='\t')
all_submissions = pd.concat([submissions_train, submissions_test])

print(f'Number of submissions samples: {len(all_submissions)}')

comments_train = pd.read_csv(os.path.join(data_dir, 'comments_train.tsv'), sep='\t')
comments_test = pd.read_csv(os.path.join(data_dir, 'comments_test.tsv'), sep='\t')
all_comments = pd.concat([comments_train, comments_test])

print(f'Number of comments samples: {len(all_comments)}')

num_sentences_submissions = []

for _, row in all_submissions.iterrows():
    text = row['article body']
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_submissions.append(len(sentences))

print(f'Avg. submission num sentences: {sum(num_sentences_submissions) / len(num_sentences_submissions)}')

num_sentences_comments = []

for _, row in all_comments.iterrows():
    text = row['comment body']
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences_comments.append(len(sentences))

print(f'Avg. comment num sentences: {sum(num_sentences_comments) / len(num_sentences_comments)}')