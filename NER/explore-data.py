import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import re

data_dir = 'data/'

def _read_pkl(input_file):
    """Reads a tab separated value file."""
    data = pickle.load(open(input_file, 'rb'))
    return data

train_data = _read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl"))
test_data = _read_pkl(os.path.join(data_dir, "sep_twitter_test.pkl"))
all_data = train_data + test_data

num_sentences = []

for elem in all_data:
    text = elem[0]
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences.append(len(sentences))

print('Twitter num sentences information:')

num_sentences = pd.Series(num_sentences)
print(num_sentences.describe())
num_sentences.sort_values(inplace=True)
num_sentences.reset_index(drop=True, inplace=True)
num_sentences.plot(fontsize=14, figsize=(10, 7))
plt.xlabel('tweet number', fontsize=16)
plt.ylabel('no. sentences', fontsize=16)
plt.show()