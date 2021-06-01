import os
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

print(f'Number of Twitter samples: {len(all_data)}')

num_sentences = []

for elem in all_data:
    text = elem[0]
    text = ' '.join(text)
    # Split text into sentences
    split_regex = re.compile(r'[.|!|?|...]')
    sentences = [t.strip() for t in split_regex.split(text) if t.strip() != '']
    num_sentences.append(len(sentences))

print(f'Avg. sentence length: {sum(num_sentences) / len(num_sentences)}')