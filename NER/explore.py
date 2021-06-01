import pickle
import os

data_dir = 'data/'

def _read_pkl(input_file):
    """Reads a tab separated value file."""
    data = pickle.load(open(input_file, 'rb'))
    return data

train_data = _read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl"))
print(f'Number of training samples: {len(train_data)}')

test_data = _read_pkl(os.path.join(data_dir, "sep_twitter_test.pkl"))
print(f'Number of test samples: {len(test_data)}')