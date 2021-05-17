import pickle

def _read_pkl(input_file):
    """Reads a tab separated value file."""
    data = pickle.load(open(input_file, 'rb'))
    return data

print(_read_pkl('conll_train.pkl')[0])
