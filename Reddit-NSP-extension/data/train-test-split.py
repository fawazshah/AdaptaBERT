import pandas as pd

submissions_df = pd.read_csv('all-data/submissions_preprocessed.tsv', sep='\t')
comments_df = pd.read_csv('all-data/comments_preprocessed.tsv', sep='\t')

# Shuffle datasets
submissions_df = submissions_df.sample(frac=1)
comments_df = comments_df.sample(frac=1)

TRAIN = 0.8
TEST = 0.2

split_point = int(TRAIN*len(submissions_df))

submissions_train_df = submissions_df.iloc[:split_point].copy()
submissions_test_df = submissions_df.iloc[split_point:].copy()

print(f"Size of submissions training set: {len(submissions_train_df)}")
print(f"Size of submissions test set: {len(submissions_test_df)}")

submissions_train_df.to_csv('train-test-split/submissions_train.tsv', sep='\t', index=False)
submissions_test_df.to_csv('train-test-split/submissions_test.tsv', sep='\t', index=False)

split_point = int(TRAIN*len(comments_df))

comments_train_df = comments_df.iloc[:split_point].copy()
comments_test_df = comments_df.iloc[split_point:].copy()

print(f"Size of comments training set: {len(comments_train_df)}")
print(f"Size of comments test set: {len(comments_test_df)}")

comments_train_df.to_csv('train-test-split/comments_train.tsv', sep='\t', index=False)
comments_test_df.to_csv('train-test-split/comments_test.tsv', sep='\t', index=False)