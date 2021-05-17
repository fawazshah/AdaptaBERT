import pandas as pd

submissions_df = pd.read_csv('all-data/submissions_preprocessed.tsv', sep='\t')
comments_df = pd.read_csv('all-data/comments_preprocessed.tsv', sep='\t')

TRAIN = 0.7
VAL = 0.1
TEST = 0.2

split_point_1 = int(TRAIN*len(submissions_df))
split_point_2 = int((TRAIN+VAL)*len(submissions_df))

submissions_train_df = submissions_df.iloc[:split_point_1].copy()
submissions_val_df = submissions_df.iloc[split_point_1:split_point_2].copy()
submissions_test_df = submissions_df.iloc[split_point_2:].copy()

print(f"Size of submissions training set: {len(submissions_train_df)}")
print(f"Size of submissions validation set: {len(submissions_val_df)}")
print(f"Size of submissions test set: {len(submissions_test_df)}")

submissions_train_df.to_csv('train-val-test/submissions_train.tsv', sep='\t', index=False)
submissions_val_df.to_csv('train-val-test/submissions_val.tsv', sep='\t', index=False)
submissions_test_df.to_csv('train-val-test/submissions_test.tsv', sep='\t', index=False)

split_point_1 = int(TRAIN*len(comments_df))
split_point_2 = int((TRAIN+VAL)*len(comments_df))

comments_train_df = comments_df.iloc[:split_point_1].copy()
comments_val_df = comments_df.iloc[split_point_1:split_point_2].copy()
comments_test_df = comments_df.iloc[split_point_2:].copy()

print(f"Size of comments training set: {len(comments_train_df)}")
print(f"Size of comments validation set: {len(comments_val_df)}")
print(f"Size of comments test set: {len(comments_test_df)}")

comments_train_df.to_csv('train-val-test/comments_train.tsv', sep='\t', index=False)
comments_val_df.to_csv('train-val-test/comments_val.tsv', sep='\t', index=False)
comments_test_df.to_csv('train-val-test/comments_test.tsv', sep='\t', index=False)