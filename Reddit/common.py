SUBMISSIONS = {
    'train_data': 'submissions_train.tsv',
    'train_data_name': 'articles_train',
    'test_data': 'submissions_test.tsv',
    'test_data_name': 'articles_test',
    'column': 'article body',
}

COMMENTS = {
    'train_data': 'comments_train.tsv',
    'train_data_name': 'comments_train',
    'test_data': 'comments_test.tsv',
    'test_data_name': 'comments_test',
    'column': 'comment body',
}

CDL = {
    'src': SUBMISSIONS,
    'trg': COMMENTS,
}

SRC_PROPORTION = 0.33