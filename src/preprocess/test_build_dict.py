from .build_dict import build_dict
from .read_imdb_data import read_imdb_data
from .prepare_imdb_data import prepare_imdb_data
from .preprocess_data import preprocess_data
from .convert_and_pad_data import convert_and_pad_data
import os
import pandas as pd


def test_build_dict():
    imdb_data_dir = '../tests/data/aclImdb'
    print('\n')

    data, labels = read_imdb_data(data_dir=imdb_data_dir)
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels, should_shuffle=False)

    print('raw imdb review data: train_X[0]')
    print(train_X[0])
    print('and its label: train_y[0]')
    print(train_y[0])
    print('\n')

    # removing cached preprocessed data here because
    # os.remove('../tests/cache/sentiment_analysis/preprocessed_data.pkl')
    cache_dir = '../tests/cache/sentiment_analysis'
    train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y,
                                                       cache_dir=cache_dir
                                                       )
    print('processed data: train_X[0]:')
    print(train_X[0])
    print('and its label: train_y[0]')
    print(train_y[0])
    print('\n')

    word_dict = build_dict(train_X)
    print(word_dict)
    train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
    print('after convert_and_pad_data: train_X[0]:')
    print(train_X[0])

    int_to_word = {v: k for k, v in word_dict.items()}

    train_X0_back_in_words = []
    for idx in train_X[0]:
        if idx in int_to_word:
            train_X0_back_in_words.append(int_to_word[idx])
        else:
            train_X0_back_in_words.append(idx)

    print(train_X0_back_in_words)

    data_dir = '../tests/data/pytorch'
    pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
