from preprocess.read_imdb_data import read_imdb_data
from preprocess.prepare_imdb_data import prepare_imdb_data
from preprocess.preprocess_data import preprocess_data
from preprocess.build_dict import build_dict
from preprocess.convert_and_pad_data import convert_and_pad_data
from preprocess.preprocess_data import preprocess_data
import pickle
import os

here = os.path.dirname(os.path.realpath(__file__))

def test_predict():

    imdb_data_dir = os.path.join(here, './data/aclImdb')
    print('{}\n'.format(imdb_data_dir))

    data, labels = read_imdb_data(data_dir=imdb_data_dir)
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels, should_shuffle=False)

    print('raw imdb review data: train_X[0]')
    print(train_X[0])
    print('and its label: train_y[0]')
    print(train_y[0])
    print('\n')

    # removing cached preprocessed data here because
    os.remove(os.path.join(here, './cache/preprocessed_data.pkl'))
    cache_dir = os.path.join(here,'./cache')
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
    test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
    print(test_X)
    print(test_X_len)
