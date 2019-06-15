import os
import torch
import torch.utils.data
import numpy as np
from serve.utils import convert_and_pad
from serve.predict import model_fn
from serve.model import LSTMClassifier

here = os.path.dirname(os.path.realpath(__file__))


def test_predict():
    test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'
    model_dir = os.path.join(here, './data/modelDir')

    model = model_fn(model_dir)
    print(model)

    data_X, data_len = convert_and_pad(model.word_dict, test_review)
    print('data_X ')
    print(data_X)
    print('data_len')
    print(data_len)

    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    data_pack = np.hstack((data_len, data_X))
    print('data_pack (shape: {})'.format(data_pack.shape))
    print(data_pack)
    data_pack = data_pack.reshape(1, -1)

    print('data_pack reshaped (shape: {})'.format(data_pack.shape))
    print(data_pack)

    data = torch.from_numpy(data_pack)

    print('data (shape: {})'.format(data.shape))
    print(data)

#
# def test_predict():
#     imdb_data_dir = os.path.join(here, './data/aclImdb')
#     print('{}\n'.format(imdb_data_dir))
#
#     data, labels = read_imdb_data(data_dir=imdb_data_dir)
#     train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels, should_shuffle=False)
#
#     print('raw imdb review data: train_X[0]')
#     print(train_X[0])
#     print('and its label: train_y[0]')
#     print(train_y[0])
#     print('\n')
#
#     # removing cached preprocessed data here because
#     os.remove(os.path.join(here, './cache/preprocessed_data.pkl'))
#     cache_dir = os.path.join(here, './cache')
#     train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y,
#                                                        cache_dir=cache_dir
#                                                        )
#     print('processed data: train_X[0]:')
#     print(train_X[0])
#     print('and its label: train_y[0]')
#     print(train_y[0])
#     print('\n')
#
#     word_dict = build_dict(train_X)
#     print(word_dict)
#     train_X, train_X_len = convert_and_pad_data(word_dict, test_X)
#     test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
#
#     print('Transforming Training Data to DataFrame')
#     print('train_y')
#     print(train_y)
#
#     print('train_X_len')
#     print(train_X_len)
#
#     print('train_X')
#     print(train_X)
#
#     print('pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1)')
#     df = pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1)
#     print(df)
#
#     data_dir = os.path.join(here, './data/pytorch')
#     df.to_csv(os.path.join(data_dir, 'train_sample.csv'), header=False, index=False)
#
#     train_sample = pd.read_csv(os.path.join(data_dir, 'train_sample.csv'), header=None, names=None, nrows=250)
#     print('train_sample')
#     print(train_sample)
#
#     # Turn the input pandas dataframe into tensors
#     train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
#     train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()
#
#     print('train_sample_y (shape: {})'.format(train_sample_y.shape))
#     print(train_sample_y)
#
#     print('train_sample_X (shape: {})'.format(train_sample_X.shape))
#     print(train_sample_X)
#
#     # Build the dataset
#     train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
#
#     print('train_sample_ds')
#     print(train_sample_ds)
#
#     # Build the dataloader
#     train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
#
#     test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)
#
#     print('test_X')
#     print(test_X)
#
#     print('test_X.values (shape: {})'.format(test_X.values.shape))
#     print(test_X.values)
#
#     split_array = np.array_split(test_X.values, int(test_X.values.shape[0] / float(512) + 1))
#     print('split_array (shape: {})'.format(split_array.shape))
#     print((split_array))
