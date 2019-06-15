import torch
import torch.utils.data
import os
import pandas as pd
import torch.optim as optim
from train.model import LSTMClassifier
from train.train import train

import pickle

here = os.path.dirname(os.path.realpath(__file__))


def test_train():
    data_dir = os.path.join(here, './data/pytorch')

    # Read in only the first 250 rows
    train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

    # Turn the input pandas dataframe into tensors
    train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
    train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

    # Build the dataset
    train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
    # Build the dataloader
    train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)

    print(train_sample_y)
    print(train_sample_X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = 32
    hidden_dim = 100
    vocab_size = 5000

    model = LSTMClassifier(embedding_dim=embedding_dim,
                           hidden_dim=hidden_dim,
                           vocab_size=vocab_size).to(device)
    print(model)

    with open(os.path.join(data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_sample_dl, 5, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_dir = os.path.join(here , './data/modelDir')

    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'vocab_size': vocab_size,
        }
        torch.save(model_info, f)

    # Save the word_dict
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
