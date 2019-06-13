import torch
import torch.utils.data
import os
import pandas as pd
import torch.optim as optim
from ..train.model import LSTMClassifier
from ..train.train import train


def test_train():
    data_dir = './data/pytorch'

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
    model = LSTMClassifier(32, 100, 5000).to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_sample_dl, 5, optimizer, loss_fn, device)
