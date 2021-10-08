import os
import random
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset

from data import MyCodeDataset
from model import MyGIN
from trainer import train


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dataset = MyCodeDataset(root='data', name='devign', embed_type='word2vec')
    # dataset = TUDataset(root='data/TUDataset', name='PROTEINS')

    device = torch.device('cuda')

    num_layers = 5
    hidden_features = 64
    model = MyGIN(dataset, num_layers=num_layers, hidden_features=hidden_features).to(device)
    tb_dir = 'runs/' + '-'.join(str(s) for s in (dataset.name, type(model).__name__, num_layers, hidden_features))
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)
    tb = SummaryWriter(tb_dir)

    train(dataset, model, device, tb)


if __name__ == '__main__':
    main()
