import os
import random
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MyCodeDataset
from model import GIN, HeteroGIN
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
    node_metadata, edge_metadata = dataset[0].metadata()
    edge_metadata = set(edge_metadata)
    for data in dataset:
        edge_metadata = edge_metadata.union(data.metadata()[1])
    edge_metadata = list(edge_metadata)
    model = HeteroGIN(dataset, num_layers=num_layers, hidden_features=hidden_features, edge_types=edge_metadata).to(device)
    tb_dir = 'runs/' + '-'.join(str(s) for s in (dataset.name, type(model).__name__, num_layers, hidden_features))
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)
    tb = SummaryWriter(tb_dir)

    train(dataset, model, device, tb)


if __name__ == '__main__':
    main()
