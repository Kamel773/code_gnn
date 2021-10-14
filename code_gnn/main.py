import os
import random
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MyDGLDataset
from model import DevignModel
from trainer import train


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dataset = MyDGLDataset(embed_type='codebert', raw_dir='data', save_dir='data', verbose=True)
    return
    device = torch.device('cuda')

    input_dim = 169
    graph_embed_size = 200
    num_steps = 8
    model = DevignModel(input_dim=input_dim, output_dim=graph_embed_size, n_etypes=dataset.max_etypes, num_steps=num_steps).to(device)
    tb_dir = 'runs/' + '-'.join(str(s) for s in (dataset.name, type(model).__name__, input_dim, graph_embed_size, num_steps))
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)
    tb = SummaryWriter(tb_dir)

    train(dataset, model, device, tb)


if __name__ == '__main__':
    main()
