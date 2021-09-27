import os
import random
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss, Sequential, Linear, ReLU, BatchNorm1d, NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt
import itertools

from pytorch_geometric.benchmark.kernel.gin import GINWithJK


def draw_9_plots(dataset):
    fig, axes = plt.subplots(nrows=3, ncols=3)
    ax = axes.flatten()
    dataset_sample = list(itertools.islice((d for d in dataset if d.y[0].item() == 0), 4)) + list(
        itertools.islice((d for d in dataset if d.y[0].item() == 1), 5))
    for i in range(3*3):
        data = dataset_sample[i]
        nx.draw_networkx(to_networkx(data, to_undirected=True), with_labels=False, ax=ax[i], node_size=10, node_color=torch.argmax(data.x, dim=1), cmap='Set1')
        ax[i].set_axis_off()
        ax[i].set_title(f'Enzyme: {data.y[0].item()}')
    fig.suptitle('9 proteins')
    plt.show()


class MyGIN(nn.Module):
    """
    Adapted from pytorch_geometric/benchmark/kernel/gin.py

    pyg implements the node embeddings in GINConv but the readout is pretty customizable, so it's in this module.
    The basic version in basic_gnn.py does not handle minibatches, so I adapted this implementation instead.
    The benchmark version uses mean pooling though the paper uses sum pooling, so I replaced it.
    """
    def __init__(self, dataset, num_layers, hidden_features):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(MyGIN.MLP(dataset.num_node_features, hidden_features), train_eps=True))
        for i in range(num_layers-1):
            self.convs.append(GINConv(MyGIN.MLP(hidden_features, hidden_features), train_eps=True))
        self.jump = JumpingKnowledge('cat')
        self.lin1 = Linear(num_layers * hidden_features, hidden_features)
        self.lin2 = Linear(hidden_features, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        x = self.jump(xs)  # JumpingKnowledge('cat') just concatenates node features from all layers
        x = global_mean_pool(x, batch)  # Readout from each graph in the batch
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    @staticmethod
    def MLP(in_features, hidden):
        return Sequential(
            Linear(in_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BatchNorm1d(hidden),
        )


def measure_performance(model, device, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        all_correct = 0
        epoch_size = 0
        losses = []
        for data in test_loader:
            data = data.to(device)
            label = data.y.to(device)
            out = model(data)
            loss = loss_fn(out, label.view(-1))
            losses.append(loss.item() * len(out))
            pred = torch.argmax(out, dim=1)
            correct = pred == label
            all_correct += correct.sum().item()
            epoch_size += len(label)
        acc = all_correct / epoch_size
        return acc, sum(losses) / len(test_loader.dataset)


def test_agg_concat():
    """
    pyg's GIN implementation uses concat(agg(h)) while the GIN paper uses sum(agg(h)).
    I guess the implementation assumes they're equal.
    I was having suspicion that this is not correct, so this function calculates both and asserts they're equal.
    """

    dataset = TUDataset(root='D:/datasets/TUDataset', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        x, batch = data.x, data.batch
        h1 = torch.randn((len(x), 64))
        h2 = torch.randn((len(x), 64))
        h3 = torch.randn((len(x), 64))

        # agg(concat(h_i for h_i in h))
        h_cat = torch.cat((h1, h2, h3), dim=1)
        h_cat_pool = global_add_pool(h_cat, batch)

        # concat(agg(h_i) for h_i in h)
        h1_pool = global_add_pool(h1, batch)
        h2_pool = global_add_pool(h2, batch)
        h3_pool = global_add_pool(h3, batch)
        h_pool_cat = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)

        assert torch.equal(h_cat_pool, h_pool_cat), 'concat/agg are not commutative??'


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    dataset = TUDataset(root='D:/datasets/TUDataset', name='PROTEINS')

    device = torch.device('cuda')

    n_train = int(len(dataset)*0.9)
    splits = [n_train, len(dataset) - n_train]
    print('Splits:', splits[0], 'train', splits[1], 'test')
    train_set, test_set = random_split(dataset, splits)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)

    num_layers = 5
    hidden_features = 64
    model = MyGIN(dataset, num_layers=num_layers, hidden_features=hidden_features).to(device)
    # model = GINWithJK(dataset, num_layers, hidden_features).to(device)
    model.reset_parameters()
    print(list((p.shape for p in model.parameters())))
    loss_fn = NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=50)
    log_every = 25
    tb_dir = 'runs/' + '-'.join(str(s) for s in (dataset.name, type(model).__name__, num_layers, hidden_features))
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)
    tb = SummaryWriter(tb_dir)
    for i in range(400):
        epoch_loss = 0
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        model.train()
        for j, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y.view(-1))
            loss.backward()
            epoch_loss += loss.item() * data.num_graphs
            optimizer.step()
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc, _ = measure_performance(model, device, train_loader, loss_fn)
        tb.add_scalar('train_acc', train_acc, i)
        tb.add_scalar('train_loss', train_loss, i)
        test_acc, test_loss = measure_performance(model, device, test_loader, loss_fn)
        tb.add_scalar('test_acc', test_acc, i)
        tb.add_scalar('val_loss', test_loss, i)
        if i % log_every == 0:
            # print('Epoch:', i, 'Train Accuracy:', train_acc, 'Train loss:', train_loss, 'Test Accuracy:', acc)
            print('Epoch:', i, 'Train loss:', train_loss, 'Test Accuracy:', test_acc)
        scheduler.step()



if __name__ == '__main__':
    main()
