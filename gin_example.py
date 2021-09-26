import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss, Sequential, Linear, ReLU, BatchNorm1d
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt
import itertools

from torch_scatter import scatter_mean


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

    def forward(self, x, edge_index, batch):
        layer_hs = []
        node_h = x
        for conv in self.convs:
            node_h = conv(node_h, edge_index)
            layer_hs.append(node_h)
        h_concat = self.jump(layer_hs)  # JumpingKnowledge('cat') just concatenates node features from all layers
        graph_h = global_mean_pool(h_concat, batch)  # Readout from each graph in the batch
        graph_h = F.relu(self.lin1(graph_h))
        graph_h = F.dropout(graph_h, p=0.5, training=self.training)
        graph_h = self.lin2(graph_h)
        return graph_h

    @staticmethod
    def MLP(in_features, hidden):
        return Sequential(
            Linear(in_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BatchNorm1d(hidden),
        )


def measure_performance(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        all_correct = 0
        epoch_size = 0
        for data in test_loader:
            x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            label = data.y.float().to(device)
            out = model(x, edge_index, batch)
            pred = torch.argmax(out, dim=1)
            correct = pred == label
            all_correct += correct.sum().item()
            epoch_size += len(label)
        acc = all_correct / epoch_size
        return acc


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
        h_cat_pool = global_mean_pool(h_cat, batch)

        # concat(agg(h_i) for h_i in h)
        h1_pool = global_mean_pool(h1, batch)
        h2_pool = global_mean_pool(h2, batch)
        h3_pool = global_mean_pool(h3, batch)
        h_pool_cat = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)

        assert torch.equal(h_cat_pool, h_pool_cat), 'concat/agg are not commutative??'


def main():
    np.random.seed(0)
    torch.random.manual_seed(0)
    dataset = TUDataset(root='D:/datasets/TUDataset', name='PROTEINS')

    device = torch.device('cuda')

    n_train = int(len(dataset)*0.9)
    splits = [n_train, len(dataset) - n_train]
    print('Splits:', splits[0], 'train', splits[1], 'test')
    train_set, test_set = random_split(dataset, splits)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    data = next(iter(train_loader))

    model = MyGIN(dataset, num_layers=5, hidden_features=64).to(device)
    model.reset_parameters()
    print(list((p.shape for p in model.parameters())))
    # I'm finding that the GIN model does not handle pyg minibatches
    # model = GIN(in_channels=3, hidden_channels=64, num_layers=5, out_channels=2, jk='cat').to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    log_every = 25
    tb = SummaryWriter()
    for i in range(400):
        epoch_loss = []
        all_correct = 0
        epoch_size = 0
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            label = data.y.to(device)
            out = model(x, edge_index, batch)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss.append(loss.detach().item())
            with torch.no_grad():
                pred = torch.argmax(out, dim=1)
                correct = pred == label
                all_correct += correct.float().sum().item()
                epoch_size += len(label)
        train_acc = all_correct / epoch_size
        train_loss = sum(epoch_loss)/len(epoch_loss)
        tb.add_scalar('acc/train', train_acc, i)
        tb.add_scalar('loss/train', train_loss, i)
        if i % log_every == 0:
            acc = measure_performance(model, device, test_loader)
            tb.add_scalar('acc/test', acc, i)
            # print('Epoch:', i, 'Train Accuracy:', train_acc, 'Train loss:', train_loss, 'Test Accuracy:', acc)
            print('Epoch:', i, 'Train loss:', train_loss, 'Test Accuracy:', acc)


if __name__ == '__main__':
    main()
