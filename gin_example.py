import torch
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN
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


class GINModel(nn.Module):
    def __init__(self):
        super().__init__()
        # With out_channels argument, it has a linear layer to convert node embedding to prediction
        self.gin = GIN(in_channels=3, hidden_channels=64, num_layers=6, out_channels=1)

    def forward(self, *args, **kwargs):
        return self.gin(*args, **kwargs)


def main():
    dataset = TUDataset(root='D:/datasets/TUDataset', name='PROTEINS')
    # dataset.shuffle()
    # print(dataset)
    # draw_9_plots(dataset)

    device = torch.device('cuda')

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GINModel().to(device)

    for i in range(10):
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        for data in loader:
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            label = data.y.to(device)
            out = model(x, edge_index)
    print('Average loss:', loss)


if __name__ == '__main__':
    main()
