import torch
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN, global_add_pool
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
    # Referenced https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial5/Aggregation%20Tutorial.ipynb
    # for readout function
    def __init__(self):
        super().__init__()
        # With out_channels argument, it has a linear layer to convert node embedding to prediction
        self.gin = GIN(in_channels=3, hidden_channels=32, num_layers=5)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        h = self.gin(x, edge_index)
        h = global_add_pool(h, batch)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h.squeeze()


def main():
    dataset = TUDataset(root='D:/datasets/TUDataset', name='PROTEINS')

    device = torch.device('cuda')

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GINModel().to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    log_every = 25
    average_loss = []
    for i in range(250):
        epoch_loss = []
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        for data in loader:
            x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            label = data.y.float().to(device)
            out = model(x, edge_index, batch)
            loss = loss_fn(out, label)
            epoch_loss.append(loss.detach().item())
            loss.backward()
            optimizer.step()
        average_loss_epoch = sum(epoch_loss)/len(epoch_loss)
        average_loss.append(average_loss_epoch)
        if i % log_every == 0:
            print('Epoch:', i, 'Average batch loss:', average_loss_epoch)
            plt.plot(average_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Average loss')
            plt.title(f'Loss at epoch {i}')
            plt.show()


if __name__ == '__main__':
    main()
