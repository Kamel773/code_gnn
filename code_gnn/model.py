import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d, functional as F, Sequential, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool


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
        self.bn = BatchNorm1d(hidden_features)

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
        x = global_add_pool(x, batch)  # Readout from each graph in the batch
        x = self.lin1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    @staticmethod
    def MLP(in_features, hidden):
        return Sequential(
            Linear(in_features, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
        )


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