import torch
from torch import nn, Tensor
from torch.nn import Linear, BatchNorm1d, functional as F, Sequential, ReLU
from torch.nn import Parameter as Param
from torch_geometric.nn.inits import uniform
from torch_sparse import SparseTensor, matmul

from torch_geometric.typing import OptTensor, Adj

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool, GINEConv, GatedGraphConv, MessagePassing, \
    HeteroConv


class GIN(nn.Module):
    """
    Adapted from pytorch_geometric/benchmark/kernel/gin.py

    pyg implements the node embeddings in GINConv but the readout is pretty customizable, so it's in this module.
    The basic version in basic_gnn.py does not handle minibatches, so I adapted this implementation instead.
    The benchmark version uses mean pooling though the paper uses sum pooling, so I replaced it.
    """
    def __init__(self, dataset, num_layers, hidden_features):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(GIN.MLP(dataset.num_node_features, hidden_features), train_eps=True))
        for i in range(num_layers-1):
            self.convs.append(GINConv(GIN.MLP(hidden_features, hidden_features), train_eps=True))
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


class HeteroGIN(nn.Module):
    """
    Adapted from pytorch_geometric/benchmark/kernel/gin.py

    pyg implements the node embeddings in GINConv but the readout is pretty customizable, so it's in this module.
    The basic version in basic_gnn.py does not handle minibatches, so I adapted this implementation instead.
    The benchmark version uses mean pooling though the paper uses sum pooling, so I replaced it.
    """
    def __init__(self, dataset, num_layers, hidden_features, edge_types):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(HeteroConv({
            t: GINConv(GIN.MLP(dataset.num_node_features, hidden_features), train_eps=True) for t in edge_types
        }))
        for i in range(num_layers-1):
            self.convs.append(HeteroConv({
                t: GINConv(GIN.MLP(hidden_features, hidden_features), train_eps=True) for t in edge_types
            }))
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
        x_dict, edge_index_dict, batch_dict = data.x_dict, data.edge_index_dict, data.batch_dict
        xs = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            xs.append(x_dict['node'])
        x = self.jump(xs)  # JumpingKnowledge('cat') just concatenates node features from all layers
        x = global_add_pool(x, batch_dict['node'])  # Readout from each graph in the batch
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


class HeteroGatedGraphConv(MessagePassing):
    def __init__(self, out_channels: int, num_layers: int, edge_types, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super(HeteroGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = {
            t: Param(Tensor(num_layers, out_channels, out_channels)) for t in edge_types
        }
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x_dict: Tensor, edge_index_dict: Adj) -> Tensor:
        """"""
        if x_dict['node'].size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x_dict['node'].size(-1) < self.out_channels:
            zero = x_dict['node'].new_zeros(x_dict['node'].size(0), self.out_channels - x_dict['node'].size(-1))
            x = torch.cat([x_dict['node'], zero], dim=1)

        for i in range(self.num_layers):
            ms = []
            for edge_type, edge_index in edge_index_dict.items():
                m = torch.matmul(x, self.weight[edge_type][i])
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                m = self.propagate(edge_index, x=m)
                ms.append(m)
            m = torch.sum(torch.tensor(ms), dim=1)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)

class HeteroGatedGraphClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, metadata):
        super(HeteroGatedGraphClassifier, self).__init__()

        self.ggnn = HeteroGatedGraphConv(output_dim, 8, metadata[1])

        # Copied from saikat107 implementation https://github.com/saikat107/Devign/blob/master/modules/model.py
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_dict, edge_index_dict, batch):
        h = self.ggnn(x_dict, edge_index_dict)

        x_i, _ = batch.de_batchify_graphs(h)
        h_i, _ = batch.de_batchify_graphs(x)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()

        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)


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