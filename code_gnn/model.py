import dgl
import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
from torch.nn import functional as F


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_etypes, num_steps):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = n_etypes
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                               n_steps=num_steps, n_etypes=n_etypes)
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

    def reset_parameters(self):
        self.ggnn.reset_parameters()
        self.conv_l1.reset_parameters()
        self.conv_l2.reset_parameters()
        self.conv_l1_for_concat.reset_parameters()
        self.conv_l2_for_concat.reset_parameters()
        self.mlp_z.reset_parameters()
        self.mlp_y.reset_parameters()

    def stack_pad_zeros(self, graph_features):
        features = []
        max_length = max(len(g) for g in graph_features)
        for feature in graph_features:
            pad_dim = max_length - len(feature)
            # feature = F.pad(feature, (0, pad_dim) + (0, 0) * (len(feature.shape)-1))
            feature = torch.cat(
                (feature, torch.zeros(size=(pad_dim, *(feature.shape[1:])), requires_grad=feature.requires_grad, device=feature.device)), dim=0)
            features.append(feature)
        return torch.stack(features, dim=0)

    def forward(self, batch):
        batch.ndata['h_next'] = self.ggnn(batch, batch.ndata['h'], batch.edata['etype'])
        graphs = dgl.unbatch(batch, batch.batch_num_nodes(), batch.batch_num_edges())
        features = self.stack_pad_zeros([g.ndata['h'] for g in graphs])
        h_i = self.stack_pad_zeros([g.ndata['h_next'] for g in graphs])
        c_i = torch.cat((h_i, features), dim=-1)
        Y_1 = self.maxpool1(
            F.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            F.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result
