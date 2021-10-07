from collections import defaultdict

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cpg import parse
# https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
from torch_geometric.data import HeteroData

code_files = ["test.c", 'x42/c/X42.c']
data_list = []
for code in code_files:
    cpg = parse(code)

    data = HeteroData()

    num_features_node = 128
    num_features_edge = 128

    data['node'].x = torch.randn((len(cpg.nodes), num_features_node))
    edges_by_type = defaultdict(list)
    for u, v, t in cpg.edges.data("type"):
        edges_by_type[t].append((u, v))
    for t, edges in edges_by_type.items():
        edge_data = data['node', t, 'node']
        edge_data.edge_index = torch.tensor(edges)
        edge_data.edge_attr = torch.randn((len(edges), num_features_edge))
    data_list.append(data)

data = data_list[0]
print(data)
print(data.metadata())
print(data.x_dict)
print(data.edge_index_dict)

loader = DataLoader(data_list, batch_size=32)
