import itertools
from collections import defaultdict

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from codebert.api import CodeBERTAPI
from cpg import parse


def get_codebert_embedding(file, cpg, codebert_api):
    # TODO: standardize the text whitespace so that there is just one space in between each token.
    #  Do the same for cpg so that the offsets match.
    with open(file) as f:
        text = f.read()
    examples = [
        {
            "func": text,
            "idx": 0,
            "target": 0,
        }
    ]
    token_embeddings, features = codebert_api.get_token_embeddings(examples, return_features=True)
    # Map cpg nodes to features given the offsets in features[0].encoded.offsets
    offsets = features[0].encoded.offsets
    node_location = nx.get_node_attributes(cpg, 'location')
    node_code = nx.get_node_attributes(cpg, 'code')
    node_type = nx.get_node_attributes(cpg, 'type')
    node_tokens = []
    node_embeddings = []
    for e in cpg.nodes:
        token_text = None
        embed = None
        loc = node_location[e]
        if loc:
            _, _, n_begin, n_end = map(int, loc.split(':'))
            absolute_t_begin_i = None
            absolute_t_end_i = None
            for t_i, (t_begin, t_end) in enumerate(offsets):
                if n_begin >= t_begin:
                    absolute_t_begin_i = t_i
                if n_end <= t_end:
                    absolute_t_end_i = t_i
                    break
            if absolute_t_begin_i is None or absolute_t_end_i is None:
                continue
            tokens = features[0].encoded.tokens[absolute_t_begin_i:absolute_t_end_i+1]
            token_text = ' '.join(tokens)
            embeddings = token_embeddings[absolute_t_begin_i:absolute_t_end_i+1]
            embed = torch.mean(torch.stack(embeddings, dim=0), dim=0).squeeze(dim=0)
        node_tokens.append(token_text)
        node_embeddings.append(embed)

    # Debug outputs
    print('Mismatches:')
    for i, token_text in enumerate(node_tokens):
        if token_text is None:
            continue
        elif token_text.replace('Ġ', '') != node_code[i]:
            print(i, token_text, node_code[i], node_type[i], node_location[i], sep='|')

    print(len([n for n in node_embeddings if n is not None]), 'embeddings out of', len(cpg.nodes), 'nodes:')
    for i, (t, e) in enumerate(zip(node_tokens, node_embeddings)):
        if i in node_code:
            code = node_code[i]
        else:
            code = '<no code>'
        if i in node_location:
            loc = node_location[i]
        else:
            loc = '<no location>'
        if e is not None:
            embed_shape = e.shape
        else:
            embed_shape = '<no embedding>'
        print(i, code, loc, t, embed_shape, sep='|')
    print("graphviz:")
    nx.set_node_attributes(cpg, {i: 1 if n is not None else 2 for i, n in zip(cpg.nodes, node_embeddings)}, "color")
    nx.set_node_attributes(cpg, 'pastel28', "colorscheme")
    nx.set_node_attributes(cpg, 'filled', "style")
    graphviz_cpg = nx.nx_pydot.to_pydot(cpg)
    print(graphviz_cpg)

    default_init = torch.zeros
    node_embeddings = [n if n is not None else default_init(codebert_api.num_features) for n in node_embeddings]
    embedding = torch.stack(node_embeddings, dim=0)

    return embedding


def get_dataset(use_codebert=False):
    # Read data into huge `Data` list.
    if use_codebert:
        codebert_api = CodeBERTAPI()
    else:
        codebert_api = None
    code_files = [
        "test.c",
        # 'x42/c/X42.c',
    ]
    data_list = []
    for code in code_files:
        cpg = parse(code)

        data = HeteroData()

        num_features_edge = 128

        if use_codebert:
            data['node'].x = get_codebert_embedding(code, cpg, codebert_api)
        else:
            num_features_node = 128
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
    return data_list


def draw_9_plots(dataset):
    """Draw plots of graphs from the PROTEINS dataset."""
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