import itertools
import os
from collections import defaultdict

import gensim
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


# Source: https://github.com/VulDetProject/ReVeal/blob/ca31b783384b4cdb09b69950e48f79fa0748ef1d/data_processing/create_ggnn_data.py#L225-L242
type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}
type_one_hot = torch.eye(len(type_map))


def get_word2vec_embedding(cpg, wv_model):
    node_code = nx.get_node_attributes(cpg, 'code')

    def get_node_embedding(node):
        code = node_code[node]
        tokens = code.strip().split()
        token_embeddings = [torch.tensor(wv_model.wv[tok]) for tok in tokens if tok in wv_model.wv]
        if len(token_embeddings) == 0:
            node_embedding = torch.zeros(wv_model.vector_size)
        else:
            node_embedding = torch.stack(token_embeddings, dim=0).mean(dim=0)
        return node_embedding

    node_embeddings = [get_node_embedding(node) for node in cpg.nodes]
    embedding = torch.stack(node_embeddings, dim=0)

    return embedding


def get_dataset(use_codebert=False, use_word2vec=False):
    if use_word2vec or use_codebert:
        assert not (use_word2vec and use_codebert), 'word2vec and codebert should be used independently'

    code_files = [
        "test.c",
        # 'x42/c/X42.c',
    ]

    # Read data into huge `Data` list.
    if use_codebert:
        codebert_api = CodeBERTAPI()
    elif use_word2vec:
        corpus_pretrained = 'word2vec/devign.wv'  # TODO: initialize this for the dataset we're actually training on
        assert os.path.exists(corpus_pretrained), f"Train Word2Vec and save to {corpus_pretrained}! You're on your own bud."
        wv_model = gensim.models.Word2Vec.load(corpus_pretrained)
    else:
        codebert_api = None
        wv_model = None

    data_list = []
    for file in code_files:
        cpg = parse(file)
        data = HeteroData()

        # Get statement embedding
        if use_codebert:
            node_embeddings = get_codebert_embedding(file, cpg, codebert_api)
        elif use_word2vec:
            node_embeddings = get_word2vec_embedding(cpg, wv_model)
        else:
            node_embeddings = torch.randn((len(cpg.nodes), 128))

        # Concatenate node type
        node_type = nx.get_node_attributes(cpg, 'type')
        type_embeddings = torch.stack([type_one_hot[type_map[node_type[node]] - 1] for node in cpg.nodes])
        node_embeddings = torch.cat((type_embeddings, node_embeddings), dim=1)

        data['node'].x = node_embeddings

        num_features_edge = 128
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