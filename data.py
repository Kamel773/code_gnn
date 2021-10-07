import itertools
from collections import defaultdict

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from cpg import parse
from embeddings import CodeBERTEmbeddingGetter, Word2VecEmbeddingGetter, RandomEmbeddingGetter

# Source: https://github.com/VulDetProject/ReVeal/blob/ca31b783384b4cdb09b69950e48f79fa0748ef1d/data_processing/create_ggnn_data.py#L225-L257
node_type_map = {
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
type_one_hot = torch.eye(len(node_type_map))
# We currently consider 12 types of edges mentioned in ICST paper
edge_type_map = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}


def get_dataset(embed_type=None):

    code_files = [
        "test.c",
        # 'x42/c/X42.c',
    ]

    # Read data into huge `Data` list.
    if embed_type == 'codebert':
        embedding_getter = CodeBERTEmbeddingGetter()
    elif embed_type == 'word2vec':
        embedding_getter = Word2VecEmbeddingGetter()
    else:
        embedding_getter = RandomEmbeddingGetter()

    data_list = []
    for file in code_files:
        cpg = parse(file)
        data = HeteroData()

        # Get statement embedding
        node_embeddings = embedding_getter.get_embedding(file, cpg)

        # Concatenate node type
        node_type = nx.get_node_attributes(cpg, 'type')
        type_embeddings = torch.stack([type_one_hot[node_type_map[node_type[node]] - 1] for node in cpg.nodes])
        node_embeddings = torch.cat((type_embeddings, node_embeddings), dim=1)

        data['node'].x = node_embeddings

        edge_type = nx.get_edge_attributes(cpg, 'type')
        edges_by_type = defaultdict(list)
        for u, v, t in cpg.edges.data("type"):
            edges_by_type[t].append((u, v))
        for t, edges in edges_by_type.items():
            edge_data = data['node', t, 'node']
            edge_data.edge_index = torch.tensor(edges)
            edge_data.edge_attr = torch.stack([torch.tensor([edge_type_map[edge_type[edge]]]) for edge in cpg.edges if edge_type[edge] != 'IS_FILE_OF'], dim=0)
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
