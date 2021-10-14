import json
import os
from collections import defaultdict
from pathlib import Path

import dgl
import networkx as nx
import torch
import tqdm
from dgl import save_graphs, load_graphs
from google_drive_downloader import GoogleDriveDownloader as gdd

from cpg import get_cpg, parse_all
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
all_edge_types = edge_type_map.keys()

from dgl.data import DGLDataset

class MyDGLDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 embed_type,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.embed_type = embed_type
        if embed_type == 'codebert':
            self.embedding_getter = CodeBERTEmbeddingGetter()
        elif embed_type == 'word2vec':
            self.embedding_getter = Word2VecEmbeddingGetter()
        elif embed_type is None:
            self.embedding_getter = RandomEmbeddingGetter()
        else:
            raise NotImplementedError('Embedding type ' + embed_type)
        self.labels = None
        self.graphs = None
        super(MyDGLDataset, self).__init__(name='devign',
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def download(self):
        # download raw data to local disk
        os.makedirs(self.raw_path, exist_ok=True)
        gdd.download_file_from_google_drive(file_id='1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF',
                                            dest_path=os.path.join(self.raw_path, 'function.json'),
                                            showsize=True)

    def process(self):
        with open(os.path.join(self.raw_path, 'function.json')) as f:
            raw_datas = json.load(f)

        # Generate names for files and sort them by name
        for d in raw_datas:
            # NOTE: Adding this attribute for convenience, but it will not be persisted
            d["file_name"] = '_'.join(map(str, [d[field] for field in ["project", "commit_id", "target"]])) + '.c'
        raw_datas = list(sorted(raw_datas, key=lambda x: x["file_name"]))
        idx = 0
        for d in raw_datas:
            d["file_name"] = f'{idx}_{d["file_name"]}'
            idx += 1

        # Write source files
        source_dir = Path(self.save_path) / 'files'
        if not source_dir.exists():
            os.makedirs(source_dir, exist_ok=True)
            for d in raw_datas:
                file_path = os.path.join(source_dir, d["file_name"])
                code = d["func"]
                try:
                    with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                        # TODO: Possibly collapse whitespaces
                        f.write(code)
                except OSError as e:
                    print('Error writing file', file_path, e)
                    open(file_path, 'a').close()

        # Parse all files at once
        parsed_dir = Path(self.save_path) / 'parsed'
        if not parsed_dir.exists():
            parse_all(parsed_dir, source_dir, [str(source_dir / d["file_name"]) for d in raw_datas])

        # Preprocess each source file
        graphs = []
        labels = []
        with tqdm.tqdm(raw_datas) as pbar:
            for i, raw_data in enumerate(pbar):
                file = os.path.join(source_dir, raw_data["file_name"])
                try:
                    cpg = get_cpg(parsed_dir, file)
                except AssertionError as e:
                    pbar.write(f'Error parsing {file}: {e}')
                    continue

                # Filter out IS_FILE_OF attribute
                edges_to_remove = [(u, v, k) for u, v, k, t in cpg.edges(keys=True, data='type') if t == 'IS_FILE_OF']
                cpg.remove_edges_from(edges_to_remove)
                cpg.remove_nodes_from(list(nx.isolates(cpg)))
                offset = min(cpg.nodes())
                # Relabel nodes starting from 0 because DGL will start node indices from 0
                offset_mapping = {old_label: old_label - offset for old_label in cpg.nodes()}
                cpg = nx.relabel_nodes(cpg, offset_mapping)

                # Remove graphs with more than 500 nodes for computational efficiency, as in the paper
                if len(cpg.nodes) > 500:
                    continue

                # Filter out IS_FILE_OF edges
                us, vs, edge_types = zip(*cpg.edges(data='type'))
                g = dgl.graph((us, vs))

                # Edge feature is just the edge type
                g.edata['etype'] = torch.tensor([edge_type_map[t] for t in edge_types])

                # Get configurable statement embedding
                # TODO: If we want to train the embedding network (CodeBERT) as well, then we need to instead store the
                #  tokens of the code and instantiate the embedding with the computational graph connected to optimizer
                node_embeddings = self.embedding_getter.get_embedding(file, cpg)
                # Concatenate node type
                node_type = nx.get_node_attributes(cpg, 'type')
                type_embeddings = torch.stack([type_one_hot[node_type_map[node_type[node]] - 1] for node in cpg.nodes])
                node_embeddings = torch.cat((type_embeddings, node_embeddings), dim=1)
                g.ndata['h'] = node_embeddings

                graphs.append(g)
                labels.append(int(raw_data["target"]))

        self.graphs = graphs
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    @property
    def processed_file(self):
        return os.path.join(self.save_path, f'dgl_data_{self.embed_type}.bin')

    @property
    def max_etypes(self):
        return max(max(g.edata['etype']) for g in self.graphs)+1

    def save(self):
        # save processed data to directory `self.save_path`
        save_graphs(self.processed_file, self.graphs, {'labels': self.labels})

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(self.processed_file)
        self.labels = label_dict['labels']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        return os.path.exists(self.processed_file)
