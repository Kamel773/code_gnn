import copy
import dataclasses
import itertools
import json
import os
import shutil
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import tqdm
from typing import List, Union

from cpg import CachedCPG
from embeddings import CodeBERTEmbeddingGetter, Word2VecEmbeddingGetter, RandomEmbeddingGetter
from google_drive_downloader import GoogleDriveDownloader as gdd

import torch
from torch_geometric.data import InMemoryDataset
from typing import Callable

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



@dataclasses.dataclass
class CodeDatasetConfig:
    download_links: List[str]
    file_names: List[str]
    label_tag: Union[str, List[int]]
    code_tag: str
    file_name_hash_fields: List[str]
    extra_file_names: List[str] = dataclasses.field(default_factory=lambda: [])
    unzip: bool = False

    def __post_init__(self):
        lengths_to_validate = [len(self.download_links), len(self.file_names)]
        if isinstance(self.file_name_hash_fields, list):
            lengths_to_validate.append(self.file_name_hash_fields)
        assert all([length == lengths_to_validate[0] for length in lengths_to_validate]), lengths_to_validate

    def file_name_hash(self, data):
        return '_'.join(map(str, (data[field] for field in self.file_name_hash_fields))) + '.c'


class MyCodeDataset(InMemoryDataset):
    configs = {
        # https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch
        "devign": CodeDatasetConfig(
            download_links=['1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF'],
            file_names=['function.json'],
            code_tag='func',
            label_tag='target',
            file_name_hash_fields=["project", "commit_id", "target"],
        ),
        # https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch
        "reveal": CodeDatasetConfig(
            download_links=['1KuIYgFcvWUXheDhT--cBALsfy1I4utOy'],
            file_names=['vulnerables.json', 'non-vulnerables.json'],
            code_tag='code',
            label_tag=[1, 0],
            extra_file_names=[],
            file_name_hash_fields=["project", "hash"],
            unzip=True,
        ),
        # https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
        "big-vuln": CodeDatasetConfig(
            download_links=['1deNsPfeh77h1SHjJURYOeyCR96JgxB_A'],
            file_names=['MSR_data_cleaned.json'],
            label_tag='vul',
            code_tag='func_before',
            extra_file_names=['MSR_data_cleaned_json.zip'],
            unzip=True,
        )
    }

    def __init__(self, root, name, transform=None, pre_transform=None, embed_type=None, workers=8):
        if embed_type == 'codebert':
            self.embedding_getter = CodeBERTEmbeddingGetter()
        elif embed_type == 'word2vec':
            self.embedding_getter = Word2VecEmbeddingGetter()
        elif embed_type is None:
            self.embedding_getter = RandomEmbeddingGetter()
        else:
            raise NotImplementedError('Embedding type ' + embed_type)
        self.name = name
        self.embed_type = embed_type

        self.config = self.__class__.configs[name]
        self.workers = workers

        root = os.path.join(root, self.__class__.__name__)
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, self.embed_type)

    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        ret = []
        ret += self.config.extra_file_names
        for file_name in self.config.file_names:
            ret.append(file_name)
            ret.append(self.get_files_path(file_name).name)
            ret.append(self.get_parsed_path(file_name).name)
            ret.append(self.get_metadata_path(file_name).name)
        return ret

    def get_files_path(self, path):
        path = Path(path)
        return path.parent / ('files_' + path.name)

    def get_parsed_path(self, path):
        path = Path(path)
        return path.parent / ('parsed_' + path.name)

    def get_metadata_path(self, path):
        path = Path(path)
        return path.parent / ('metadata_' + path.name)

    def download(self):
        for link, raw_path in zip(self.config.download_links, self.raw_paths):
            raw_path = Path(raw_path)
            if not raw_path.exists():
                gdd.download_file_from_google_drive(file_id=link,
                                                    dest_path=str(raw_path),
                                                    showsize=True)

            with open(raw_path) as f:
                data = json.load(f)
            for d in data:
                # NOTE: Adding this attribute for convenience, but it will not be persisted
                d["file_name"] = self.config.filename_hash(d)
            data = list(sorted(data, key=lambda x: x["file_name"]))
            idx = 0
            for d in data:
                d["file_name"] = f'{idx}_{d["file_name"]}'
                idx += 1

            metadata_path = self.get_metadata_path(raw_path)
            if not metadata_path.exists():
                metadata = copy.deepcopy(data)
                for d in metadata:
                    del d[self.config.code_tag]
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)

            source_dir = self.get_files_path(raw_path)
            if not source_dir.exists():
                source_dir.mkdir()
                try:
                    for d in data:
                        file_path = source_dir / d["file_name"]
                        with open(file_path, 'w') as f:
                            # TODO: Possibly collapse whitespaces
                            f.write(d["func"])
                except Exception:
                    shutil.rmtree(source_dir)
                    raise

            parsed_path = self.get_parsed_path(raw_path)
            if not parsed_path.exists():
                try:
                    parser = CachedCPG(source_dir, parsed_path, [source_dir / d["file_name"] for d in data])
                    parser.pre_parse()
                except Exception:
                    shutil.rmtree(parsed_path)
                    raise

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for i, raw_file_name in enumerate(self.config.file_names):
            raw_path = Path(self.raw_dir) / raw_file_name
            source_dir = self.get_files_path(raw_path)
            with open(self.get_metadata_path(raw_path)) as f:
                data = json.load(f)

            code_files = [source_dir / d["file_name"] for d in data]
            if isinstance(self.config.label_tag, str):
                labels = [d[self.config.label_tag] for d in data]
            else:
                labels = [d[self.config.label_tag[i]] for d in data]

            parser = CachedCPG(source_dir, self.get_parsed_path(raw_path), code_files)

            input_data = list(zip(code_files, labels))
            # input_data = input_data[:100]
            assert len(input_data) > 0, f'No input data from {raw_file_name}'
            errored_files = defaultdict(list)
            with tqdm.tqdm(input_data, desc=raw_file_name) as pbar:
                for file, label in pbar:
                    try:
                        cpg = parser.get_cpg(file)

                        # Get statement embedding
                        node_embeddings = self.embedding_getter.get_embedding(file, cpg)

                        # Concatenate node type
                        node_type = nx.get_node_attributes(cpg, 'type')
                        type_embeddings = torch.stack([type_one_hot[node_type_map[node_type[node]] - 1] for node in cpg.nodes])
                        node_embeddings = torch.cat((type_embeddings, node_embeddings), dim=1)

                        # TODO: Find and remove isolated nodes because of this filtering
                        u_idxs, v_idxs, edge_types = zip(*[e for e in cpg.edges(data='type') if e[2] != 'IS_FILE_OF'])
                        edge_index = torch.tensor((u_idxs, v_idxs), dtype=torch.long)

                        edge_attr = torch.stack([torch.tensor([edge_type_map[edge_type]]) for edge_type in edge_types], dim=0)
                        data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
                    except AssertionError as e:
                        errored_files[e].append(file)
                    except Exception:
                        pbar.write(traceback.format_exc())
                    data_list.append(data)
            for error_msg, files in errored_files.items():
                print(f'Error "{error_msg}":')
                print(files)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

        data = data_list[0]
        print(data)
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
