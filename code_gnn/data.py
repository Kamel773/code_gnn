import copy
import dataclasses
import itertools
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from typing import List, Tuple, Dict

from cpg import get_cpg, parse_all
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


def bigvuln_dict_to_list(data, *args):
    data = list(data.values())
    for d in data:
        del d["Summary"]
        del d["func_after"]
        del d["patch"]
        del d["vul_func_with_fix"]
        d["commit_id"] = re.split(r'[^a-zA-Z0-9]', d["commit_id"])[0]
    return data


def add_reveal_label(data, raw_code_file):
    label = 1 if raw_code_file == 'vulnerables.json' else 0
    for d in data:
        d["label"] = label
    return data


def noop(data, *args):
    return data


@dataclasses.dataclass
class CodeDatasetConfig:
    download_links: List[Tuple[str, str]]
    file_names: List[str]
    label_tag: str
    code_tag: str
    file_name_hash_fields: List[str]
    transform: Callable[[List[Dict]], List[Dict]] = noop
    unzip: bool = False

    def __post_init__(self):
        if isinstance(self.label_tag, list):
            assert len(self.file_names) == len(self.label_tag), f'{len(self.file_names)=} should equal {len(self.label_tag)=}'

    def file_name_hash(self, data):
        return '_'.join(map(str, (data[field] for field in self.file_name_hash_fields))) + '.c'


class MyCodeDataset(InMemoryDataset):
    configs = {
        # https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch
        "devign": CodeDatasetConfig(
            download_links=[('1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF', 'function.json')],
            file_names=['function.json'],
            code_tag='func',
            label_tag='target',
            file_name_hash_fields=["project", "commit_id", "target"],
        ),
        # https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch
        "reveal": CodeDatasetConfig(
            download_links=[
                ('1NPtliGlRpR0CxKIUW4ah2EUggKTHnM71', 'non-vulnerables.json'),
                ('1I6N0Kxv5gmNzi12tpX8OZWQo7Z4w8Unz', 'vulnerables.json'),
            ],
            file_names=['vulnerables.json', 'non-vulnerables.json'],
            code_tag='code',
            label_tag='label',
            file_name_hash_fields=["project", "hash", 'label'],
            transform=add_reveal_label,
        ),
        # https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
        "big-vuln": CodeDatasetConfig(
            download_links=[('1deNsPfeh77h1SHjJURYOeyCR96JgxB_A', 'MSR_data_cleaned_json.zip')],
            file_names=['MSR_data_cleaned.json'],
            label_tag='vul',
            code_tag='func_before',
            file_name_hash_fields=['project', 'commit_id', 'vul'],
            transform=bigvuln_dict_to_list,
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
        ret = []
        for raw_path in self.raw_paths:
            ret.append(self.get_processed_path(raw_path).name)
        return ret

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        ret = self.raw_source_file_paths
        for raw_path in self.raw_source_file_paths:
            ret.append(self.get_files_path(raw_path).name)
            ret.append(self.get_parsed_path(raw_path).name)
            ret.append(self.get_metadata_path(raw_path).name)
        return ret

    @property
    def raw_source_file_paths(self):
        ret = []
        for file_name in self.config.file_names:
            ret.append(Path(self.raw_dir) / file_name)
        return ret

    def get_processed_path(self, path):
        path = Path(path)
        return path.parent / ('processed_' + path.name + '.pt')

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
        for link, dst_file in self.config.download_links:
            dst_file = Path(self.raw_dir, dst_file)
            if not dst_file.exists():
                gdd.download_file_from_google_drive(file_id=link,
                                                    dest_path=str(dst_file),
                                                    showsize=True,
                                                    unzip=self.config.unzip)
        for i, raw_path in enumerate(self.raw_source_file_paths):
            # Preprocess raw file
            raw_path = Path(raw_path)
            with open(raw_path) as f:
                data = json.load(f)
            data = self.config.transform(data, raw_path.name)
            for d in data:
                # NOTE: Adding this attribute for convenience, but it will not be persisted
                d["file_name"] = self.config.file_name_hash(d)
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
                        try:
                            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                                # TODO: Possibly collapse whitespaces
                                f.write(d[self.config.code_tag])
                        except OSError as e:
                            print('Error writing file', file_path, e)
                            open(file_path, 'a').close()
                except Exception:
                    shutil.rmtree(source_dir)
                    raise

            parsed_path = self.get_parsed_path(raw_path)
            if not parsed_path.exists():
                try:
                    parse_all(source_dir, parsed_path, [source_dir / d["file_name"] for d in data])
                except Exception:
                    shutil.rmtree(parsed_path)
                    raise


    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for i, raw_path in enumerate(self.raw_source_file_paths):
            raw_path = Path(raw_path)

            # Preprocess each source file
            processed_path = Path(self.processed_dir) / self.get_processed_path(raw_path).name
            if not processed_path.exists():
                source_dir = self.get_files_path(raw_path)
                with open(self.get_metadata_path(raw_path)) as f:
                    data = json.load(f)

                code_files = [source_dir / d["file_name"] for d in data]
                labels = [int(d[self.config.label_tag]) for d in data]

                parse_dir = self.get_parsed_path(raw_path)

                input_data = list(zip(code_files, labels))
                assert len(input_data) > 0, f'No input data from {raw_path}'
                for file, label in input_data:
                    try:
                        cpg = get_cpg(parse_dir, file)
                    except AssertionError as e:
                        # pbar.write(f'Error parsing {file}: {e}')
                        print(f'Error parsing {file}: {e}')
                        continue
                    # Make data
                    data = HeteroData()

                    # Get statement embedding
                    node_embeddings = self.embedding_getter.get_embedding(file, cpg)

                    # Concatenate node type
                    node_type = nx.get_node_attributes(cpg, 'type')
                    type_embeddings = torch.stack([type_one_hot[node_type_map[node_type[node]] - 1] for node in cpg.nodes])
                    node_embeddings = torch.cat((type_embeddings, node_embeddings), dim=1)
                    node_tag = 'node'
                    data[node_tag].x = node_embeddings

                    edge_idx_by_type = defaultdict(list)
                    # TODO: Find and remove isolated nodes because of this filtering
                    edge_idx = [e for e in cpg.edges(data='type') if e[2] != 'IS_FILE_OF']
                    for u, v, t in edge_idx:
                        edge_idx_by_type[t].append((u, v))
                    edge_idx_by_type = dict(edge_idx_by_type)
                    for t in edge_idx_by_type:
                        data[node_tag, t, node_tag].edge_index = torch.tensor(edge_idx_by_type[t]).t()

                    data[node_tag].y = torch.tensor([label])
                    data_list.append(data)

                self.data, self.slices = self.collate(data_list)
                torch.save((self.data, self.slices), processed_path)

                data = data_list[0]
                print(data)

    @property
    def num_node_features(self) -> int:
        data = self[0]
        if isinstance(data, HeteroData):
            return data['node'].num_features
        else:
            return super().num_node_features

    @property
    def num_classes(self) -> int:
        data = self[0]
        if isinstance(data, HeteroData):
            y = self.data['node'].y
            if y is None:
                return 0
            elif y.numel() == y.size(0) and not torch.is_floating_point(y):
                return int(y.max()) + 1
            elif y.numel() == y.size(0) and torch.is_floating_point(y):
                return torch.unique(y).numel()
            else:
                return y.size(-1)
        else:
            return super().num_node_features


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
