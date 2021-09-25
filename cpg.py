import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

joern_bin = Path(__file__).parent / 'old-joern/joern-parse'
assert joern_bin.exists(), joern_bin


def gather_stmts(nodes):
    statements = []
    for node in nodes:
        if node["isCFGNode"] == True and node["type"].endswith('Statement') and node["code"]:
            statements.append(node)
    return statements


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.count(os.sep)
        indent = ' ' * 4 * (level)
        logger.debug('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.debug('{}{}'.format(subindent, f))


def parse(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    tmp_root = Path('./tmp')
    if not tmp_root.exists():
        tmp_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=str(tmp_root.absolute()) + '/') as tmp_dir:
        tmp_dir = Path(tmp_dir)
        # Invoke joern
        tmpfile_dir = tmp_dir / 'tmpfile'
        tmpfile_dir.mkdir()
        dst_filepath = tmpfile_dir / filepath.name
        shutil.copyfile(filepath, dst_filepath)
        joern_dir = tmp_dir / 'parsed'
        try:
            cmd = f'bash {joern_bin} {tmpfile_dir.absolute()} -outdir {joern_dir.absolute()}'
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                logger.error(proc.stdout.decode())
                return
            output_path = joern_dir / str(dst_filepath.absolute())[1:]
            assert output_path.exists(), output_path
            nodes_path = output_path / 'nodes.csv'
            edges_path = output_path / 'edges.csv'
            assert nodes_path.exists(), nodes_path
            assert edges_path.exists(), edges_path
            nodes_df = pd.read_csv(nodes_path, sep='\t')
            edges_df = pd.read_csv(edges_path, sep='\t')
        finally:
            shutil.rmtree(joern_dir)

    cpg = nx.MultiDiGraph()
    nodes_attributes = [{k: v if not pd.isnull(v) else '' for k, v in dict(row).items()} for i, row in
                        nodes_df.iterrows()]
    for na in nodes_attributes:
        na.update({"label": f'{na["key"]} ({na["type"]}): {na["code"]}'})  # Graphviz label

        # Cover fault in Joern exposed by tests/acceptance/loop_exchange/chrome_debian/18159_0.c
        if na["type"].endswith('Statement'):
            col, line, offset, end_offset = (int(x) for x in na["location"].split(':'))
            if na["type"] == 'CompoundStatement':
                na["location"] = ':'.join(str(o) for o in (col, line, offset, end_offset))
    nodes = list(zip(nodes_df["key"].values.tolist(), nodes_attributes))
    cpg.add_nodes_from(nodes)

    # Multigraph
    edges_attributes = [dict(row) for i, row in edges_df.iterrows()]
    unique_edge_types = sorted(set(ea["type"] for ea in edges_attributes))
    edge_type_idx = {et: i for i, et in enumerate(unique_edge_types)}
    for ea in edges_attributes:
        ea.update({"label": f'({ea["type"]}): {ea["var"]}', "color": edge_type_idx[ea["type"]], "colorscheme": "set19"})  # Graphviz label
    edges = list(zip(edges_df["start"].values.tolist(), edges_df["end"].values.tolist(), edges_attributes))
    cpg.add_edges_from(edges)

    return cpg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('code', help='Name of code file')
    args = parser.parse_args()
    code = Path(args.code)
    assert code.exists()
    cpg = parse(code)
    graphviz_cpg = nx.nx_pydot.to_pydot(cpg)
    print(graphviz_cpg)
