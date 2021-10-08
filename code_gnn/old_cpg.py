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

joern_bin = Path(__file__).parent.parent / 'old-joern/joern-parse'
assert joern_bin.exists(), joern_bin

jars = [
    Path("old-joern/projects/extensions/joern-fuzzyc/build/libs/joern-fuzzyc.jar"),
    Path('old-joern/projects/extensions/jpanlib/build/libs/jpanlib.jar'),
]
jars += Path('old-joern/projects/octopus/lib').glob('*.jar')
sep = ';' if os.name == 'nt' else ':'
jars_str = sep.join(str(j) for j in jars)


def run_joern(joern_dir, tmpfile_dir):
    cmd = f'java ' \
          f'-cp "{jars_str}" ' \
          f'tools.parser.ParserMain ' \
          f'-outformat csv ' \
          f'-outdir {joern_dir} ' \
          f'{tmpfile_dir}'
    print(cmd)  # For debugging
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise Exception(proc.stdout.decode())


def to_graph(nodes_df, edges_df):
    cpg = nx.MultiDiGraph()
    nodes_attributes = [{k: v if not pd.isnull(v) else '' for k, v in dict(row).items()} for i, row in
                        nodes_df.iterrows()]
    for na in nodes_attributes:
        na.update({"label": f'{na["key"]} ({na["type"]}): {na["code"]}'})  # Graphviz label

        # Cover fault in Joern exposed by tests/acceptance/loop_exchange/chrome_debian/18159_0.c
        if na["type"].endswith('Statement'):
            line, col, offset, end_offset = (int(x) for x in na["location"].split(':'))
            if na["type"] == 'CompoundStatement':
                na["location"] = ':'.join(str(o) for o in (line, col, offset, end_offset))
    nodes = list(zip([x-1 for x in nodes_df["key"].values.tolist()], nodes_attributes))
    cpg.add_nodes_from(nodes)

    # Multigraph
    edges_attributes = [dict(row) for i, row in edges_df.iterrows()]
    unique_edge_types = sorted(set(ea["type"] for ea in edges_attributes))
    edge_type_idx = {et: i for i, et in enumerate(unique_edge_types)}
    for ea in edges_attributes:
        ea.update({"label": f'({ea["type"]}): {ea["var"]}', "color": edge_type_idx[ea["type"]], "colorscheme": "pastel28"})  # Graphviz label
    edges = list(zip([x-1 for x in edges_df["start"].values.tolist()], [x-1 for x in edges_df["end"].values.tolist()], edges_attributes))
    cpg.add_edges_from(edges)

    return cpg


def parse(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    parsed_dir = filepath.parent.with_name('parsed_' + filepath.name)
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    run_joern(parsed_dir, filepath.parent)
    output_path = parsed_dir / str(filepath)
    assert output_path.exists(), output_path
    nodes_path = output_path / 'nodes.csv'
    edges_path = output_path / 'edges.csv'
    assert nodes_path.exists(), nodes_path
    assert edges_path.exists(), edges_path
    nodes_df = pd.read_csv(nodes_path, sep='\t')
    edges_df = pd.read_csv(edges_path, sep='\t')

    return to_graph(nodes_df, edges_df)


def parse_with_tmp(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    tmp_root = Path('./tmp')
    if not tmp_root.exists():
        tmp_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=str(tmp_root.absolute()) + '/') as tmp_dir:
        tmp_dir = Path(tmp_dir).relative_to(Path.cwd())
        # Invoke joern
        tmpfile_dir = tmp_dir / 'tmpfile'
        tmpfile_dir.mkdir()
        dst_filepath = tmpfile_dir / filepath.name
        shutil.copyfile(filepath, dst_filepath)
        joern_dir = tmp_dir / 'parsed'
        run_joern(joern_dir, tmpfile_dir)

        output_path = joern_dir / str(dst_filepath)
        assert output_path.exists(), output_path
        nodes_path = output_path / 'nodes.csv'
        edges_path = output_path / 'edges.csv'
        assert nodes_path.exists(), nodes_path
        assert edges_path.exists(), edges_path
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        edges_df = pd.read_csv(edges_path, sep='\t')

    return to_graph(nodes_df, edges_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('code', help='Name of code file')
    args = parser.parse_args()
    code = Path(args.code)
    assert code.exists()
    cpg = parse(code)
    graphviz_cpg = nx.nx_pydot.to_pydot(cpg)
    print(graphviz_cpg)
