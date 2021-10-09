import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import tqdm

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


def run_joern(joern_dir, src_dir, src_files=None):
    cmd = f'java ' \
          f'-cp "{jars_str}" ' \
          f'tools.parser.ParserMain ' \
          f'-outformat csv ' \
          f'-outdir {joern_dir} ' \
          f'{src_dir}'
    status = {
        "failed": 0,
        "succeeded": 0,
        "warnings": 0,
    }
    with tqdm.tqdm(total=len(src_files)) as pbar:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        for line in proc.stdout:
            if 'skipping' in line.lower():
                pbar.update(1)
                status["failed"] += 1
            elif line.strip() in src_files:
                pbar.update(1)
                status["succeeded"] += 1
            elif 'warning' in line:
                status["warnings"] += 1
            else:
                pbar.write(line)
            pbar.set_postfix(status)
    if proc.returncode != 0:
        print(f'Error running command: {cmd}. Last output: {line}')
    print('Done parsing', src_dir, 'to', joern_dir)


def to_graph(nodes_df, edges_df):
    cpg = nx.MultiDiGraph()
    nodes_attributes = nodes_df
    for na in nodes_attributes:
        na.update({"label": f'{na["key"]} ({na["type"]}): {na["code"]}'})  # Graphviz label
        # Cover fault in Joern exposed by tests/acceptance/loop_exchange/chrome_debian/18159_0.c
        if na["type"].endswith('Statement'):
            line, col, offset, end_offset = (int(x) for x in na["location"].split(':'))
            if na["type"] == 'CompoundStatement':
                na["location"] = ':'.join(str(o) for o in (line, col, offset, end_offset))
    nodes = list(zip([int(x["key"])-1 for x in nodes_attributes], nodes_attributes))
    assert len(nodes) > 0, 'No nodes'
    cpg.add_nodes_from(nodes)

    # Multigraph
    edges_attributes = edges_df
    unique_edge_types = sorted(set(ea["type"] for ea in edges_attributes))
    edge_type_idx = {et: i for i, et in enumerate(unique_edge_types)}
    for ea in edges_attributes:
        ea.update({"label": f'({ea["type"]}): {ea["var"]}', "color": edge_type_idx[ea["type"]], "colorscheme": "pastel28"})  # Graphviz label
    edges = [(int(x["start"])-1, int(x["end"])-1, x) for x in edges_attributes]
    assert len(edges) > 0, 'No edges'
    cpg.add_edges_from(edges)

    return cpg


def read_csv(filename):
    """
    This is probably faster than pd.read_csv
    though not tested
    """
    with open(filename) as f:
        lines = f.read().splitlines()
        headers = lines.pop(0).split('\t')
        rows = []
        for line in lines:
            row_data = {}
            for i, field in enumerate(line.split('\t')):
                header = headers[i]
                row_data[header] = field
            rows.append(row_data)
        return rows


def get_dfs(parsed_dir, filepath):
    output_path = parsed_dir / str(filepath)
    assert output_path.exists(), output_path
    nodes_path = output_path / 'nodes.csv'
    edges_path = output_path / 'edges.csv'
    assert nodes_path.exists(), nodes_path
    assert edges_path.exists(), edges_path
    nodes_df = read_csv(nodes_path)
    edges_df = read_csv(edges_path)
    return nodes_df, edges_df


def parse(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    parsed_dir = filepath.parent.with_name('parsed_' + filepath.name)
    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    run_joern(parsed_dir, filepath.parent)
    nodes_df, edges_df = get_dfs(parsed_dir, filepath)

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

        nodes_df, edges_df = get_dfs(joern_dir, dst_filepath)

    return to_graph(nodes_df, edges_df)


class CachedCPG:
    def __init__(self, source_dir, parsed_dir, files):
        self.source_dir = Path(source_dir)
        self.parse_dir = Path(parsed_dir)
        self.files = files

    def pre_parse(self):
        print(f'Parsing {len(self.files)} files...')
        run_joern(self.parse_dir, self.source_dir, src_files=set(map(str, self.files)))

    def get_cpg(self, file):
        nodes_df, edges_df = get_dfs(self.parse_dir, file)
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
