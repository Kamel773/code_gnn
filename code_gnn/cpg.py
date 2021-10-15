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
octopus_jars = list(Path('old-joern/projects/octopus/lib').glob('*.jar'))
assert len(octopus_jars) > 0, 'no jars in old-joern/projects/octopus/lib'
jars += octopus_jars
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
        # raise Exception(f'Error running command: {cmd}. Last output: {line}')
        print(f'Error running command: {cmd}. Last output: {line}')
    print('Done parsing', src_dir, 'to', joern_dir)


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


def read_graph(parsed_dir, filepath):
    output_path = parsed_dir / str(filepath)
    assert output_path.exists(), f'joern output file not found: {output_path}'
    nodes_path = output_path / 'nodes.csv'
    edges_path = output_path / 'edges.csv'
    assert nodes_path.exists(), f'node file not found: {nodes_path}'
    assert edges_path.exists(), f'edge file not found: {edges_path}'
    nodes_data = read_csv(nodes_path)
    edges_data = read_csv(edges_path)
    return nodes_data, edges_data


def parse_all(parse_dir, source_dir, files):
    parse_dir = Path(parse_dir)
    source_dir = Path(source_dir)
    print(f'Parsing {len(files)} files...')
    run_joern(parse_dir, source_dir, src_files=set(map(str, files)))


def get_cpg(parse_dir, file):
    parse_dir = Path(parse_dir)
    file = Path(file)
    nodes_data, edges_data = read_graph(parse_dir, file)
    cpg = nx.MultiDiGraph()
    assert nodes_data[0]["type"] == 'File', 'type of first node must be "File"'
    key_offset = int(nodes_data[0]["key"])
    for na in nodes_data:
        na.update({"label": f'{na["key"]} ({na["type"]}): {na["code"]}'})  # Graphviz label
        # Cover fault in Joern exposed by tests/acceptance/loop_exchange/chrome_debian/18159_0.c
        if na["type"].endswith('Statement'):
            line, col, offset, end_offset = (int(x) for x in na["location"].split(':'))
            if na["type"] == 'CompoundStatement':
                na["location"] = ':'.join(str(o) for o in (line, col, offset, end_offset))
    nodes = list(zip([int(x["key"])-key_offset for x in nodes_data], nodes_data))
    assert len(nodes) > 0, 'no nodes in the file'
    cpg.add_nodes_from(nodes)

    # Multigraph
    unique_edge_types = sorted(set(ea["type"] for ea in edges_data))
    edge_type_idx = {et: i for i, et in enumerate(unique_edge_types)}
    for ea in edges_data:
        ea.update({"label": f'({ea["type"]}): {ea["var"]}', "color": edge_type_idx[ea["type"]], "colorscheme": "pastel28"})  # Graphviz label
    edges = [(int(x["start"])-key_offset, int(x["end"])-key_offset, x) for x in edges_data]
    assert len(edges) > 0, 'no edges in the file'
    cpg.add_edges_from(edges)

    return cpg
