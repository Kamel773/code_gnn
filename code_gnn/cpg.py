import json
from io import StringIO

import networkx as nx
from cpgqls_client import CPGQLSClient, import_code_query, workspace_query


class CPG:
    def __init__(self):
        server_endpoint = "127.0.0.1:8080"
        self.client = CPGQLSClient(server_endpoint)

    def run_query(self, query):
        result = self.client.execute(query)
        print(query, ':', json.dumps(result, indent=2))
        return result

    def get_cpg(self, file):
        filepath = str(file.parent).replace("\\", "/")
        self.run_query(f'importCode.c("/share/{filepath}")')
        self.run_query(f'run.ossdataflow')
        result = self.run_query('cpg.runScript("custom/serialize-with-locations.sc", Map("methodName" -> cpg.method.order(1).head.name))')
        dot = result["stdout"]
        dot = dot[dot.find('"""') + len('"""'):dot.rfind('"""')]
        # print(dot)
        # dst_dot = file.parent.parent / (file.name + '.dot')
        # with open(dst_dot, 'w') as f:
        #     f.write(dot)
        # cpg = nx.drawing.nx_pydot.from_pydot(graphviz.Source.from_file(dst_dot))
        cpg = nx.drawing.nx_pydot.read_dot(StringIO(dot))
        node_attr = {}
        labels = dict(nx.get_node_attributes(cpg, 'label'))
        for i, (key, label) in enumerate(labels.items()):
            label_split = label[2:-1].split(',')
            node_type = label_split[0]
            code = label_split[1]
            if i < 10:
                print(node_type, code)
            node_attr[key] = {"type": node_type, "code": code}
        nx.set_node_attributes(cpg, {key: {"code": values["label"]} for key, values in zip(*cpg.nodes)})

        return cpg
