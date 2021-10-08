import json

from cpgqls_client import CPGQLSClient, import_code_query, workspace_query

server_endpoint = "127.0.0.1:8080"
client = CPGQLSClient(server_endpoint)


def run_query(query):
    result = client.execute(query)
    print(query, ':', json.dumps(result, indent=2))


# execute a simple CPGQuery
# run_query("val a = 1")

# execute a `workspace` CPGQuery
# run_query(workspace_query())
run_query(import_code_query("/share/data/test-project", "my-c-project"))
# run_query('cpg')
# run_query('cpg.method("main").l')

query = 'cpg.runScript("custom/serialize-with-locations.sc")'
result = client.execute(query)
print(query, ':', json.dumps(result, indent=2))
dot = result["stdout"]
dot = dot[dot.find('"""') + len('"""'):dot.rfind('"""')]
print(dot)
