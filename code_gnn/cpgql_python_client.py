from cpgqls_client import CPGQLSClient, import_code_query, workspace_query

server_endpoint = "127.0.0.1:8080"
client = CPGQLSClient(server_endpoint)

# execute a simple CPGQuery
query = "val a = 1"
result = client.execute(query)
print(result)

# execute a `workspace` CPGQuery
query = workspace_query()
result = client.execute(query)
print(result['stdout'])

# execute an `importCode` CPGQuery
query = import_code_query("/share/project", "my-c-project")
result = client.execute(query)
print(result['stdout'])
