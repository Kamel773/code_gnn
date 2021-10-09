docker build -t joern:latest .
docker run -vD:\weile-lab\thesis\code_gnn:/share --name joern -itd joern:latest bash
docker run -vD:\weile-lab\thesis\code_gnn:/share -p 8080:8080 --name joern_server -itd joern:latest /joern/joern --server --server-host 0.0.0.0 --server-port 8080
