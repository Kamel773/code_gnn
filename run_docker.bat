docker build -t joern:latest .
docker run -vD:\weile-lab\thesis\ssl-gnn\joern:/share -p 127.0.0.1:8080:8080 --name joern_server -itd joern:latest bash
