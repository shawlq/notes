docker run -d \
  --name code-server \
  -p 12200:8080 \
  -e PASSWORD="111" \
  -v /home/atom/code:/code \
  -w /code \
  -u "$(id -u):$(id -g)" \
  --restart unless-stopped \
  codercom/code-server:latest
