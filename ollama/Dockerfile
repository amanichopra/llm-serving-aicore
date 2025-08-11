# Start from the official Ollama base image
FROM ollama/ollama:latest

# Expose Ollama's default API port
EXPOSE 11434

# SAP AI Core usually expects the container to start serving directly.
# This will run the Ollama API server
ENTRYPOINT ["ollama", "serve"]