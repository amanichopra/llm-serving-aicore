# Specify the base layers (default dependencies) to use
FROM vllm/vllm-openai:latest

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    nginx \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose port and set environment variables for Ollama
ENV OLLAMA_HOST=0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Configure nginx for reverse proxy
RUN echo "events { use epoll; worker_connections 128; } \
    http { \
        server { \
                    listen 8080; \
                        location ^~ /v1/api/ { \
                            proxy_pass http://localhost:11434/api/; \
                            proxy_set_header Host \$host; \
                            proxy_set_header X-Real-IP \$remote_addr; \
                            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for; \
                            proxy_set_header X-Forwarded-Proto \$scheme; \
                        } \
                        location ^~ /v1/chat/ { \
                            proxy_pass http://localhost:11434/v1/chat/; \
                            proxy_set_header Host \$host; \
                            proxy_set_header X-Real-IP \$remote_addr; \
                            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for; \
                            proxy_set_header X-Forwarded-Proto \$scheme; \
                        } \
                } \
        }" > /etc/nginx/nginx.conf && \
    chmod -R 777 /var/log/nginx /var/lib/nginx /run

EXPOSE 8080

# Create directory for user nobody SAP AI Core run-time
RUN mkdir -p /nonexistent/.ollama && \
    chown -R nobody:nogroup /nonexistent && \
    chmod -R 770 /nonexistent

# Start nginx and Ollama service
CMD service nginx start && /usr/local/bin/ollama serve