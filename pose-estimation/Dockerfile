FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir "fastapi" "uvicorn[standard]" "torch" "ultralytics" "opencv-python" "numpy"

COPY model/ ./model
COPY app.py .

# Create directory for user nobody SAP AI Core run-time
RUN mkdir -p /nonexistent/.pose-detection && \
    chown -R nobody:nogroup /nonexistent && \
    chmod -R 770 /nonexistent

CMD uvicorn app:app --host 0.0.0.0 --port 8080