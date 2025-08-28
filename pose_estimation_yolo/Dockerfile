FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Update package lists and install wget & curl
RUN apt-get update && \
    apt-get install -y wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "fastapi" "uvicorn[standard]" "torch" "ultralytics" "opencv-python" "numpy" "python-multipart"

COPY app.py .

# Create directory for user nobody SAP AI Core run-time
RUN mkdir -p /nonexistent/.pose-detection && \
    chown -R nobody:nogroup /nonexistent && \
    chmod -R 770 /nonexistent

RUN mkdir -p /app/model
RUN wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -O ./model/yolo11n-pose.pt

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port 8080