FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "fastapi[all]" "transformers>=4.44" "torch" "accelerate"

EXPOSE 8080
ENV HF_MODEL_ID="openai/gpt-oss-20b"

COPY app.py /app/app.py
WORKDIR /app

CMD uvicorn app:app --host 0.0.0.0 --port 8080