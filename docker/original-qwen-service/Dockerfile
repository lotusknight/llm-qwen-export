# Original Qwen inference service — supports CPU or CUDA via build arg
# For China mainland: Docker Hub mirror + Tsinghua pip/apt (override BASE_IMAGE=python:3.11-slim if needed)
ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${BASE_IMAGE}

ARG RUNTIME=cpu
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# Tsinghua mirror for PyTorch CUDA wheels (China); use https://download.pytorch.org/whl/cu121 elsewhere
ARG PYTORCH_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121

WORKDIR /app

# Tsinghua apt mirror + system deps in one RUN (avoids cached apt without mirror)
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
      [ -f "$f" ] && sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' "$f"; \
    done \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r requirements.txt
RUN if [ "$RUNTIME" = "cuda" ]; then \
    pip install --no-cache-dir torch --index-url "${PYTORCH_INDEX_URL}"; \
    fi

COPY app.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8000
# Use HF mirror so MODEL_ID download works from China (override with -e HF_ENDPOINT= for default Hugging Face)
ENV HF_ENDPOINT=https://hf-mirror.com
EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]
