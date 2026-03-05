#!/bin/bash
# 构建 Python venv 供精简 Docker 镜像使用
# 使用 Docker 多阶段构建确保二进制兼容性（在 Linux 环境中编译）
#
# 用法：bash scripts/build_venv.sh [输出目录]
# 默认输出到 ./venv_docker/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_DIR/venv_docker}"

echo "==> 构建 Python venv（GPU 版本，含 CUDA 12.1 支持）..."
echo "    输出目录: $OUTPUT_DIR"

# 使用临时 Dockerfile 构建 venv
TEMP_DOCKERFILE=$(mktemp)
cat > "$TEMP_DOCKERFILE" << 'DOCKERFILE'
FROM python:3.10-slim AS builder

WORKDIR /build
ENV DEBIAN_FRONTEND=noninteractive

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建 venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# 安装 PyTorch (CUDA 12.1) + 其他依赖
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    -r requirements.txt \
    fastapi uvicorn python-multipart websockets
DOCKERFILE

# 构建
BUILDER_IMAGE="voiceprint-venv-builder:tmp"
docker build -t "$BUILDER_IMAGE" -f "$TEMP_DOCKERFILE" "$PROJECT_DIR"
rm -f "$TEMP_DOCKERFILE"

# 导出 venv
echo "==> 导出 venv 到 $OUTPUT_DIR ..."
mkdir -p "$OUTPUT_DIR"

# 通过临时容器 cp 出 venv
CONTAINER_ID=$(docker create "$BUILDER_IMAGE")
docker cp "$CONTAINER_ID:/opt/venv/." "$OUTPUT_DIR/"
docker rm "$CONTAINER_ID" > /dev/null

# 清理 builder 镜像
docker rmi "$BUILDER_IMAGE" > /dev/null 2>&1 || true

echo "==> 完成！venv 已导出到: $OUTPUT_DIR"
echo "    大小: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "启动精简容器："
echo "    docker compose --profile slim up -d"
