# ============ 阶段1: 构建阶段 (安装编译依赖 + pip 包) ============
FROM nvcr.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖 + Python (含编译工具)
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# 创建 venv 并安装所有 Python 依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt \
    fastapi uvicorn python-multipart websockets

# ============ 阶段2: 运行阶段 (只保留运行时必要的内容) ============
FROM nvcr.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 只安装运行时依赖 (不装 build-essential)
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libportaudio2 \
    ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制整个 venv (包含所有依赖和可执行文件)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 设置挂载点 (用于持久化声纹数据库)
VOLUME /app/voiceprint_db

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
