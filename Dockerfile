# 使用官方 Python 基础镜像 (推荐使用 Slim 版本减小体积)
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 安装系统依赖
# portaudio19-dev: 用于音频处理 (PyAudio)
# ffmpeg: 用于音频格式转换
# git: 用于下载模型 (如果需要)

RUN sed -i 's@deb.debian.org@mirrors.aliyun.com@g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt .
# 补充服务端需要的库
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt \
    fastapi uvicorn python-multipart websockets

# 3. 复制项目代码
COPY . .

# 4. 安装额外依赖
# 模型通过挂载本地目录或环境变量 (VAD_MODEL_PATH/ASR_MODEL_PATH/SPK_MODEL_PATH) 指定
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U openai-whisper

# 5. 暴露端口
EXPOSE 8000

# 6. 设置挂载点 (用于持久化声纹数据库)
VOLUME /app/voiceprint_db

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

