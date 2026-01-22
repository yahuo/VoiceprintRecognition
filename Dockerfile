# 使用官方 Python 基础镜像 (推荐使用 Slim 版本减小体积)
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 安装系统依赖
# portaudio19-dev: 用于音频处理 (PyAudio)
# ffmpeg: 用于音频格式转换
# git: 用于下载模型 (如果需要)
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt .
# 补充服务端需要的库
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn python-multipart websockets

# 3. 复制项目代码
COPY . .

# 4. [关键步骤] 将模型烘焙进镜像 (Bake Models)
# 这一步会执行下载脚本，将几 GB 的模型文件下载到镜像内的 .cache 目录
# 这样用户启动容器时，就不需要再联网下载模型了，做到"开箱即用"
# 注意：你需要先编写一个 download_models.py 脚本
# RUN python download_models.py

# 5. 暴露端口
EXPOSE 8000

# 6. 设置挂载点 (用于持久化声纹数据库)
VOLUME /app/voiceprint_db

# 7. 启动服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
