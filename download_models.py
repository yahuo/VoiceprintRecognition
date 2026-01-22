import os
import sys

# 将当前目录添加到 PYTHONPATH，确保能导入 app 模块
sys.path.insert(0, os.getcwd())

from app.core import ModelService

def download():
    print("开始下载模型到镜像中...")
    service = ModelService()
    # 强制全部加载，触发下载
    # 注意：device='cpu' 是安全的，构建环境通常只有 CPU
    service.load_models(device="cpu", load_vad=True)
    print("✅ 模型下载完成！")

if __name__ == "__main__":
    download()
