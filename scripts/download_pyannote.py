import os
import sys
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

def download_and_patch():
    # 加载环境变量
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ 错误: 未找到 HF_TOKEN。")
        print("请在 .env 文件中配置 HF_TOKEN，或设置环境变量。")
        sys.exit(1)
    
    # 定义本地目录
    project_root = os.getcwd()
    models_dir = os.path.join(project_root, "models")
    pyannote_dir = os.path.join(models_dir, "pyannote", "diarization")
    
    print(f"📦 准备下载 Pyannote 模型到: {pyannote_dir}")
    print("注意：将下载完整仓库，包含 segmentation, embedding, plda 等依赖...")
    
    # 1. 下载完整仓库
    print("\n⬇️  下载 pyannote/speaker-diarization-community-1...")
    try:
        snapshot_download(
            repo_id="pyannote/speaker-diarization-community-1",
            local_dir=pyannote_dir,
            local_dir_use_symlinks=False, # 确保主要文件是实体文件
            token=token
            # 不再通过 allow_patterns 限制，下载整个包
        )
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请确保已在 Hugging Face 接受用户协议。")
        sys.exit(1)

    print("\n✅ 模型下载完成！")
    print(f"本地路径: {pyannote_dir}")
    
    # 检查关键子文件夹是否存在
    required_subs = ["segmentation"] # embedding, plda 有时是可选或内嵌
    for sub in required_subs:
        sub_path = os.path.join(pyannote_dir, sub)
        if os.path.exists(sub_path):
            print(f"  - {sub}: Found")
        else:
            print(f"  - {sub}: ❌ MISSING (这可能导致加载失败)")

    print("\n🎉 现在可以在无网络/无 Token 环境运行了。")

if __name__ == "__main__":
    download_and_patch()
