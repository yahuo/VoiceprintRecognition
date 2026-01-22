#!/usr/bin/env python3
"""
交互式语音识别 (Interactive Mode)
模型只加载一次，可以反复处理多个音频文件

使用方法:
    python interactive.py
    python interactive.py --device cuda:0
"""

import os
import sys
import argparse

# 添加 Fun-ASR 目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Fun-ASR"))

from funasr import AutoModel


def create_model(device: str = "cpu"):
    """创建 FunASR 模型"""
    print("正在加载模型（首次加载需要一些时间）...")
    
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    fun_asr_dir = os.path.join(PROJECT_ROOT, "Fun-ASR")
    model_py_path = os.path.join(fun_asr_dir, "model.py")
    
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code=model_py_path,
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=device,
        disable_update=True,
    )
    
    print("✅ 模型加载完成！现在可以快速处理音频了。\n")
    return model


def process_audio(model, audio_path: str, language: str = "中文"):
    """处理单个音频文件"""
    
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return
    
    print(f"处理中: {audio_path}")
    
    result = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        language=language,
        itn=True,
        return_spk_res=True,
    )
    
    if result:
        text = result[0].get("text", "")
        print(f"识别结果: {text}\n")
    else:
        print("未检测到语音\n")


def interactive_loop(model):
    """交互式循环"""
    
    print("=" * 60)
    print("  交互式语音识别")
    print("  输入音频文件路径进行识别，输入 'quit' 或 'q' 退出")
    print("  输入 'lang 英文' 切换语言")
    print("=" * 60)
    
    language = "中文"
    
    while True:
        try:
            user_input = input(f"\n[{language}] 请输入音频路径: ").strip()
            
            if not user_input:
                continue
            
            # 退出命令
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("再见！")
                break
            
            # 切换语言
            if user_input.startswith('lang '):
                language = user_input[5:].strip()
                print(f"已切换到: {language}")
                continue
            
            # 处理音频
            process_audio(model, user_input, language)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="交互式语音识别")
    parser.add_argument("--device", "-d", default="cpu", 
                        help="运行设备 (cuda:0/cpu，默认: cpu)")
    args = parser.parse_args()
    
    # 加载模型（只加载一次）
    model = create_model(device=args.device)
    
    # 进入交互循环
    interactive_loop(model)


if __name__ == "__main__":
    main()
