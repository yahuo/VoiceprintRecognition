#!/usr/bin/env python3
"""
说话人分离示例 (Speaker Diarization)
识别音频中有哪些说话人，以及每个人说了什么

基于 Fun-ASR-Nano-2512 模型
参考: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512

使用方法:
    python diarization.py --audio your_audio.wav
    python diarization.py --audio your_audio.wav --language 英文 --device cuda:0
"""

import argparse
import os
import sys

# 添加 Fun-ASR 目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Fun-ASR"))

from funasr import AutoModel


def create_model(device: str = "cpu"):
    """
    创建 FunASR 模型（使用 Fun-ASR-Nano-2512 端到端大模型）
    
    Args:
        device: 运行设备，如 "cuda:0" 或 "cpu"
    """
    print("正在加载模型（首次运行需要下载，请耐心等待）...")
    
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    
    # 获取 Fun-ASR 目录中的 model.py 路径
    fun_asr_dir = os.path.join(PROJECT_ROOT, "Fun-ASR")
    model_py_path = os.path.join(fun_asr_dir, "model.py")
    
    # 使用 Fun-ASR-Nano 端到端大模型
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code=model_py_path,  # Fun-ASR 仓库中的 model.py
        # 说话人嵌入模型（CAM++ 声纹模型）
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=device,
        disable_update=True,
    )
    
    print("模型加载完成！")
    return model


def process_audio(model, audio_path: str, language: str = "中文"):
    """
    处理音频文件，返回说话人分离结果
    
    Args:
        model: FunASR 模型
        audio_path: 音频文件路径
        language: 语言，如 "中文"、"英文"、"日文" 等
    """
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    print(f"\n正在处理音频: {audio_path}")
    print(f"语言: {language}")
    print("-" * 50)
    
    # 执行识别
    result = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        language=language,
        itn=True,  # 逆文本正则化（数字、日期等格式化）
        return_spk_res=True,  # 返回说话人识别结果
    )
    
    return result


def format_result(result):
    """格式化输出结果"""
    
    if not result:
        print("未检测到语音内容")
        return
    
    for item in result:
        # 获取识别文本
        text = item.get("text", "")
        
        # 获取说话人分离结果
        sentence_info = item.get("sentence_info", [])
        
        if sentence_info:
            print("\n【说话人分离结果】")
            print("=" * 60)
            
            for sent in sentence_info:
                spk_id = sent.get("spk", "未知")
                start_time = sent.get("start", 0) / 1000  # 转换为秒
                end_time = sent.get("end", 0) / 1000
                content = sent.get("text", "")
                
                print(f"[说话人 {spk_id}] {start_time:.2f}s - {end_time:.2f}s")
                print(f"    {content}")
                print()
        else:
            print("\n【识别结果】")
            print("=" * 60)
            print(text)
    
    # 统计说话人
    if result and result[0].get("sentence_info"):
        speakers = set()
        for sent in result[0]["sentence_info"]:
            speakers.add(sent.get("spk", "未知"))
        print(f"\n共检测到 {len(speakers)} 位说话人: {sorted(speakers)}")


def main():
    parser = argparse.ArgumentParser(description="说话人分离示例 (Fun-ASR-Nano-2512)")
    parser.add_argument("--audio", "-a", required=True, help="音频文件路径")
    parser.add_argument("--language", "-l", default="中文", 
                        help="语言 (中文/英文/日文 等，默认: 中文)")
    parser.add_argument("--device", "-d", default="cpu", 
                        help="运行设备 (cuda:0/cpu，默认: cpu)")
    args = parser.parse_args()
    
    # 创建模型
    model = create_model(device=args.device)
    
    # 处理音频
    result = process_audio(model, args.audio, args.language)
    
    # 格式化输出
    format_result(result)


if __name__ == "__main__":
    main()
