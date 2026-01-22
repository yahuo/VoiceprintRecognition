#!/usr/bin/env python3
"""
说话人验证示例 (Speaker Verification)
比对两段音频是否来自同一说话人

使用方法:
    python verification.py --audio1 speaker1.wav --audio2 speaker2.wav
"""

import argparse
import os
import sys
import numpy as np

# 添加 Fun-ASR 目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Fun-ASR"))

from funasr import AutoModel


def create_model(device: str = "cpu"):
    """创建说话人验证模型"""
    print("正在加载说话人验证模型...")
    
    # 使用 CAM++ 说话人验证模型
    model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=device,
        disable_update=True,
    )
    
    print("模型加载完成！")
    return model


def extract_embedding(model, audio_path: str):
    """从音频中提取说话人嵌入向量（声纹特征）"""
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    print(f"提取声纹特征: {audio_path}")
    
    # 提取嵌入向量
    result = model.generate(input=audio_path)
    
    if result and len(result) > 0:
        embedding = result[0].get("spk_embedding", None)
        if embedding is not None:
            return np.array(embedding)
    
    raise ValueError(f"无法从音频提取声纹特征: {audio_path}")


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算两个嵌入向量的余弦相似度"""
    
    # 归一化
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    
    # 计算余弦相似度
    similarity = np.dot(emb1_norm, emb2_norm)
    
    return float(similarity)


def verify_speakers(model, audio1: str, audio2: str, threshold: float = 0.5):
    """
    验证两段音频是否来自同一说话人
    
    Args:
        model: 说话人验证模型
        audio1: 第一段音频路径
        audio2: 第二段音频路径
        threshold: 判定阈值（默认 0.5）
    
    Returns:
        (is_same_speaker, similarity_score)
    """
    
    # 提取两段音频的声纹特征
    print("\n" + "=" * 50)
    emb1 = extract_embedding(model, audio1)
    emb2 = extract_embedding(model, audio2)
    
    # 计算相似度
    similarity = cosine_similarity(emb1, emb2)
    
    # 判断是否同一人
    is_same = similarity >= threshold
    
    return is_same, similarity


def main():
    parser = argparse.ArgumentParser(description="说话人验证示例")
    parser.add_argument("--audio1", "-a1", required=True, help="第一段音频路径")
    parser.add_argument("--audio2", "-a2", required=True, help="第二段音频路径")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                        help="判定阈值（0-1，默认 0.5）")
    parser.add_argument("--device", "-d", default="cpu", 
                        help="运行设备 (cuda:0/cpu，默认: cpu)")
    args = parser.parse_args()
    
    # 创建模型
    model = create_model(device=args.device)
    
    # 执行验证
    is_same, score = verify_speakers(
        model, 
        args.audio1, 
        args.audio2, 
        args.threshold
    )
    
    # 输出结果
    print("\n" + "=" * 50)
    print("【说话人验证结果】")
    print("=" * 50)
    print(f"音频 1: {args.audio1}")
    print(f"音频 2: {args.audio2}")
    print(f"相似度得分: {score:.4f}")
    print(f"判定阈值: {args.threshold}")
    print("-" * 50)
    
    if is_same:
        print("✅ 判定结果: 同一说话人")
    else:
        print("❌ 判定结果: 不同说话人")
    
    # 置信度说明
    print("\n【置信度参考】")
    if score >= 0.8:
        print("🟢 高置信度: 非常可能是同一人")
    elif score >= 0.6:
        print("🟡 中等置信度: 较可能是同一人")
    elif score >= 0.4:
        print("🟠 低置信度: 不太确定")
    else:
        print("🔴 很低置信度: 很可能不是同一人")


if __name__ == "__main__":
    main()
