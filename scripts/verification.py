#!/usr/bin/env python3
"""
说话人验证示例 (Speaker Verification)
比对两段音频是否来自同一说话人

使用方法:
    python scripts/verification.py --audio1 speaker1.wav --audio2 speaker2.wav
"""

import argparse
import os
import sys

# 复用声纹 CLI 的 CAM++ 加载/提取和核心相似度计算，不维护另一套实现。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.core import cosine_similarity
from app.utils.voiceprint import create_model, extract_embedding


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
    
    print("\n分数是余弦相似度，不是身份准确率或概率；阈值须按实际场景校准。")


if __name__ == "__main__":
    main()
