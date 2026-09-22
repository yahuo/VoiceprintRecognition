#!/usr/bin/env python3
"""
声纹注册与识别示例 (Voiceprint Registration & Identification)
支持注册多个说话人的声纹，然后识别未知音频属于哪个说话人

使用方法:
    # 注册声纹
    python -m app.utils.voiceprint register --name "张三" --id "external-id" --audio zhangsan.wav
    
    # 识别说话人
    python -m app.utils.voiceprint identify --audio unknown.wav
    
    # 列出已注册的声纹
    python -m app.utils.voiceprint list
    
    # 删除声纹
    python -m app.utils.voiceprint delete --id "external-id"
"""

import argparse
import os
import numpy as np
from typing import List, Tuple

# 导入核心模块 (自动配置 Fun-ASR 路径)
from app.core import (
    CONFIG,
    delete_voiceprint_by_id,
    generate_voiceprint_id,
    load_voiceprint_index,
    save_voiceprint_embedding,
    cosine_similarity,
)

from funasr import AutoModel


def create_model(device: str = "cpu"):
    """创建说话人验证模型 (仅加载 CAM++)"""
    print("正在加载声纹模型...")
    
    model_kwargs = {
        "model": "iic/speech_campplus_sv_zh-cn_16k-common",
        "device": device,
        "disable_update": True,
    }
    if CONFIG.get("spk_model_path"):
        model_kwargs["model_path"] = CONFIG["spk_model_path"]
    model = AutoModel(**model_kwargs)
    
    print("模型加载完成！")
    return model


def extract_embedding(model, audio_path: str) -> np.ndarray:
    """从音频中提取声纹嵌入向量"""
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    result = model.generate(input=audio_path)
    
    if result and len(result) > 0:
        embedding = result[0].get("spk_embedding", None)
        if embedding is not None:
            import torch
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            return np.asarray(embedding).reshape(-1)
    
    raise ValueError(f"无法从音频提取声纹特征: {audio_path}")


def register_voiceprint(model, name: str, audio_path: str, speaker_id: str | None = None):
    """注册声纹"""
    if speaker_id is None:
        speaker_id = generate_voiceprint_id()
    elif speaker_id == "":
        raise ValueError("声纹 id 不能为空")
    
    print(f"\n注册声纹: {name}")
    print(f"声纹 ID: {speaker_id}")
    print(f"音频文件: {audio_path}")
    print("-" * 50)
    
    # 提取声纹特征
    embedding = extract_embedding(model, audio_path)
    
    entry = save_voiceprint_embedding(speaker_id, name, embedding)
    
    print(f"✅ 声纹注册成功！")
    print(f"   声纹维度: {embedding.shape}")
    print(f"   保存位置: {entry['file']}")


def identify_speaker(model, audio_path: str, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    识别音频中的说话人
    
    Returns:
        按相似度排序的 (id, name, score) 列表
    """
    
    print(f"\n识别说话人: {audio_path}")
    print("-" * 50)
    
    # 加载索引
    index = load_voiceprint_index()
    
    if not index:
        print("⚠️ 声纹数据库为空，请先注册声纹")
        return []
    
    # 提取待识别音频的声纹
    query_embedding = extract_embedding(model, audio_path)
    
    # 与所有已注册声纹比对
    results = []
    for speaker_id, entry in index.items():
        embedding_file = entry.get("file")
        if embedding_file and os.path.exists(embedding_file):
            registered_embedding = np.load(embedding_file)
            score = cosine_similarity(query_embedding, registered_embedding)
            results.append((speaker_id, entry.get("name", speaker_id), score))
    
    # 按相似度排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results


def list_voiceprints():
    """列出所有已注册的声纹"""
    
    index = load_voiceprint_index()
    
    print("\n【已注册的声纹】")
    print("=" * 50)
    
    if not index:
        print("（空）")
        return
    
    for i, (speaker_id, entry) in enumerate(index.items(), 1):
        embedding_file = entry.get("file")
        exists = "✅" if embedding_file and os.path.exists(embedding_file) else "❌ 文件丢失"
        print(f"{i}. {entry.get('name', speaker_id)} ({speaker_id}) {exists}")
    
    print(f"\n共 {len(index)} 个声纹")


def delete_voiceprint(speaker_id: str):
    """删除声纹"""
    entry = delete_voiceprint_by_id(speaker_id)
    if entry is None:
        print(f"❌ 未找到声纹 ID: {speaker_id}")
        return

    print(f"✅ 已删除声纹: {entry['name']} ({entry['id']})")


def main():
    parser = argparse.ArgumentParser(description="声纹注册与识别")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 注册命令
    register_parser = subparsers.add_parser("register", help="注册声纹")
    register_parser.add_argument("--name", "-n", required=True, help="说话人姓名")
    register_parser.add_argument("--id", default=None, help="外部声纹 ID；不传则自动生成")
    register_parser.add_argument("--audio", "-a", required=True, help="音频文件路径")
    register_parser.add_argument("--device", "-d", default="cpu", help="运行设备")
    
    # 识别命令
    identify_parser = subparsers.add_parser("identify", help="识别说话人")
    identify_parser.add_argument("--audio", "-a", required=True, help="音频文件路径")
    identify_parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                                  help="判定阈值（默认 0.5）")
    identify_parser.add_argument("--top", "-k", type=int, default=3, 
                                  help="显示前 K 个结果（默认 3）")
    identify_parser.add_argument("--device", "-d", default="cpu", help="运行设备")
    
    # 列出命令
    subparsers.add_parser("list", help="列出已注册的声纹")
    
    # 删除命令
    delete_parser = subparsers.add_parser("delete", help="删除声纹")
    delete_parser.add_argument("--id", required=True, help="声纹 ID")
    
    args = parser.parse_args()
    
    if args.command == "register":
        model = create_model(device=args.device)
        register_voiceprint(model, args.name, args.audio, args.id)
        
    elif args.command == "identify":
        model = create_model(device=args.device)
        results = identify_speaker(model, args.audio, args.threshold)
        
        if results:
            print("\n【识别结果】")
            print("=" * 50)
            
            best_id, best_name, best_score = results[0]
            
            for i, (speaker_id, name, score) in enumerate(results[:args.top], 1):
                marker = "👤" if score >= args.threshold else "  "
                print(f"{marker} {i}. {name} ({speaker_id}): {score:.4f}")
            
            print("-" * 50)
            if best_score >= args.threshold:
                print(f"✅ 最佳匹配: {best_name} ({best_id}) (相似度: {best_score:.4f})")
            else:
                print(f"❌ 未找到匹配的说话人（最高相似度: {best_score:.4f} < 阈值 {args.threshold}）")
        
    elif args.command == "list":
        list_voiceprints()
        
    elif args.command == "delete":
        delete_voiceprint(args.id)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
