#!/usr/bin/env python3
"""
声纹注册与识别示例 (Voiceprint Registration & Identification)
支持注册多个说话人的声纹，然后识别未知音频属于哪个说话人

使用方法:
    # 注册声纹
    python voiceprint.py register --name "张三" --audio zhangsan.wav
    
    # 识别说话人
    python voiceprint.py identify --audio unknown.wav
    
    # 列出已注册的声纹
    python voiceprint.py list
    
    # 删除声纹
    python voiceprint.py delete --name "张三"
"""

import argparse
import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple

# 添加 Fun-ASR 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Fun-ASR"))

from funasr import AutoModel


# 声纹数据库目录
VOICEPRINT_DB_DIR = os.path.join(os.path.dirname(__file__), "voiceprint_db")
VOICEPRINT_INDEX_FILE = os.path.join(VOICEPRINT_DB_DIR, "index.json")


def ensure_db_dir():
    """确保声纹数据库目录存在"""
    if not os.path.exists(VOICEPRINT_DB_DIR):
        os.makedirs(VOICEPRINT_DB_DIR)


def load_index() -> Dict[str, str]:
    """加载声纹索引（name -> embedding_file）"""
    if os.path.exists(VOICEPRINT_INDEX_FILE):
        with open(VOICEPRINT_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(index: Dict[str, str]):
    """保存声纹索引"""
    ensure_db_dir()
    with open(VOICEPRINT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def create_model(device: str = "cpu"):
    """创建说话人验证模型"""
    print("正在加载声纹模型...")
    
    model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=device,
        disable_update=True,
    )
    
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
            return np.array(embedding)
    
    raise ValueError(f"无法从音频提取声纹特征: {audio_path}")


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算余弦相似度"""
    # 展平数组确保是 1D
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def register_voiceprint(model, name: str, audio_path: str):
    """注册声纹"""
    
    print(f"\n注册声纹: {name}")
    print(f"音频文件: {audio_path}")
    print("-" * 50)
    
    # 提取声纹特征
    embedding = extract_embedding(model, audio_path)
    
    # 保存声纹
    ensure_db_dir()
    embedding_file = os.path.join(VOICEPRINT_DB_DIR, f"{name}.npy")
    np.save(embedding_file, embedding)
    
    # 更新索引
    index = load_index()
    index[name] = embedding_file
    save_index(index)
    
    print(f"✅ 声纹注册成功！")
    print(f"   声纹维度: {embedding.shape}")
    print(f"   保存位置: {embedding_file}")


def identify_speaker(model, audio_path: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
    """
    识别音频中的说话人
    
    Returns:
        按相似度排序的 (name, score) 列表
    """
    
    print(f"\n识别说话人: {audio_path}")
    print("-" * 50)
    
    # 加载索引
    index = load_index()
    
    if not index:
        print("⚠️ 声纹数据库为空，请先注册声纹")
        return []
    
    # 提取待识别音频的声纹
    query_embedding = extract_embedding(model, audio_path)
    
    # 与所有已注册声纹比对
    results = []
    for name, embedding_file in index.items():
        if os.path.exists(embedding_file):
            registered_embedding = np.load(embedding_file)
            score = cosine_similarity(query_embedding, registered_embedding)
            results.append((name, score))
    
    # 按相似度排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def list_voiceprints():
    """列出所有已注册的声纹"""
    
    index = load_index()
    
    print("\n【已注册的声纹】")
    print("=" * 50)
    
    if not index:
        print("（空）")
        return
    
    for i, (name, embedding_file) in enumerate(index.items(), 1):
        exists = "✅" if os.path.exists(embedding_file) else "❌ 文件丢失"
        print(f"{i}. {name} {exists}")
    
    print(f"\n共 {len(index)} 个声纹")


def delete_voiceprint(name: str):
    """删除声纹"""
    
    index = load_index()
    
    if name not in index:
        print(f"❌ 未找到声纹: {name}")
        return
    
    # 删除文件
    embedding_file = index[name]
    if os.path.exists(embedding_file):
        os.remove(embedding_file)
    
    # 更新索引
    del index[name]
    save_index(index)
    
    print(f"✅ 已删除声纹: {name}")


def main():
    parser = argparse.ArgumentParser(description="声纹注册与识别")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 注册命令
    register_parser = subparsers.add_parser("register", help="注册声纹")
    register_parser.add_argument("--name", "-n", required=True, help="说话人姓名")
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
    delete_parser.add_argument("--name", "-n", required=True, help="说话人姓名")
    
    args = parser.parse_args()
    
    if args.command == "register":
        model = create_model(device=args.device)
        register_voiceprint(model, args.name, args.audio)
        
    elif args.command == "identify":
        model = create_model(device=args.device)
        results = identify_speaker(model, args.audio, args.threshold)
        
        if results:
            print("\n【识别结果】")
            print("=" * 50)
            
            best_name, best_score = results[0]
            
            for i, (name, score) in enumerate(results[:args.top], 1):
                marker = "👤" if score >= args.threshold else "  "
                print(f"{marker} {i}. {name}: {score:.4f}")
            
            print("-" * 50)
            if best_score >= args.threshold:
                print(f"✅ 最佳匹配: {best_name} (相似度: {best_score:.4f})")
            else:
                print(f"❌ 未找到匹配的说话人（最高相似度: {best_score:.4f} < 阈值 {args.threshold}）")
        
    elif args.command == "list":
        list_voiceprints()
        
    elif args.command == "delete":
        delete_voiceprint(args.name)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
