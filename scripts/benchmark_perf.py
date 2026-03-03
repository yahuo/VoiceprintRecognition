#!/usr/bin/env python3
"""
性能优化 Round 2 基准测试

对比优化前后的纯计算性能差异（不需要加载模型）:
1. match_speaker (Python循环) vs match_speaker_fast (向量化)
2. bytes += data (O(N²)) vs bytearray.extend(data) (O(N))
"""

import time
import numpy as np

# ========== 1. 声纹匹配：循环 vs 向量化 ==========

def cosine_similarity(emb1, emb2):
    """原始逐对计算"""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def match_speaker_old(embedding, registered, threshold=0.30):
    """优化前：Python for 循环逐个算余弦相似度"""
    if not registered:
        return ("未知", 0.0)
    best_name = "未知"
    best_score = 0.0
    for name, reg_emb in registered.items():
        score = cosine_similarity(embedding, reg_emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= threshold:
        return (best_name, best_score)
    return ("未知", best_score)


def match_speaker_fast(embedding, emb_matrix, emb_names, threshold=0.30):
    """优化后：预归一化矩阵 + 单次 np.dot"""
    if emb_matrix is None or len(emb_names) == 0:
        return ("未知", 0.0)
    q = embedding.flatten()
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return ("未知", 0.0)
    q = q / q_norm
    scores = emb_matrix @ q
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return (emb_names[best_idx], best_score)
    return ("未知", best_score)


def benchmark_speaker_matching():
    print("=" * 60)
    print("声纹匹配性能对比: Python循环 vs 向量化矩阵运算")
    print("=" * 60)

    dim = 192  # CAM++ embedding 维度
    iterations = 1000

    for n_speakers in [5, 10, 20, 50, 100]:
        # 构造模拟数据
        registered = {}
        for i in range(n_speakers):
            registered[f"speaker_{i}"] = np.random.randn(dim).astype(np.float32)

        # 预归一化矩阵（优化后的预处理）
        emb_names = list(registered.keys())
        matrix = np.array([registered[n] for n in emb_names])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = matrix / norms

        query = np.random.randn(dim).astype(np.float32)

        # 验证结果一致性
        old_name, old_score = match_speaker_old(query, registered)
        new_name, new_score = match_speaker_fast(query, emb_matrix, emb_names)
        assert old_name == new_name, f"结果不一致: {old_name} vs {new_name}"
        assert abs(old_score - new_score) < 1e-5, f"分数不一致: {old_score} vs {new_score}"

        # 计时: 旧方法
        t0 = time.perf_counter()
        for _ in range(iterations):
            match_speaker_old(query, registered)
        old_time = time.perf_counter() - t0

        # 计时: 新方法
        t0 = time.perf_counter()
        for _ in range(iterations):
            match_speaker_fast(query, emb_matrix, emb_names)
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time
        print(f"  {n_speakers:3d} 人 | 旧: {old_time*1000:.1f}ms | 新: {new_time*1000:.1f}ms | "
              f"加速 {speedup:.1f}x ({iterations} 次调用)")


# ========== 2. bytes += vs bytearray.extend ==========

def benchmark_buffer():
    print()
    print("=" * 60)
    print("音频缓冲区性能对比: bytes += vs bytearray.extend")
    print("=" * 60)

    chunk_size = 2048  # 模拟每次 WebSocket 收到的数据大小

    for n_chunks in [100, 500, 1000, 3000]:
        chunks = [bytes(np.random.bytes(chunk_size)) for _ in range(n_chunks)]
        total_bytes = n_chunks * chunk_size

        # bytes +=
        t0 = time.perf_counter()
        buf = b""
        for c in chunks:
            buf += c
        old_time = time.perf_counter() - t0

        # bytearray.extend
        t0 = time.perf_counter()
        buf2 = bytearray()
        for c in chunks:
            buf2.extend(c)
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time if new_time > 0 else float('inf')
        print(f"  {n_chunks:4d} 块 ({total_bytes/1024:.0f}KB) | "
              f"bytes+=: {old_time*1000:.2f}ms | bytearray: {new_time*1000:.2f}ms | "
              f"加速 {speedup:.1f}x")


if __name__ == "__main__":
    print("性能优化 Round 2 基准测试")
    print()
    benchmark_speaker_matching()
    benchmark_buffer()
    print()
    print("注意: ASR并行、Pyannote内存直传等优化需要加载模型才能测试")
