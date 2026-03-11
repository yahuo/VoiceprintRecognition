#!/usr/bin/env python3
"""
会议记录工具 (Meeting Transcription)
上传会议音频 → 自动识别并输出带姓名的文字记录

策略：
1. 使用 VAD 模型显式切分音频
2. 循环处理每个片段：ASR 识别文本 + CAM++ 识别说话人
3. 聚合结果，生成会议纪要
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import librosa

# 导入核心模块
from app.core import (
    CONFIG,
    ModelService,
    format_time,
    merge_diarization_segments,
)


def process_meeting(service: ModelService, audio_path: str, 
                    threshold: float = None) -> list:
    """
    处理会议音频
    
    策略优先级：
    1. 尝试使用 pyannote 进行说话人分离 (更准确)
    2. 回退到 VAD 分段 + DBSCAN 聚类 (原方案)
    
    Args:
        service: ModelService 实例
        audio_path: 音频文件路径
        threshold: 声纹匹配阈值
    
    Returns:
        transcript 列表
    """
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    print(f"\n正在处理会议录音: {audio_path}")
    print("-" * 60)
    
    # 读取音频
    speech_full, sr = librosa.load(audio_path, sr=16000)
    
    # ========== 尝试使用 pyannote diarization ==========
    diarization_segments = service.diarize(audio_path, audio_data=speech_full)
    
    if diarization_segments:
        # 合并同一说话人的相邻碎片段
        diarization_segments = merge_diarization_segments(diarization_segments)
        # 使用 pyannote 分段结果
        return _process_with_diarization(
            service, audio_path, speech_full, sr, 
            diarization_segments, threshold
        )
    else:
        # 回退到 VAD 分段
        return _process_with_vad(
            service, audio_path, speech_full, sr, threshold
        )


def _process_with_diarization(service: ModelService, audio_path: str,
                               speech_full: np.ndarray, sr: int,
                               segments: list, threshold: float) -> list:
    """
    使用 pyannote diarization 结果处理会议
    
    Args:
        segments: [(start_ms, end_ms, speaker_id), ...]
    """
    print(f"Step 1: 使用 pyannote 分离结果 ({len(segments)} 个片段)")
    
    transcript = []
    total_segments = len(segments)
    
    # 建立 pyannote speaker_id -> 最终说话人名 的映射
    speaker_mapping = {}  # "SPEAKER_00" -> "张三" 或 "陌生人1"
    stranger_counter = 0
    
    print("Step 2: 逐段识别文本与匹配声纹（并行推理）...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (start_ms, end_ms, pyannote_speaker) in enumerate(segments):
            print(f"\r处理片段 {i+1}/{total_segments} [{format_time(start_ms)}]", end="", flush=True)

            # 提取片段
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            speech = speech_full[start_sample:end_sample]

            if len(speech) < 0.2 * sr:
                continue

            # 直接传 numpy 数组给模型，避免临时文件 IO
            need_embedding = pyannote_speaker not in speaker_mapping
            future_text = pool.submit(service.transcribe_segment, speech)
            if need_embedding:
                future_emb = pool.submit(service.extract_embedding, speech)

            text = future_text.result()
            if not text:
                continue

            # 确定说话人
            if pyannote_speaker in speaker_mapping:
                # 已经确定过这个说话人
                speaker = speaker_mapping[pyannote_speaker]
                confidence = 1.0
            else:
                # 首次遇到这个说话人，尝试匹配已注册声纹
                try:
                    emb = future_emb.result()
                    if emb is not None:
                        matched_name, score = service.match_speaker_fast(
                            emb, threshold
                        )
                        if matched_name != "未知":
                            speaker_mapping[pyannote_speaker] = matched_name
                            speaker = matched_name
                            confidence = score
                        else:
                            stranger_counter += 1
                            stranger_name = f"陌生人{stranger_counter}"
                            speaker_mapping[pyannote_speaker] = stranger_name
                            speaker = stranger_name
                            confidence = 1.0
                    else:
                        stranger_counter += 1
                        stranger_name = f"陌生人{stranger_counter}"
                        speaker_mapping[pyannote_speaker] = stranger_name
                        speaker = stranger_name
                        confidence = 1.0
                except Exception:
                    stranger_counter += 1
                    stranger_name = f"陌生人{stranger_counter}"
                    speaker_mapping[pyannote_speaker] = stranger_name
                    speaker = stranger_name
                    confidence = 1.0

            segment_info = {
                "time": format_time(start_ms),
                "speaker": speaker,
                "confidence": round(confidence, 2),
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }

            transcript.append(segment_info)

    print(f"\n✅ 处理完成! 识别出 {len(speaker_mapping)} 位说话人")
    for pyannote_id, name in speaker_mapping.items():
        print(f"   {pyannote_id} -> {name}")
    
    return transcript


def _process_with_vad(service: ModelService, audio_path: str,
                      speech_full: np.ndarray, sr: int,
                      threshold: float) -> list:
    """
    使用 VAD 分段 + 后聚类方案处理会议 (fallback)
    """
    print("Step 1: 正在进行 VAD 切分...")
    segments = service.vad_segment(audio_path)
    
    if not segments:
        print("VAD 未返回切分结果，尝试使用默认分段...")
        dur = librosa.get_duration(filename=audio_path)
        dur_ms = int(dur * 1000)
        segments = [[t, min(t+10000, dur_ms)] for t in range(0, dur_ms, 10000)]
    
    print(f"获得 {len(segments)} 个语音片段")
    
    transcript = []
    total_segments = len(segments)
    
    print("Step 2: 逐段识别文本与说话人（并行推理）...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, seg in enumerate(segments):
            start_ms, end_ms = seg
            print(f"\r处理片段 {i+1}/{total_segments} [{format_time(start_ms)}]", end="", flush=True)

            # 提取片段
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            speech = speech_full[start_sample:end_sample]

            if len(speech) < 0.2 * sr:
                continue

            # 直接传 numpy 数组给模型，避免临时文件 IO
            future_text = pool.submit(service.transcribe_segment, speech)
            future_emb = pool.submit(service.extract_embedding, speech)

            text = future_text.result()
            emb = future_emb.result()
            if not text:
                continue

            speaker = "未知"
            score = 0.0

            # 第一阶段：尝试匹配已注册声纹
            if emb is not None:
                speaker, score = service.match_speaker_fast(emb, threshold)

            segment_info = {
                "time": format_time(start_ms),
                "speaker": speaker,
                "confidence": round(score, 2),
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "embedding": emb  # 暂存 embedding 用于后续聚类
            }

            transcript.append(segment_info)
    
    # 💥 第二阶段：对陌生人进行聚类 (Diarization)
    from app.core import cluster_embeddings
    
    # 1. 收集所有"未知"且有声纹的片段
    unknown_indices = []
    unknown_embeddings = []
    
    for i, item in enumerate(transcript):
        if item["speaker"] == "未知" and item.get("embedding") is not None:
            unknown_indices.append(i)
            unknown_embeddings.append(item["embedding"])
    
    # 2. 如果未知片段足够多，执行聚类
    if len(unknown_embeddings) >= 2:
        print(f"\n检测到 {len(unknown_embeddings)} 个未知片段，正在进行聚类分析...")
        try:
            # 自动聚类
            labels = cluster_embeddings(unknown_embeddings)
            
            # 3. 将聚类结果回填
            cluster_map = {}  # label -> "陌生人 X"
            next_stranger_id = 1
            
            for idx, label in zip(unknown_indices, labels):
                if label not in cluster_map:
                    cluster_map[label] = f"陌生人{next_stranger_id}"
                    next_stranger_id += 1
                
                transcript[idx]["speaker"] = cluster_map[label]
                transcript[idx]["confidence"] = 1.0  # 聚类结果置信度设为1
                
            print(f"✅ 成功分离出 {len(cluster_map)} 位陌生人")
            
        except Exception as e:
            print(f"聚类失败: {e}")
            
    # 清理 embedding 数据 (不返回给前端)
    for item in transcript:
        if "embedding" in item:
            del item["embedding"]
            
    print("\n处理完成!")
    return transcript



def print_transcript(transcript: list):
    """打印会议记录"""
    print("\n" + "=" * 60)
    print("【会议记录】")
    print("=" * 60 + "\n")
    
    current_speaker = None
    
    for item in transcript:
        speaker = item["speaker"]
        time = item["time"]
        text = item["text"]
        
        # 只有当说话人变化时才打印新标题
        if speaker != current_speaker:
            if current_speaker is not None:
                print()
            print(f"[{time}] {speaker}:")
            current_speaker = speaker
        
        print(f"    {text}")
    
    print("\n" + "=" * 60)


def export_markdown(transcript: list, output_path: str, audio_path: str):
    """导出为 Markdown 文件"""
    
    speakers = set(item["speaker"] for item in transcript)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 会议记录\n\n")
        f.write(f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **音频文件**: {os.path.basename(audio_path)}\n")
        f.write(f"- **参会人**: {', '.join(sorted(speakers))}\n\n")
        f.write("---\n\n")
        f.write("## 会议内容\n\n")
        
        current_speaker = None
        
        for item in transcript:
            speaker = item["speaker"]
            time = item["time"]
            text = item["text"]
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                f.write(f"**[{time}] {speaker}**:\n\n")
                current_speaker = speaker
            
            f.write(f"> {text}\n")
        
        f.write("\n---\n\n")
        f.write("*由 FunASR 声纹识别系统自动生成*\n")
    
    print(f"\n✅ 已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="会议记录工具")
    parser.add_argument("--audio", "-a", required=True, help="会议音频文件路径")
    parser.add_argument("--output", "-o", help="输出 Markdown 文件路径")
    parser.add_argument("--threshold", "-t", type=float, default=None, 
                        help=f"声纹匹配阈值 (默认: {CONFIG['speaker_threshold']})")
    parser.add_argument("--device", "-d", default="cpu", help="运行设备 (默认: cpu)")
    args = parser.parse_args()
    
    # 创建模型服务
    service = ModelService()
    service.load_models(device=args.device, load_vad=True)
    
    # 处理会议录音
    transcript = process_meeting(
        service, 
        args.audio, 
        threshold=args.threshold
    )
    
    # 打印结果
    if transcript:
        print_transcript(transcript)
        
        # 导出 Markdown
        if args.output:
            export_markdown(transcript, args.output, args.audio)
    else:
        print("未能生成会议记录")


if __name__ == "__main__":
    main()
