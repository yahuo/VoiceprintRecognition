#!/usr/bin/env python3
"""
声纹识别会议系统 - 服务端 API
提供 RESTful API 和 WebSocket 接口，支持声纹注册、会议转写和实时识别

启动方式: uvicorn server:app --reload
或: python server.py
"""

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import tempfile
import time
import numpy as np
from typing import Dict
from datetime import datetime
import wave
import librosa
import soundfile as sf

# 导入核心模块
from core import (
    CONFIG,
    VOICEPRINT_DB_DIR,
    VOICEPRINT_INDEX_FILE,
    ModelService,
    SpeakerTracker,
    load_voiceprint_index,
    save_voiceprint_index,
    load_voiceprint_embeddings,
    match_speaker,
    format_time,
)


app = FastAPI(
    title="Voiceprint Meeting System API",
    description="基于 Fun-ASR-Nano + CAM++ 的智能会议记录系统",
    version="2.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 全局服务实例 ==========

service = ModelService()


# ========== 生命周期 ==========

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    print("正在初始化服务端模型...")
    service.load_models(device="cpu", load_vad=True)


# ========== API 端点 ==========

@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "service": "Voiceprint Meeting System",
        "version": "2.0.0",
        "models_loaded": service.is_loaded,
        "registered_speakers": len(service.registered_embeddings),
        "config": CONFIG
    }


@app.get("/v1/voiceprint/list")
async def list_speakers():
    """列出已注册的声纹"""
    index = load_voiceprint_index()
    return {
        "status": "success",
        "count": len(index),
        "speakers": list(index.keys())
    }


@app.post("/v1/voiceprint/reload")
async def reload_voiceprints():
    """热重载声纹库（无需重启服务）"""
    service.reload_voiceprints()
    return {
        "status": "success",
        "message": "声纹库已重新加载",
        "count": len(service.registered_embeddings),
        "speakers": list(service.registered_embeddings.keys())
    }


@app.post("/v1/voiceprint/register")
async def register_speaker(name: str = Form(...), file: UploadFile = File(...)):
    """
    注册声纹
    
    - **name**: 说话人姓名
    - **file**: 音频文件 (WAV, MP3, M4A 等)
    """
    # 保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name
    
    try:
        # 提取声纹
        embedding = service.extract_embedding(audio_path)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="无法提取声纹特征")
        
        # 保存声纹
        os.makedirs(VOICEPRINT_DB_DIR, exist_ok=True)
        embedding_file = os.path.join(VOICEPRINT_DB_DIR, f"{name}.npy")
        np.save(embedding_file, embedding)
        
        # 更新索引
        index = load_voiceprint_index()
        index[name] = embedding_file
        save_voiceprint_index(index)
        
        # 重新加载声纹库
        service.reload_voiceprints()
        
        return {
            "status": "success",
            "message": f"声纹 '{name}' 注册成功",
            "embedding_shape": embedding.shape
        }
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.delete("/v1/voiceprint/{name}")
async def delete_speaker(name: str):
    """删除已注册的声纹"""
    index = load_voiceprint_index()
    
    if name not in index:
        raise HTTPException(status_code=404, detail=f"未找到声纹: {name}")
    
    # 删除文件
    embedding_file = index[name]
    if os.path.exists(embedding_file):
        os.remove(embedding_file)
    
    # 更新索引
    del index[name]
    save_voiceprint_index(index)
    
    # 重新加载声纹库
    service.reload_voiceprints()
    
    return {
        "status": "success",
        "message": f"已删除声纹: {name}"
    }


@app.post("/v1/meeting/transcribe")
async def transcribe_meeting(
    file: UploadFile = File(...),
    threshold: float = Form(default=None)
):
    """
    上传音频文件生成会议记录
    
    - **file**: 会议音频文件
    - **threshold**: 声纹匹配阈值 (默认使用 CONFIG 中的值)
    """
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    
    # 保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name
    
    try:
        # 1. VAD 切分
        segments = service.vad_segment(audio_path)
        
        if not segments:
            # 兜底：固定时长切分
            dur = librosa.get_duration(filename=audio_path)
            dur_ms = int(dur * 1000)
            segments = [[t, min(t+10000, dur_ms)] for t in range(0, dur_ms, 10000)]
        
        # 2. 读取音频
        speech_full, sr = librosa.load(audio_path, sr=16000)
        
        # 3. 逐段处理
        transcript = []
        
        for seg in segments:
            start_ms, end_ms = seg
            
            # 提取片段
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            speech = speech_full[start_sample:end_sample]
            
            if len(speech) < 0.2 * sr:
                continue
            
            # 保存临时片段
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_tmp:
                sf.write(seg_tmp.name, speech, sr)
                seg_path = seg_tmp.name
            
            try:
                # ASR
                text = service.transcribe_segment(seg_path)
                if not text:
                    continue
                
                # 声纹
                emb = service.extract_embedding(seg_path)
                speaker = "未知"
                score = 0.0
                if emb is not None:
                    speaker, score = match_speaker(emb, service.registered_embeddings, threshold)
                
                transcript.append({
                    "time": format_time(start_ms),
                    "speaker": speaker,
                    "confidence": round(score, 2),
                    "text": text
                })
            finally:
                if os.path.exists(seg_path):
                    os.remove(seg_path)
        
        # 4. 生成 Markdown
        md_lines = [
            "# 会议记录\n",
            f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"- **音频文件**: {file.filename}\n",
            "---\n\n## 会议内容\n"
        ]
        
        current_speaker = None
        for item in transcript:
            if item["speaker"] != current_speaker:
                md_lines.append(f"\n**[{item['time']}] {item['speaker']}**:\n")
                current_speaker = item["speaker"]
            md_lines.append(f"> {item['text']}\n")
        
        markdown = "".join(md_lines)
        
        return {
            "status": "success",
            "segments": len(transcript),
            "transcript": transcript,
            "markdown": markdown
        }
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


from fastapi.responses import StreamingResponse
import asyncio
import json as json_module


@app.post("/v1/meeting/transcribe/stream")
async def transcribe_meeting_stream(
    file: UploadFile = File(...),
    threshold: float = Form(default=None)
):
    """
    流式处理会议音频（Server-Sent Events）
    
    - **file**: 会议音频文件
    - **threshold**: 声纹匹配阈值
    
    返回 SSE 流，每个片段处理完成后立即推送
    """
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    
    # 保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name
    
    async def generate():
        try:
            # 1. VAD 切分
            segments = service.vad_segment(audio_path)
            
            if not segments:
                dur = librosa.get_duration(filename=audio_path)
                dur_ms = int(dur * 1000)
                segments = [[t, min(t+10000, dur_ms)] for t in range(0, dur_ms, 10000)]
            
            # 发送进度信息
            yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(segments)})}\n\n"
            
            # 2. 读取音频
            speech_full, sr = librosa.load(audio_path, sr=16000)
            
            # 3. 逐段处理并流式输出
            for i, seg in enumerate(segments):
                start_ms, end_ms = seg
                
                # 提取片段
                start_sample = int(start_ms / 1000 * sr)
                end_sample = int(end_ms / 1000 * sr)
                speech = speech_full[start_sample:end_sample]
                
                if len(speech) < 0.2 * sr:
                    continue
                
                # 保存临时片段
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_tmp:
                    sf.write(seg_tmp.name, speech, sr)
                    seg_path = seg_tmp.name
                
                try:
                    # ASR
                    text = service.transcribe_segment(seg_path)
                    if not text:
                        continue
                    
                    # 声纹
                    emb = service.extract_embedding(seg_path)
                    speaker = "未知"
                    score = 0.0
                    if emb is not None:
                        speaker, score = match_speaker(emb, service.registered_embeddings, threshold)
                    
                    # 立即发送结果
                    result = {
                        "type": "segment",
                        "index": i + 1,
                        "time": format_time(start_ms),
                        "speaker": speaker,
                        "confidence": round(score, 2),
                        "text": text
                    }
                    yield f"data: {json_module.dumps(result, ensure_ascii=False)}\n\n"
                    
                finally:
                    if os.path.exists(seg_path):
                        os.remove(seg_path)
                
                # 让出控制权，避免阻塞
                await asyncio.sleep(0)
            
            # 发送完成信号
            yield f"data: {json_module.dumps({'type': 'done'})}\n\n"
            
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
async def websocket_live(websocket: WebSocket):
    """
    实时会议 WebSocket 接口
    
    客户端发送音频流 (bytes)，服务端返回识别结果 (JSON)
    """
    await websocket.accept()
    print("WebSocket 连接建立")
    
    audio_buffer = b""
    silence_duration = 0
    
    # 使用 SpeakerTracker 进行说话人追踪
    tracker = SpeakerTracker()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer += data
            
            # 简单的静音检测逻辑
            audio_np = np.frombuffer(data, dtype=np.int16)
            energy = np.abs(audio_np).mean()
            
            if energy < CONFIG["silence_energy"]:
                silence_duration += len(data) / (16000 * 2)  # 16kHz, 16bit
            else:
                silence_duration = 0
            
            # 如果静音超过阈值且有足够长的音频，则处理
            if silence_duration > CONFIG["silence_duration"] and len(audio_buffer) > 16000 * 2:
                # 保存临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_buffer)
                    tmp_path = tmp.name
                
                try:
                    # ASR
                    text = service.transcribe_segment(tmp_path)
                    
                    if text:
                        # 声纹
                        emb = service.extract_embedding(tmp_path)
                        speaker = "未知"
                        score = 0.0
                        
                        if emb is not None:
                            speaker, score = match_speaker(emb, service.registered_embeddings)
                            
                            # 使用 SpeakerTracker 处理继承逻辑
                            speaker, score = tracker.update(speaker, score, service.registered_embeddings)
                        
                        # 过滤置信度极低的结果
                        if score < CONFIG["min_confidence"]:
                            continue
                        
                        await websocket.send_json({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "speaker": speaker,
                            "confidence": round(score, 2),
                            "text": text
                        })
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                # 重置缓冲区
                audio_buffer = b""
                silence_duration = 0
                
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        print("WebSocket 连接关闭")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
