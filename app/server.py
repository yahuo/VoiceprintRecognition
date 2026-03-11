#!/usr/bin/env python3
"""
声纹识别会议系统 - 服务端 API
提供 RESTful API 和 WebSocket 接口，支持声纹注册、会议转写和实时识别

启动方式: uvicorn server:app --reload
或: python server.py
"""

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
import tempfile
import time
import numpy as np
from typing import Dict
from datetime import datetime
import librosa

# 导入核心模块
from .core import (
    CONFIG,
    VOICEPRINT_DB_DIR,
    ModelService,
    SpeakerTracker,
    load_voiceprint_index,
    save_voiceprint_index,
    format_time,
    merge_diarization_segments,
)


app = FastAPI(
    title="Voiceprint Meeting System API",
    description="基于 Fun-ASR-Nano + CAM++ 的智能会议记录系统",
    version="2.0.0"
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/client")
async def client():
    return FileResponse(os.path.join(static_dir, "web_client.html"))

# 允许跨域
# 允许跨域
# 恢复为通配符模式，方便各种 IP 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, # 保持 True 以防万一前端需要
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 全局服务实例 ==========

service = ModelService()
DEVICE = "cpu"  # 默认设备，可通过命令行参数修改


# ========== 生命周期 ==========

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    print("正在初始化服务端模型...")
    service.load_models(device=DEVICE, load_vad=True)
    # 加载 pyannote diarization 模型 (可选)
    # 使用 service.device：若 CUDA 不可用，load_models 已回退到 cpu
    service.load_diarization_model(device=service.device)


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


@app.post("/v1/meeting/summarize")
async def summarize_meeting_api(request: Request):
    """
    生成会议总结
    
    Body: { "transcript": [{"speaker": "...", "text": "...", "time": "..."}, ...] }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是合法的 JSON")
    
    transcript = body.get("transcript", [])
    
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript 不能为空")
    
    # 可选：允许前端覆盖 LLM 参数
    base_url = body.get("base_url", None)
    api_key = body.get("api_key", None)
    model = body.get("model", None)
    
    from .services.summarizer import summarize_meeting
    result = summarize_meeting(
        transcript_items=transcript,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    
    if result["status"] == "error":
        return JSONResponse(status_code=502, content=result)
    
    return result


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
        # 使用 services/meeting.py 中的处理逻辑 (包含聚类功能)
        from .services.meeting import process_meeting, export_markdown

        # 处理会议录音
        transcript = await asyncio.to_thread(process_meeting, service, audio_path, threshold)
        
        # 生成 Markdown
        # 模拟 export_markdown 的逻辑，但返回字符串
        md_lines = [
            "# 会议记录\n",
            f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"- **音频文件**: {file.filename}\n",
            "---\n\n## 会议内容\n"
        ]
        
        current_speaker = None
        for item in transcript:
            speaker = item["speaker"]
            time_str = item["time"]
            text = item["text"]
            
            if speaker != current_speaker:
                md_lines.append(f"\n**[{time_str}] {speaker}**:\n")
                current_speaker = speaker
            md_lines.append(f"> {text}\n")
        
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
            # 读取音频
            speech_full, sr = await asyncio.to_thread(librosa.load, audio_path, sr=16000)

            # ========== 尝试使用 pyannote diarization ==========
            print(f"🔍 尝试 pyannote diarization, pipeline loaded: {service.diarization_pipeline is not None}")
            diarization_segments = await asyncio.to_thread(service.diarize, audio_path, speech_full)
            print(f"🔍 diarization 结果: {diarization_segments is not None}, 片段数: {len(diarization_segments) if diarization_segments else 0}")

            if diarization_segments and len(diarization_segments) > 0:
                # 不再合并相邻片段：逐段声纹匹配下，短段声纹更纯净、匹配更准确

                yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(diarization_segments), 'method': 'pyannote'})}\n\n"

                # pyannote speaker -> 陌生人编号（仅用于未匹配注册人的片段）
                stranger_mapping = {}
                stranger_counter = 0

                for i, (start_ms, end_ms, pyannote_speaker) in enumerate(diarization_segments):
                    # 提取片段
                    start_sample = int(start_ms / 1000 * sr)
                    end_sample = int(end_ms / 1000 * sr)
                    speech = speech_full[start_sample:end_sample]

                    if len(speech) < 0.2 * sr:
                        continue

                    # 直接传 numpy 数组给模型，避免临时文件 IO（Docker overlay 很慢）
                    text, emb = await asyncio.gather(
                        asyncio.to_thread(service.transcribe_segment, speech),
                        asyncio.to_thread(service.extract_embedding, speech),
                    )

                    if not text:
                        continue

                    # 每段独立用 CAM++ 匹配注册声纹
                    speaker = "未知"
                    confidence = 0.0
                    if emb is not None:
                        speaker, confidence = service.match_speaker_fast(emb, threshold)

                    # 未匹配到注册人时，用 pyannote speaker 分组做陌生人聚类
                    if speaker == "未知":
                        if pyannote_speaker not in stranger_mapping:
                            stranger_counter += 1
                            stranger_mapping[pyannote_speaker] = f"陌生人{stranger_counter}"
                        speaker = stranger_mapping[pyannote_speaker]

                    # 发送结果
                    result = {
                        "type": "segment",
                        "index": i,
                        "time": format_time(start_ms),
                        "speaker": speaker,
                        "confidence": round(confidence, 2),
                        "text": text
                    }
                    yield f"data: {json_module.dumps(result, ensure_ascii=False)}\n\n"

                    await asyncio.sleep(0)

            else:
                # ========== Fallback: VAD ==========
                segments = await asyncio.to_thread(service.vad_segment, audio_path)

                if not segments:
                    dur = await asyncio.to_thread(librosa.get_duration, filename=audio_path)
                    dur_ms = int(dur * 1000)
                    segments = [[t, min(t+10000, dur_ms)] for t in range(0, dur_ms, 10000)]

                yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(segments), 'method': 'vad'})}\n\n"

                for i, seg in enumerate(segments):
                    start_ms, end_ms = seg

                    start_sample = int(start_ms / 1000 * sr)
                    end_sample = int(end_ms / 1000 * sr)
                    speech = speech_full[start_sample:end_sample]

                    if len(speech) < 0.2 * sr:
                        continue

                    # 直接传 numpy 数组给模型，避免临时文件 IO
                    text, emb = await asyncio.gather(
                        asyncio.to_thread(service.transcribe_segment, speech),
                        asyncio.to_thread(service.extract_embedding, speech),
                    )
                    if not text:
                        continue

                    speaker = "未知"
                    score = 0.0
                    if emb is not None:
                        speaker, score = service.match_speaker_fast(emb, threshold)

                    result = {
                        "type": "segment",
                        "index": i,
                        "time": format_time(start_ms),
                        "speaker": speaker,
                        "confidence": round(score, 2),
                        "text": text
                    }
                    yield f"data: {json_module.dumps(result, ensure_ascii=False)}\n\n"

                    await asyncio.sleep(0)

            # 发送完成信号
            yield f"data: {json_module.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json_module.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

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
@app.websocket("/ws/meeting/live")
async def websocket_live(websocket: WebSocket):
    """
    实时会议 WebSocket 接口
    
    客户端发送音频流 (bytes)，服务端返回识别结果 (JSON)
    """
    await websocket.accept()
    print("WebSocket 连接建立")
    
    audio_buffer = bytearray()
    is_speaking = False
    silence_duration = 0
    min_segment_bytes = 16000  # 约 0.5 秒，16kHz * 16bit * 1ch

    # 使用 SpeakerTracker 进行说话人追踪
    tracker = SpeakerTracker()
    segment_queue = asyncio.Queue()

    async def process_segment_worker():
        """后台处理已切分片段，避免阻塞 WebSocket 收包循环。"""
        while True:
            audio_chunk = await segment_queue.get()
            if audio_chunk is None:
                segment_queue.task_done()
                break

            try:
                segment_duration = len(audio_chunk) / (16000 * 2)
                queue_size = segment_queue.qsize()
                started_at = time.perf_counter()
                print(
                    f"🎙️ WebSocket 片段开始处理: "
                    f"duration={segment_duration:.2f}s, queue_size={queue_size}"
                )

                text, emb = await asyncio.gather(
                    asyncio.to_thread(service.transcribe_segment, audio_chunk),
                    asyncio.to_thread(service.extract_embedding, audio_chunk),
                )

                if text:
                    speaker = "未知"
                    score = 0.0

                    if emb is not None:
                        speaker, score = service.match_speaker_fast(emb)
                        speaker, score = tracker.update(
                            speaker, score, service.registered_embeddings
                        )

                    # 过滤置信度极低的结果
                    if not (speaker != "未知" and score < CONFIG["min_confidence"]):
                        await websocket.send_json({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "speaker": speaker,
                            "confidence": round(score, 2),
                            "text": text
                        })
                        elapsed = time.perf_counter() - started_at
                        print(
                            f"✅ WebSocket 片段处理完成: "
                            f"duration={segment_duration:.2f}s, elapsed={elapsed:.3f}s, "
                            f"speaker={speaker}, confidence={score:.2f}"
                        )
                else:
                    elapsed = time.perf_counter() - started_at
                    print(
                        f"ℹ️ WebSocket 片段无有效文本: "
                        f"duration={segment_duration:.2f}s, elapsed={elapsed:.3f}s"
                    )
            except Exception as e:
                print(f"WebSocket 片段处理错误: {e}")
            finally:
                segment_queue.task_done()

    worker_task = asyncio.create_task(process_segment_worker())

    try:
        while True:
            data = await websocket.receive_bytes()

            # 简单的静音检测逻辑
            audio_np = np.frombuffer(data, dtype=np.int16)
            energy = np.abs(audio_np).mean()

            if energy > CONFIG["silence_energy"]:
                is_speaking = True
                silence_duration = 0
            else:
                if is_speaking:
                    silence_duration += len(data) / (16000 * 2)  # 16kHz, 16bit

            # 只有检测到开始说话后才持续累积音频，避免把纯静音/环境噪声送进 ASR
            if is_speaking:
                audio_buffer.extend(data)

            # 如果静音超过阈值且有足够长的音频，则处理
            if is_speaking and silence_duration > CONFIG["silence_duration"]:
                if len(audio_buffer) >= min_segment_bytes:
                    segment_duration = len(audio_buffer) / (16000 * 2)
                    await segment_queue.put(bytes(audio_buffer))
                    print(
                        f"📥 WebSocket 片段入队: "
                        f"duration={segment_duration:.2f}s, queue_size={segment_queue.qsize()}"
                    )

                # 重置缓冲区
                audio_buffer = bytearray()
                is_speaking = False
                silence_duration = 0
                
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        if is_speaking and len(audio_buffer) >= min_segment_bytes:
            segment_duration = len(audio_buffer) / (16000 * 2)
            await segment_queue.put(bytes(audio_buffer))
            print(
                f"📥 WebSocket 尾片段入队: "
                f"duration={segment_duration:.2f}s, queue_size={segment_queue.qsize()}"
            )
        await segment_queue.put(None)
        await worker_task
        print("WebSocket 连接关闭")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voiceprint Meeting System Server")
    parser.add_argument("--device", "-d", default="cpu", help="运行设备 (cpu, cuda:0, mps)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", "-p", type=int, default=8000, help="监听端口")
    args = parser.parse_args()
    
    # 使用全局变量传递 device
    DEVICE = args.device
    
    uvicorn.run(app, host=args.host, port=args.port)
