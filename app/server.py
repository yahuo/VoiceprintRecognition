#!/usr/bin/env python3
"""
声纹识别会议系统 - 服务端 API
提供 RESTful API 和 WebSocket 接口，支持声纹注册、会议转写和实时识别

启动方式: uvicorn server:app --reload
或: python server.py
"""

from fastapi import FastAPI, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
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
from .services.recording_store import (
    CHANNELS as RECORDING_CHANNELS,
    SAMPLE_RATE as RECORDING_SAMPLE_RATE,
    recording_store,
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


def _normalize_allowed_speakers(raw_allowed_speakers: list[str] | None) -> list[str] | None:
    """
    规范化本次会议允许匹配的注册人名单。

    返回 None 表示未选择参会人；是否全库匹配由调用方决定。
    """
    if not raw_allowed_speakers:
        return None

    normalized = []
    seen = set()
    for name in raw_allowed_speakers:
        cleaned = (name or "").strip()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)

    if not normalized:
        return None

    missing = [name for name in normalized if name not in service.registered_embeddings]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"以下参会人未注册声纹: {', '.join(missing)}",
        )

    return normalized


def _selected_speaker_matching_enabled(allowed_speakers: list[str] | None) -> bool:
    """只有显式选择参会人时才跑注册声纹匹配。"""
    return bool(allowed_speakers)


class DeleteRecordingsRequest(BaseModel):
    fileIds: list[str] = Field(default_factory=list)
    reason: str | None = None
    requestId: str | None = None


def _normalize_recording_file_id(file_id: str) -> str:
    try:
        return recording_store.normalize_file_id(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法 fileId")


def _normalize_recording_file_ids(file_ids: list[str]) -> list[str]:
    if not file_ids:
        raise HTTPException(status_code=400, detail="fileIds 不能为空")
    normalized = [_normalize_recording_file_id(file_id) for file_id in file_ids]
    return list(dict.fromkeys(normalized))


def _build_meeting_markdown(transcript: list[dict], audio_filename: str) -> str:
    md_lines = [
        "# 会议记录\n",
        f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"- **音频文件**: {audio_filename}\n",
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

    return "".join(md_lines)


async def _transcribe_meeting_audio(
    audio_path: str,
    audio_filename: str,
    threshold: float | None,
    allowed_speakers: list[str] | None,
):
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    allowed_speakers = _normalize_allowed_speakers(allowed_speakers)
    should_match_speaker = _selected_speaker_matching_enabled(allowed_speakers)
    matching_scope = service.build_matching_scope(allowed_speakers) if should_match_speaker else None

    from .services.meeting import process_meeting

    transcript = await asyncio.to_thread(
        process_meeting,
        service,
        audio_path,
        threshold,
        allowed_speakers,
        matching_scope,
        should_match_speaker,
    )
    markdown = _build_meeting_markdown(transcript, audio_filename)

    return {
        "status": "success",
        "segments": len(transcript),
        "transcript": transcript,
        "markdown": markdown
    }


# ========== 生命周期 ==========

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    removed_temp_recordings = recording_store.cleanup_temp_files()
    if removed_temp_recordings:
        print(f"已清理未完成录音临时文件: {removed_temp_recordings}")
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


@app.post(
    "/v1/meeting/transcribe",
    summary="上传音频生成会议记录",
    response_description="完整会议转写结果与 Markdown",
)
async def transcribe_meeting(
    file: UploadFile = File(
        ...,
        description="会议音频文件，支持 WAV、MP3、M4A 等格式。",
    ),
    threshold: float = Form(
        default=None,
        description="可选的声纹匹配阈值；不传时使用服务端默认值。",
    ),
    allowed_speakers: list[str] | None = Form(
        default=None,
        description="可选的参会人白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。",
    ),
):
    """
    上传音频文件，返回完整会议记录。

    - `file`: 会议音频文件
    - `threshold`: 可选的声纹匹配阈值，默认使用服务端配置
    - `allowed_speakers`: 可选的参会人白名单。未传时不匹配注册声纹；传入后只在指定注册人范围内识别说话人
    """
    # 保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name

    try:
        return await _transcribe_meeting_audio(
            audio_path,
            file.filename,
            threshold,
            allowed_speakers,
        )
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


from fastapi.responses import StreamingResponse
import asyncio
import json as json_module


@app.post(
    "/v1/meeting/transcribe/stream",
    summary="流式处理会议音频",
    response_description="SSE 流，按阶段和分段持续返回转写结果",
)
async def transcribe_meeting_stream(
    file: UploadFile = File(
        ...,
        description="会议音频文件，支持 WAV、MP3、M4A 等格式。",
    ),
    threshold: float = Form(
        default=None,
        description="可选的声纹匹配阈值；不传时使用服务端默认值。",
    ),
    allowed_speakers: list[str] | None = Form(
        default=None,
        description="可选的参会人白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。",
    ),
):
    """
    流式处理会议音频（Server-Sent Events）。

    - `file`: 会议音频文件
    - `threshold`: 可选的声纹匹配阈值
    - `allowed_speakers`: 可选的参会人白名单。未传时不匹配注册声纹；传入后只在指定注册人范围内识别说话人

    返回 `text/event-stream`，会按阶段推送 `status / info / segment / done / error` 事件。
    """
    content = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    return await _stream_meeting_transcription(content, suffix, threshold, allowed_speakers)


async def _stream_meeting_transcription(
    content: bytes | None,
    suffix: str,
    threshold: float | None,
    allowed_speakers: list[str] | None,
    *,
    source_path: str | None = None,
):
    """SSE 流式转录。`content` 为上传音频字节；或传 `source_path` 直接复用磁盘上的录音。"""
    import io

    if (content is None) == (source_path is None):
        raise ValueError("must supply exactly one of content or source_path")

    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    allowed_speakers = _normalize_allowed_speakers(allowed_speakers)
    should_match_speaker = _selected_speaker_matching_enabled(allowed_speakers)
    matching_scope = service.build_matching_scope(allowed_speakers) if should_match_speaker else None

    if source_path is not None:
        audio_path = source_path
        owns_temp_file = False
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            audio_path = tmp.name
        owns_temp_file = True

    async def generate():
        try:
            t_start = time.perf_counter()
            yield f"data: {json_module.dumps({'type': 'status', 'phase': 'loading', 'message': '正在解码音频...'}, ensure_ascii=False)}\n\n"

            def _transcribe_full_audio_timed(audio_input, backend):
                started_at = time.perf_counter()
                result = service.transcribe_full_audio(audio_input, backend, True)
                return result, time.perf_counter() - started_at

            def _load_audio_from_upload():
                if content is None:
                    return librosa.load(audio_path, sr=16000)
                try:
                    return librosa.load(io.BytesIO(content), sr=16000)
                except Exception as exc:
                    print(f"⚠️ 内存解码失败，回退到临时文件: {exc}")
                    return librosa.load(audio_path, sr=16000)

            speech_full, sr = await asyncio.to_thread(_load_audio_from_upload)
            audio_duration = len(speech_full) / sr
            audio_duration_ms = int(audio_duration * 1000)
            t_load = time.perf_counter()
            print(f"⏱️ SSE 音频加载: {t_load - t_start:.2f}s, 时长: {audio_duration:.1f}s")

            full_audio_asr_task = None
            if service.upload_asr_backend == "paraformer":
                yield f"data: {json_module.dumps({'type': 'status', 'phase': 'transcribing', 'message': '正在并行进行整段识别与说话人分离...'}, ensure_ascii=False)}\n\n"
                full_audio_asr_task = asyncio.create_task(
                    asyncio.to_thread(
                        _transcribe_full_audio_timed,
                        speech_full,
                        service.upload_asr_backend,
                    )
                )

            # CUDA/MPS 设备共享，双线程 asyncio.gather 只增加调度开销无真正并行
            def _infer_segment(audio):
                text = service.transcribe_segment(audio)
                emb = service.extract_embedding(audio) if should_match_speaker else None
                return text, emb

            # ========== 尝试使用 pyannote diarization ==========
            yield f"data: {json_module.dumps({'type': 'status', 'phase': 'diarizing', 'message': '正在进行说话人分离...'}, ensure_ascii=False)}\n\n"
            diarization_segments = await asyncio.to_thread(
                service.diarize, None, speech_full
            )
            t_diarize = time.perf_counter()
            n_raw = len(diarization_segments) if diarization_segments else 0
            print(f"⏱️ SSE 说话人分离: {t_diarize - t_load:.2f}s, 原始片段数: {n_raw}")

            if diarization_segments and len(diarization_segments) > 0:
                diarization_segments = merge_diarization_segments(
                    diarization_segments,
                    gap_threshold_ms=CONFIG["diarization_merge_gap_ms"],
                    short_segment_ms=CONFIG["diarization_short_segment_ms"],
                    max_merged_duration_ms=CONFIG["diarization_max_merged_ms"],
                )
                print(
                    "⏱️ SSE 合并后片段数: "
                    f"{len(diarization_segments)} "
                    f"(gap<={CONFIG['diarization_merge_gap_ms']}ms, "
                    f"short<={CONFIG['diarization_short_segment_ms']}ms, "
                    f"max<={CONFIG['diarization_max_merged_ms']}ms)"
                )

                asr_sentences = []
                if full_audio_asr_task is not None:
                    asr_result, asr_elapsed = await full_audio_asr_task
                    asr_sentences = asr_result.get("sentences", [])
                    print(
                        f"⏱️ SSE 整段 ASR({service.upload_asr_backend}): "
                        f"{asr_elapsed:.2f}s, 句子数: {len(asr_sentences)}"
                    )
                else:
                    print(f"ℹ️ 上传 ASR 后端 {service.upload_asr_backend} 不支持时间轴对齐，直接走逐段识别")

                if asr_sentences:
                    def _choose_best_speaker(start_ms: int, end_ms: int):
                        best = None
                        best_overlap = -1
                        for seg_start, seg_end, seg_speaker in diarization_segments:
                            overlap = min(end_ms, seg_end) - max(start_ms, seg_start)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best = (seg_start, seg_end, seg_speaker)
                        return best

                    def _split_sentence_by_token_timestamps(text: str, token_timestamps):
                        if not token_timestamps:
                            return [], False

                        punctuation = set("，。！？；、,.!?:; ")
                        parts = []
                        current = None
                        pending_prefix = ""
                        token_idx = 0
                        pause_split_used = False
                        pause_gap_ms = CONFIG["asr_pause_split_gap_ms"]

                        def flush_current():
                            nonlocal current
                            if current is not None and current["text"].strip():
                                parts.append(current)
                            current = None

                        for ch in text:
                            if ch in punctuation:
                                if current is not None:
                                    current["text"] += ch
                                else:
                                    pending_prefix += ch
                                continue

                            if token_idx >= len(token_timestamps):
                                if current is not None:
                                    current["text"] += ch
                                else:
                                    pending_prefix += ch
                                continue

                            tok_start_ms, tok_end_ms = token_timestamps[token_idx]
                            token_idx += 1
                            best_seg = _choose_best_speaker(tok_start_ms, tok_end_ms)
                            gap_ms = 0
                            if current is not None:
                                gap_ms = max(0, tok_start_ms - current["end_ms"])

                            if (
                                current is not None
                                and current["seg"] == best_seg
                                and gap_ms <= pause_gap_ms
                            ):
                                current["text"] += ch
                                current["end_ms"] = tok_end_ms
                            else:
                                if current is not None and gap_ms > pause_gap_ms:
                                    pause_split_used = True
                                flush_current()
                                current = {
                                    "seg": best_seg,
                                    "text": pending_prefix + ch,
                                    "start_ms": tok_start_ms,
                                    "end_ms": tok_end_ms,
                                }
                                pending_prefix = ""

                        if pending_prefix:
                            if current is not None:
                                current["text"] += pending_prefix
                            elif parts:
                                parts[-1]["text"] += pending_prefix

                        flush_current()
                        return parts, pause_split_used

                    def _merge_split_outputs(outputs):
                        if not outputs:
                            return []

                        merged = []
                        for item in outputs:
                            if not item["text"].strip():
                                continue
                            if merged and merged[-1]["speaker"] == item["speaker"]:
                                merged[-1]["text"] += item["text"]
                                merged[-1]["end_ms"] = item["end_ms"]
                                merged[-1]["confidence"] = max(merged[-1]["confidence"], item["confidence"])
                            else:
                                merged.append(dict(item))

                        def _compact_len(text: str) -> int:
                            punctuation = set("，。！？；、,.!?:; ")
                            return sum(1 for ch in text if ch not in punctuation)

                        idx = 0
                        while idx < len(merged):
                            item = merged[idx]
                            compact_len = _compact_len(item["text"])
                            if item["speaker"] == "未知" and compact_len <= 1:
                                if idx + 1 < len(merged):
                                    merged[idx + 1]["text"] = item["text"] + merged[idx + 1]["text"]
                                    merged[idx + 1]["start_ms"] = item["start_ms"]
                                    merged.pop(idx)
                                    continue
                                if idx > 0:
                                    merged[idx - 1]["text"] += item["text"]
                                    merged[idx - 1]["end_ms"] = item["end_ms"]
                                    merged.pop(idx)
                                    idx -= 1
                                    continue
                            idx += 1

                        return merged

                    single_allowed_mode = bool(
                        matching_scope is not None
                        and matching_scope.is_restricted
                        and matching_scope.size == 1
                    )
                    selected_allowed_speaker = matching_scope.names[0] if single_allowed_mode else None
                    single_allowed_stranger = "陌生人1"
                    effective_matching_scope = None if single_allowed_mode else matching_scope
                    effective_allowed_speakers = None if single_allowed_mode else allowed_speakers

                    async def _match_registered_for_window(
                        start_ms: int,
                        end_ms: int,
                    ):
                        if not should_match_speaker:
                            return ("未知", 0.0)
                        start_sample = int(start_ms / 1000 * sr)
                        end_sample = int(end_ms / 1000 * sr)
                        speech = speech_full[start_sample:end_sample]
                        emb = await asyncio.to_thread(service.extract_embedding, speech)
                        if emb is None:
                            return ("未知", 0.0)
                        return service.match_registered_speaker_guarded(
                            emb,
                            threshold=threshold,
                            duration_ms=end_ms - start_ms,
                            match_scope=effective_matching_scope,
                            allowed_speakers=effective_allowed_speakers,
                        )

                    async def _match_registered_for_short_sentence(
                        start_ms: int,
                        end_ms: int,
                    ):
                        if not should_match_speaker:
                            return ("未知", 0.0)
                        start_sample = int(start_ms / 1000 * sr)
                        end_sample = int(end_ms / 1000 * sr)
                        speech = speech_full[start_sample:end_sample]
                        emb = await asyncio.to_thread(service.extract_embedding, speech)
                        if emb is None:
                            return ("未知", 0.0)
                        return service.match_registered_speaker_short_window(
                            emb,
                            threshold=threshold,
                            duration_ms=end_ms - start_ms,
                            match_scope=effective_matching_scope,
                            allowed_speakers=effective_allowed_speakers,
                        )

                    async def _match_registered_for_sentence(
                        start_ms: int,
                        end_ms: int,
                    ):
                        if (end_ms - start_ms) > CONFIG["offline_sentence_exact_match_max_duration_ms"]:
                            if single_allowed_mode:
                                return await _match_registered_for_window(start_ms, end_ms)
                            return ("未知", 0.0)
                        return await _match_registered_for_short_sentence(
                            start_ms,
                            end_ms,
                        )

                    async def _resolve_single_allowed_inheritance(
                        seg,
                        start_ms: int,
                        end_ms: int,
                    ):
                        if seg is None:
                            return ("未知", 0.0)

                        seg_start, seg_end, pyannote_speaker = seg
                        mapped_speaker, mapped_confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))
                        if mapped_speaker == "未知":
                            return ("未知", 0.0)

                        overlap = max(0, min(end_ms, seg_end) - max(start_ms, seg_start))
                        duration = max(1, end_ms - start_ms)
                        overlap_ratio = overlap / duration
                        if (
                            duration >= CONFIG["offline_scoped_single_inherit_min_duration_ms"]
                            and overlap > 0
                            and
                            mapped_confidence >= CONFIG["offline_scoped_single_inherit_min_confidence"]
                            and overlap_ratio >= CONFIG["offline_scoped_single_inherit_min_overlap_ratio"]
                        ):
                            verify_start_ms = max(start_ms, seg_start)
                            verify_end_ms = min(end_ms, seg_end)
                            verified_name, verified_score = await _match_registered_for_window(
                                verify_start_ms,
                                verify_end_ms,
                            )
                            if verified_name != "未知":
                                if str(mapped_speaker).startswith("陌生人"):
                                    return (verified_name, verified_score)
                                if verified_name == mapped_speaker:
                                    return (mapped_speaker, max(mapped_confidence, verified_score))
                        if str(mapped_speaker).startswith("陌生人"):
                            return (mapped_speaker, mapped_confidence)
                        return ("未知", 0.0)

                    # 每个 pyannote speaker 使用多个代表片段做保守匹配，
                    # 避免单个"最长片段"把整组句子都带偏。
                    speaker_mapping = {}
                    stranger_counter = 0
                    speaker_candidate_segments = {}
                    top_k = max(1, CONFIG["offline_registered_match_top_k"])
                    for start_ms, end_ms, pyannote_speaker in diarization_segments:
                        speaker_candidate_segments.setdefault(pyannote_speaker, []).append((start_ms, end_ms))

                    for pyannote_speaker, segments_for_speaker in speaker_candidate_segments.items():
                        candidate_segments = sorted(
                            segments_for_speaker,
                            key=lambda item: item[1] - item[0],
                            reverse=True,
                        )[:top_k]
                        candidate_embeddings = []
                        if should_match_speaker:
                            for start_ms, end_ms in candidate_segments:
                                start_sample = int(start_ms / 1000 * sr)
                                end_sample = int(end_ms / 1000 * sr)
                                speech = speech_full[start_sample:end_sample]
                                emb = await asyncio.to_thread(service.extract_embedding, speech)
                                candidate_embeddings.append((emb, end_ms - start_ms))

                        speaker = "未知"
                        confidence = 0.0
                        if candidate_embeddings:
                            speaker, confidence = service.match_registered_speaker_consensus(
                                candidate_embeddings,
                                threshold=threshold,
                                match_scope=effective_matching_scope,
                                allowed_speakers=effective_allowed_speakers,
                            )
                        if single_allowed_mode:
                            if speaker != selected_allowed_speaker:
                                speaker = single_allowed_stranger
                                confidence = 1.0
                        elif speaker == "未知":
                            stranger_counter += 1
                            speaker = f"陌生人{stranger_counter}"
                            confidence = 1.0
                        speaker_mapping[pyannote_speaker] = (speaker, confidence)

                    yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(asr_sentences), 'method': f'align-{service.upload_asr_backend}'})}\n\n"
                    yield f"data: {json_module.dumps({'type': 'status', 'phase': 'processing', 'message': f'正在对齐说话人与文本，共 {len(asr_sentences)} 句...'}, ensure_ascii=False)}\n\n"

                    for i, item in enumerate(asr_sentences):
                        start_ms = item["start_ms"]
                        end_ms = item["end_ms"]
                        text = item["text"].strip()
                        token_timestamps = item.get("token_timestamps") or []
                        if not text:
                            continue

                        split_parts, pause_split_used = _split_sentence_by_token_timestamps(text, token_timestamps)
                        distinct_part_segs = {
                            part["seg"] for part in split_parts if part.get("seg") is not None
                        }
                        if split_parts and (len(distinct_part_segs) > 1 or pause_split_used):
                            sentence_outputs = []
                            for part in split_parts:
                                part_text = part["text"].strip()
                                if not part_text:
                                    continue
                                seg = part["seg"]
                                speaker = "未知"
                                confidence = 0.0
                                locally_verified = False
                                short_name, short_score = await _match_registered_for_short_sentence(
                                    part["start_ms"],
                                    part["end_ms"],
                                )
                                if short_name != "未知":
                                    if not single_allowed_mode or short_name == selected_allowed_speaker:
                                        speaker = short_name
                                        confidence = short_score
                                        locally_verified = True
                                    elif single_allowed_mode:
                                        speaker = single_allowed_stranger
                                        confidence = 1.0
                                        locally_verified = True
                                elif single_allowed_mode:
                                    verified_name, verified_score = await _match_registered_for_window(
                                        part["start_ms"],
                                        part["end_ms"],
                                    )
                                    if verified_name == selected_allowed_speaker:
                                        speaker = verified_name
                                        confidence = verified_score
                                        locally_verified = True
                                    elif verified_name != "未知":
                                        speaker = single_allowed_stranger
                                        confidence = 1.0
                                        locally_verified = True
                                    else:
                                        if seg is not None:
                                            _, _, pyannote_speaker = seg
                                            speaker, confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))
                                elif seg is not None:
                                    _, _, pyannote_speaker = seg
                                    speaker, confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))

                                sentence_outputs.append(
                                    {
                                        "start_ms": part["start_ms"],
                                        "end_ms": part["end_ms"],
                                        "speaker": speaker,
                                        "confidence": round(confidence, 2),
                                        "text": part_text,
                                    }
                                )

                            sentence_outputs = _merge_split_outputs(sentence_outputs)
                            for part_idx, out in enumerate(sentence_outputs):
                                result = {
                                    "type": "segment",
                                    "index": i if part_idx == 0 else f"{i}-{part_idx}",
                                    "time": format_time(out["start_ms"]),
                                    "speaker": out["speaker"],
                                    "confidence": out["confidence"],
                                    "text": out["text"].strip(),
                                }
                                yield f"data: {json_module.dumps(result, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0)
                            continue

                        speaker = "未知"
                        confidence = 0.0
                        sentence_speaker, sentence_confidence = await _match_registered_for_sentence(
                            start_ms,
                            end_ms,
                        )
                        if sentence_speaker != "未知":
                            if not single_allowed_mode or sentence_speaker == selected_allowed_speaker:
                                speaker = sentence_speaker
                                confidence = sentence_confidence
                            elif single_allowed_mode:
                                speaker = single_allowed_stranger
                                confidence = 1.0
                            else:
                                best_seg = _choose_best_speaker(start_ms, end_ms)
                                if best_seg is not None:
                                    _, _, pyannote_speaker = best_seg
                                    speaker, confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))
                        else:
                            best_seg = _choose_best_speaker(start_ms, end_ms)
                            if best_seg is not None:
                                if single_allowed_mode:
                                    _, _, pyannote_speaker = best_seg
                                    speaker, confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))
                                else:
                                    _, _, pyannote_speaker = best_seg
                                    mapped_speaker, mapped_confidence = speaker_mapping.get(pyannote_speaker, ("未知", 0.0))
                                    speaker, confidence = mapped_speaker, mapped_confidence

                        result = {
                            "type": "segment",
                            "index": i,
                            "time": format_time(start_ms),
                            "speaker": speaker,
                            "confidence": round(confidence, 2),
                            "text": text,
                        }
                        yield f"data: {json_module.dumps(result, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)
                else:
                    print("⚠️ 上传 ASR 未返回句级时间信息，回退到逐段识别模式")
                    yield f"data: {json_module.dumps({'type': 'status', 'phase': 'fallback', 'message': '整段 ASR 未返回时间戳，回退逐段识别...'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(diarization_segments), 'method': 'pyannote'})}\n\n"
                    yield f"data: {json_module.dumps({'type': 'status', 'phase': 'processing', 'message': f'正在识别，共 {len(diarization_segments)} 个片段...'}, ensure_ascii=False)}\n\n"

                    # pyannote speaker -> 陌生人编号（仅用于未匹配注册人的片段）
                    stranger_mapping = {}
                    stranger_counter = 0

                    for i, (start_ms, end_ms, pyannote_speaker) in enumerate(diarization_segments):
                        start_sample = int(start_ms / 1000 * sr)
                        end_sample = int(end_ms / 1000 * sr)
                        speech = speech_full[start_sample:end_sample]

                        if len(speech) < 0.2 * sr:
                            continue

                        text, emb = await asyncio.to_thread(_infer_segment, speech)

                        if not text:
                            continue

                        speaker = "未知"
                        confidence = 0.0
                        if emb is not None:
                            speaker, confidence = service.match_registered_speaker_guarded(
                                emb,
                                threshold=threshold,
                                duration_ms=end_ms - start_ms,
                                match_scope=matching_scope,
                                allowed_speakers=allowed_speakers,
                            )

                        if speaker == "未知":
                            if pyannote_speaker not in stranger_mapping:
                                stranger_counter += 1
                                stranger_mapping[pyannote_speaker] = f"陌生人{stranger_counter}"
                            speaker = stranger_mapping[pyannote_speaker]

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
                # ========== Fallback: VAD（传 numpy，避免文件 IO）==========
                segments = await asyncio.to_thread(service.vad_segment, speech_full)

                if not segments:
                    dur_ms = int(audio_duration * 1000)
                    segments = [[t, min(t+10000, dur_ms)] for t in range(0, dur_ms, 10000)]

                yield f"data: {json_module.dumps({'type': 'info', 'total_segments': len(segments), 'method': 'vad'})}\n\n"
                yield f"data: {json_module.dumps({'type': 'status', 'phase': 'processing', 'message': f'正在识别，共 {len(segments)} 个片段...'}, ensure_ascii=False)}\n\n"

                for i, seg in enumerate(segments):
                    start_ms, end_ms = seg

                    start_sample = int(start_ms / 1000 * sr)
                    end_sample = int(end_ms / 1000 * sr)
                    speech = speech_full[start_sample:end_sample]

                    if len(speech) < 0.2 * sr:
                        continue

                    text, emb = await asyncio.to_thread(_infer_segment, speech)
                    if not text:
                        continue

                    speaker = "未知"
                    score = 0.0
                    if emb is not None:
                            speaker, score = service.match_registered_speaker_guarded(
                                emb,
                                threshold=threshold,
                                duration_ms=end_ms - start_ms,
                                match_scope=matching_scope,
                                allowed_speakers=allowed_speakers,
                            )

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
            t_done = time.perf_counter()
            print(f"⏱️ SSE 片段处理: {t_done - t_diarize:.2f}s, 总耗时: {t_done - t_start:.2f}s")
            yield f"data: {json_module.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json_module.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if owns_temp_file and audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/v1/meeting/recordings/delete")
async def delete_meeting_recordings(request: DeleteRecordingsRequest):
    """
    按 fileId 批量删除实时会议录音。

    该接口由业务系统在病人出院等业务事件发生时调用。服务端不保存
    patientId/住院号等业务字段，只按 fileId 执行幂等删除。
    """
    file_ids = _normalize_recording_file_ids(request.fileIds)
    result = recording_store.delete_many(file_ids)
    return {
        "status": "partial" if result["failed"] else "success",
        **result,
    }


@app.post(
    "/v1/meeting/recordings/{file_id}/transcribe",
    summary="根据 fileId 识别实时录音内容",
    response_description="完整会议转写结果与 Markdown",
)
async def transcribe_meeting_recording(
    file_id: str,
    threshold: float = Query(
        default=None,
        description="可选的声纹匹配阈值；不传时使用服务端默认值。",
    ),
    allowed_speakers: list[str] | None = Query(
        default=None,
        description="可选的参会人白名单。可重复传多个同名查询参数；传入后只会在这些已注册声纹中匹配。",
    ),
):
    """
    根据实时录音返回的 fileId 识别录音内容，不需要客户端重新上传音频。

    返回结构与 `/v1/meeting/transcribe` 保持一致。
    """
    normalized = _normalize_recording_file_id(file_id)
    try:
        recording = recording_store.resolve(normalized)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="录音文件不存在")

    return await _transcribe_meeting_audio(
        recording.path,
        recording.filename,
        threshold,
        allowed_speakers,
    )


@app.post(
    "/v1/meeting/recordings/{file_id}/transcribe/stream",
    summary="按 fileId 流式识别实时录音",
    response_description="SSE 流，与 /v1/meeting/transcribe/stream 同结构",
)
async def transcribe_meeting_recording_stream(
    file_id: str,
    threshold: float = Query(
        default=None,
        description="可选的声纹匹配阈值；不传时使用服务端默认值。",
    ),
    allowed_speakers: list[str] | None = Query(
        default=None,
        description="可选的参会人白名单。可重复传多个同名查询参数；传入后只会在这些已注册声纹中匹配。",
    ),
):
    """
    根据实时录音返回的 fileId 以 SSE 流式方式识别录音内容，事件结构与
    `/v1/meeting/transcribe/stream` 完全一致：`status / info / segment / done / error`。
    """
    normalized = _normalize_recording_file_id(file_id)
    try:
        recording = recording_store.resolve(normalized)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="录音文件不存在")

    suffix = os.path.splitext(recording.filename)[1] or ".wav"
    return await _stream_meeting_transcription(
        None, suffix, threshold, allowed_speakers, source_path=recording.path
    )


@app.get("/v1/meeting/recordings/{file_id}")
async def download_meeting_recording(file_id: str):
    """按 fileId 下载实时会议录音 WAV 文件。"""
    normalized = _normalize_recording_file_id(file_id)
    try:
        recording = recording_store.resolve(normalized)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="录音文件不存在")

    return FileResponse(
        recording.path,
        media_type="audio/wav",
        filename=recording.filename,
    )

@app.websocket("/ws/meeting/live")
async def websocket_live(websocket: WebSocket):
    """
    实时会议 WebSocket 接口。

    Swagger/OpenAPI 不展示 WebSocket 参数；当前支持通过 query string
    重复传 `allowed_speakers` 来限制本次会议的匹配范围，例如：
    `/ws/meeting/live?allowed_speakers=张三&allowed_speakers=李四`

    客户端发送音频流 (bytes)，服务端返回识别结果 (JSON)。
    """
    raw_allowed_speakers = websocket.query_params.getlist("allowed_speakers")
    await websocket.accept()
    try:
        allowed_speakers = _normalize_allowed_speakers(raw_allowed_speakers)
    except HTTPException as exc:
        await websocket.send_json({
            "type": "error",
            "message": exc.detail,
        })
        await websocket.close(code=1008)
        return
    should_match_speaker = _selected_speaker_matching_enabled(allowed_speakers)
    matching_scope = service.build_matching_scope(allowed_speakers) if should_match_speaker else None

    print(f"WebSocket 连接建立，实时声纹识别: {'开启' if should_match_speaker else '关闭'}")
    
    audio_buffer = bytearray()
    recording_writer = recording_store.begin_pcm_wav()
    is_speaking = False
    silence_duration = 0
    min_segment_bytes = 16000  # 约 0.5 秒，16kHz * 16bit * 1ch
    stop_requested = False
    is_paused = False

    # 使用 SpeakerTracker 进行说话人追踪
    tracker = SpeakerTracker()
    segment_queue = asyncio.Queue()

    async def safe_send_json(payload: dict):
        try:
            await websocket.send_json(payload)
            return True
        except Exception as e:
            print(f"WebSocket 发送失败: {e}")
            return False

    async def flush_audio_buffer(reason: str):
        nonlocal audio_buffer, is_speaking, silence_duration
        if len(audio_buffer) >= min_segment_bytes:
            segment_duration = len(audio_buffer) / (16000 * 2)
            await segment_queue.put(bytes(audio_buffer))
            print(
                f"📥 WebSocket {reason}片段入队: "
                f"duration={segment_duration:.2f}s, queue_size={segment_queue.qsize()}"
            )
        audio_buffer = bytearray()
        is_speaking = False
        silence_duration = 0

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

                if should_match_speaker:
                    text, emb = await asyncio.gather(
                        asyncio.to_thread(service.transcribe_segment, audio_chunk),
                        asyncio.to_thread(service.extract_embedding, audio_chunk),
                    )
                else:
                    text = await asyncio.to_thread(service.transcribe_segment, audio_chunk)
                    emb = None

                if text:
                    speaker = "未知"
                    score = 0.0

                    if emb is not None:
                        speaker, score = service.match_speaker_fast(
                            emb,
                            match_scope=matching_scope,
                            allowed_speakers=allowed_speakers,
                        )
                        speaker, score = tracker.update(
                            speaker, score, service.registered_embeddings
                        )

                    # 过滤置信度极低的结果
                    if not (speaker != "未知" and score < CONFIG["min_confidence"]):
                        await safe_send_json({
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
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            control_text = message.get("text")
            if control_text is not None:
                try:
                    control = json_module.loads(control_text)
                except json_module.JSONDecodeError:
                    await safe_send_json({
                        "type": "error",
                        "message": "控制消息必须是合法 JSON",
                    })
                    continue

                if control.get("type") == "stop_recording":
                    stop_requested = True
                    break

                if control.get("type") == "pause_recording":
                    if not is_paused:
                        await flush_audio_buffer("暂停前")
                    is_paused = True
                    await safe_send_json({"type": "recording_paused"})
                    continue

                if control.get("type") == "resume_recording":
                    if is_paused:
                        is_paused = False
                        audio_buffer = bytearray()
                        is_speaking = False
                        silence_duration = 0
                    await safe_send_json({"type": "recording_resumed"})
                    continue

                await safe_send_json({
                    "type": "error",
                    "message": "未知控制消息",
                })
                continue

            data = message.get("bytes")
            if data is None:
                continue
            if is_paused:
                continue

            recording_writer.write_pcm(data)

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
                await flush_audio_buffer("")
                
    except WebSocketDisconnect:
        # Abnormal disconnects are not committed as recordings; callers must
        # explicitly send stop_recording to persist and receive a fileId.
        pass
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        if not is_paused:
            await flush_audio_buffer("尾")
        await segment_queue.put(None)
        await worker_task
        if stop_requested:
            try:
                file_id = recording_writer.commit()
                print(f"录音已落盘: {file_id}")
                await safe_send_json({
                    "type": "recording_saved",
                    "fileId": file_id,
                    "format": "wav",
                    "sampleRate": RECORDING_SAMPLE_RATE,
                    "channels": RECORDING_CHANNELS,
                })
            except Exception as e:
                recording_writer.abort()
                print(f"WebSocket 录音保存错误: {e}")
                await safe_send_json({
                    "type": "error",
                    "message": f"录音保存失败: {e}",
                })
        else:
            recording_writer.abort()
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
