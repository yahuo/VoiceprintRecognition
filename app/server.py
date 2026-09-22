#!/usr/bin/env python3
"""会议 API：MOSS 离线统一入口、两遍实时转写、CAM++ 身份验证。"""

import asyncio
from datetime import datetime
import math
import os
import tempfile
import threading
from typing import Annotated

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from .core import (
    CONFIG, ModelService, delete_voiceprint_by_id, generate_voiceprint_id,
    load_voiceprint_index, save_voiceprint_embedding,
)
from .services.live_session import ACCURACY_HOTWORDS, final_token_budget, run_live_session
from .services.meeting_jobs import MeetingJobStore, normalize_job_id
from .services.moss import MossError, max_audio_seconds
from .services.recording_store import recording_store

app = FastAPI(
    title="Voiceprint Meeting System API",
    description="MOSS 离线转写与分人、Paraformer Streaming + Nano 实时转写、CAM++ 身份验证",
    version="3.0.0",
)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Meeting-Job-Id"])
service = ModelService()
meeting_jobs = MeetingJobStore()
DEVICE = "cpu"
TRANSCRIPTION_PRIORITIES = frozenset({"speed", "accuracy"})
OFFLINE_TRANSCRIPTION_PRIORITY_DESCRIPTION = (
    "已弃用的兼容参数：接受 speed（默认）或 accuracy，但均使用同一完整上下文 MOSS 链路。"
    "旧 priority 字段仅兼容回显，processingMode 固定为 unified；超限或推理失败明确报错，不降级、不截断。"
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
# 保留供旧调用方使用的短片段输出预算函数；不再用于离线选路。
_accuracy_max_length = final_token_budget


@app.get("/client")
async def client():
    return FileResponse(os.path.join(static_dir, "web_client.html"))


def _normalize_transcription_priority(priority):
    normalized = (priority or "speed").strip().lower() or "speed"
    if normalized not in TRANSCRIPTION_PRIORITIES:
        raise HTTPException(status_code=400, detail="priority 仅支持 speed 或 accuracy")
    return normalized


def _priority_metadata(priority):
    # 保留旧 speed/accuracy 枚举的反序列化兼容；不再作为模型选路依据。
    return {"requestedPriority": priority, "effectivePriority": priority,
            "processingMode": "unified", "fallbackReason": None,
            "priorityDeprecated": True, "method": "moss"}


def _normalize_voiceprint_id(raw_id):
    if raw_id is None:
        return generate_voiceprint_id()
    return _normalize_required_voiceprint_id(raw_id)


def _normalize_required_voiceprint_id(raw_id):
    if raw_id is None or raw_id == "":
        raise HTTPException(status_code=400, detail="声纹 id 不能为空")
    return raw_id


def _normalize_speaker_name(raw_name):
    cleaned = (raw_name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="说话人姓名不能为空")
    return cleaned


def _normalize_allowed_speaker_ids(raw_allowed_speaker_ids):
    normalized = list(dict.fromkeys(x for x in (raw_allowed_speaker_ids or []) if x))
    missing = [x for x in normalized if x not in service.registered_embeddings]
    if missing:
        raise HTTPException(status_code=400, detail=f"以下参会人 id 未注册声纹: {', '.join(missing)}")
    return normalized or None


def _selected_speaker_matching_enabled(allowed_speaker_ids):
    return bool(allowed_speaker_ids)


def _voiceprint_list_items(index):
    return [{"id": entry["id"], "name": entry["name"]} for entry in index.values()]


def _meeting_options(threshold, allowed_speaker_ids, priority):
    priority = _normalize_transcription_priority(priority)
    threshold = CONFIG["speaker_threshold"] if threshold is None else threshold
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold 必须是 0 到 1 的有限数值")
    return threshold, _normalize_allowed_speaker_ids(allowed_speaker_ids), priority


class DeleteRecordingsRequest(BaseModel):
    fileIds: list[str] = Field(default_factory=list)
    reason: str | None = None
    requestId: str | None = None


def _normalize_recording_file_id(file_id):
    try:
        return recording_store.normalize_file_id(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法 fileId")


def _normalize_recording_file_ids(file_ids):
    if not file_ids:
        raise HTTPException(status_code=400, detail="fileIds 不能为空")
    return list(dict.fromkeys(_normalize_recording_file_id(x) for x in file_ids))


def _resolve_recording(file_id):
    try:
        return recording_store.resolve(_normalize_recording_file_id(file_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="录音文件不存在")


def _build_meeting_markdown(transcript, audio_filename):
    lines = ["# 会议复核稿\n", f"- **日期**: {datetime.now():%Y-%m-%d %H:%M}\n",
             f"- **音频文件**: {audio_filename}\n",
             "> 自动转写，医学数字、术语与身份归属需人工核对。\n\n## 会议内容\n"]
    speaker = None
    for item in transcript:
        if speaker != item["speaker"]:
            lines.append(f"\n**[{item['time']}] {item['speaker']}**:\n")
            speaker = item["speaker"]
        lines.append(f"> {item['text']}\n")
    return "".join(lines)


async def _transcribe_meeting_audio(audio_path, audio_filename, threshold, allowed_speaker_ids,
                                    priority="speed", *, progress=None, cancelled=None):
    from .services.meeting import process_meeting
    threshold, allowed_speaker_ids, priority = _meeting_options(threshold, allowed_speaker_ids, priority)
    matching = _selected_speaker_matching_enabled(allowed_speaker_ids)
    scope = service.build_matching_scope(allowed_speaker_ids) if matching else None
    if cancelled is None:
        cancelled = threading.Event()
    def run():
        # 取消后的线程异常也必须被消费；Python 3.14 的 shield 会记录未交付的异常。
        try:
            return process_meeting(
                service, audio_path, threshold, allowed_speaker_ids, scope, matching,
                progress=progress, cancelled=cancelled,
            ), None
        except Exception as exc:
            return None, exc

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        transcript, error = await asyncio.shield(task)
        if error is not None:
            raise error
        return {
            "status": "success", "segments": len(transcript), "transcript": transcript,
            "markdown": _build_meeting_markdown(transcript, audio_filename),
            **_priority_metadata(priority),
        }
    except MossError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        if not task.done():
            cancelled.set()
            # asyncio.to_thread 取消不会终止线程；等待其回收 worker 后再清理输入文件。
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass


async def _save_upload(file):
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".wav") as tmp:
            path = tmp.name
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传文件超过大小限制")
                tmp.write(chunk)
            if not size:
                raise HTTPException(status_code=422, detail="上传音频为空")
        return path
    except BaseException:
        _remove_temp(path)
        raise


def _remove_temp(path):
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@app.on_event("startup")
async def startup_event():
    recording_store.cleanup_temp_files()
    await asyncio.to_thread(service.load_models, device=DEVICE)


@app.on_event("shutdown")
async def shutdown_event():
    await meeting_jobs.close()
    await asyncio.to_thread(service.close)


@app.get("/")
async def root():
    return {
        "status": "ok", "service": "Voiceprint Meeting System", "version": "3.0.0",
        "models_loaded": service.is_loaded, "registered_speakers": len(service.registered_embeddings),
        "config": {"asr_backend": "nano", "live_pipeline": "streaming", "offline_pipeline": "moss",
                   "asr_language": CONFIG["asr_language"], "speaker_threshold": CONFIG["speaker_threshold"]},
        "offline_configured": bool(os.environ.get("MOSS_PYTHON") and os.environ.get("MOSS_MODEL_PATH")),
        "offline_max_audio_seconds": max_audio_seconds(),
        "active_meeting_jobs": meeting_jobs.active_count,
    }


@app.get("/v1/voiceprint/list")
async def list_speakers():
    speakers = _voiceprint_list_items(load_voiceprint_index())
    return {"status": "success", "count": len(speakers), "speakers": speakers}


@app.post("/v1/voiceprint/reload")
async def reload_voiceprints():
    service.reload_voiceprints()
    return {"status": "success", "message": "声纹库已重新加载",
            "count": len(service.registered_embeddings), "speakers": _voiceprint_list_items(service.registered_speakers)}


@app.post("/v1/meeting/summarize")
async def summarize_meeting_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是合法的 JSON")
    if not isinstance(body, dict) or not body.get("transcript"):
        raise HTTPException(status_code=400, detail="transcript 不能为空")
    from .services.summarizer import summarize_meeting
    result = await asyncio.to_thread(
        summarize_meeting, transcript_items=body["transcript"],
        base_url=body.get("base_url"), api_key=body.get("api_key"), model=body.get("model"),
    )
    return JSONResponse(status_code=502, content=result) if result["status"] == "error" else result


@app.post("/v1/voiceprint/register")
async def register_speaker(name: str = Form(...), file: UploadFile = File(...),
                           voiceprint_id: str | None = Form(default=None, alias="id")):
    speaker_id, speaker_name = _normalize_voiceprint_id(voiceprint_id), _normalize_speaker_name(name)
    # 与会议上传复用大小限制和异常清理，避免读入失败后遗留半写文件。
    audio_path = await _save_upload(file)
    try:
        embedding = await asyncio.to_thread(service.extract_embedding, audio_path)
        if embedding is None:
            raise HTTPException(status_code=400, detail="无法提取声纹特征")
        entry = save_voiceprint_embedding(speaker_id, speaker_name, embedding)
        service.reload_voiceprints()
        return {"status": "success", "id": entry["id"], "name": entry["name"],
                "message": f"声纹 '{speaker_name}' 注册成功", "embedding_shape": embedding.shape}
    finally:
        _remove_temp(audio_path)


@app.get("/v1/voiceprint/exists")
async def voiceprint_exists(voiceprint_id: str = Query(..., alias="id")):
    speaker_id = _normalize_required_voiceprint_id(voiceprint_id)
    entry = load_voiceprint_index().get(speaker_id)
    return {"registered": entry is not None, "id": speaker_id, "name": entry["name"] if entry else None}


@app.delete("/v1/voiceprint")
async def delete_speaker(voiceprint_id: str = Query(..., alias="id")):
    speaker_id = _normalize_required_voiceprint_id(voiceprint_id)
    entry = delete_voiceprint_by_id(speaker_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"未找到声纹 id: {speaker_id}")
    service.reload_voiceprints()
    return {"status": "success", "id": entry["id"], "name": entry["name"], "message": f"已删除声纹: {entry['name']}"}


@app.post("/v1/meeting/transcribe", summary="上传音频生成 MOSS 会议复核稿")
async def transcribe_meeting(
    file: UploadFile = File(...), threshold: float = Form(default=None),
    allowed_speaker_ids: list[str] | None = Form(default=None),
    priority: Annotated[str, Form(description=OFFLINE_TRANSCRIPTION_PRIORITY_DESCRIPTION)] = "speed",
):
    _meeting_options(threshold, allowed_speaker_ids, priority)
    path = await _save_upload(file)
    try:
        return await _transcribe_meeting_audio(path, file.filename, threshold, allowed_speaker_ids, priority)
    finally:
        _remove_temp(path)


@app.post("/v1/meeting/transcribe/stream", summary="SSE 处理完整录音（不是实时音频输入）")
async def transcribe_meeting_stream(
    file: UploadFile = File(...), threshold: float = Form(default=None),
    allowed_speaker_ids: list[str] | None = Form(default=None),
    priority: Annotated[str, Form(description=OFFLINE_TRANSCRIPTION_PRIORITY_DESCRIPTION)] = "speed",
    job_id: Annotated[str | None, Form()] = None,
):
    job_id = normalize_job_id(job_id)
    _meeting_options(threshold, allowed_speaker_ids, priority)
    path = await _save_upload(file)
    try:
        return await _stream_meeting_transcription(
            None, ".wav", threshold, allowed_speaker_ids, priority=priority,
            source_path=path, owns_source=True, source_label=file.filename, job_id=job_id,
        )
    except BaseException:
        _remove_temp(path)
        raise


async def _stream_meeting_transcription(content, suffix, threshold, allowed_speaker_ids, *,
                                         priority="speed", source_path=None, owns_source=False,
                                         source_label=None, job_id=None):
    job_id = normalize_job_id(job_id)
    if (content is None) == (source_path is None):
        raise ValueError("must supply exactly one of content or source_path")
    threshold, allowed_speaker_ids, priority = _meeting_options(threshold, allowed_speaker_ids, priority)
    if content is not None:
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="上传文件超过大小限制")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            source_path = tmp.name
        owns_source = True

    def cleanup():
        if owns_source:
            _remove_temp(source_path)

    source_label = source_label or os.path.basename(source_path)

    async def run(progress, cancelled):
        def publish(event):
            progress({**event, **_priority_metadata(priority)} if event["type"] == "info" else event)
        result = await _transcribe_meeting_audio(
            source_path, source_label, threshold, allowed_speaker_ids, priority,
            progress=publish, cancelled=cancelled,
        )
        return {"segments": result["segments"], **_priority_metadata(priority)}

    try:
        job = meeting_jobs.start(job_id, source_label, run, cleanup)
    except BaseException:
        cleanup()
        raise
    # 任务拥有输入；订阅断开不清理音频、不取消推理。
    return _meeting_job_response(job)


def _meeting_job_response(job, after=0):
    if after < 0 or after > len(job.events):
        raise HTTPException(status_code=400, detail="非法转写事件游标")
    if job.subscribers >= job.MAX_SUBSCRIBERS:
        raise HTTPException(status_code=429, detail="同一任务的订阅连接过多")
    return StreamingResponse(
        job.stream(after), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "Connection": "keep-alive", "X-Accel-Buffering": "no",
                 "X-Meeting-Job-Id": job.id},
    )


@app.get("/v1/meeting/jobs/{job_id}")
async def get_meeting_job(job_id: str):
    return JSONResponse(meeting_jobs.get(job_id).describe(), headers={"Cache-Control": "no-store"})


@app.get("/v1/meeting/jobs/{job_id}/events")
async def subscribe_meeting_job(job_id: str, after: int = Query(default=0, ge=0)):
    return _meeting_job_response(meeting_jobs.get(job_id), after)


@app.post("/v1/meeting/jobs/{job_id}/cancel")
async def cancel_meeting_job(job_id: str):
    return meeting_jobs.cancel(job_id)


@app.post("/v1/meeting/recordings/delete")
async def delete_meeting_recordings(request: DeleteRecordingsRequest):
    result = recording_store.delete_many(_normalize_recording_file_ids(request.fileIds))
    return {"status": "partial" if result["failed"] else "success", **result}


@app.post("/v1/meeting/recordings/{file_id}/transcribe", summary="按 fileId 生成 MOSS 会后复核稿")
async def transcribe_meeting_recording(
    file_id: str, threshold: float = Query(default=None),
    allowed_speaker_ids: list[str] | None = Query(default=None),
    priority: Annotated[str, Query(description=OFFLINE_TRANSCRIPTION_PRIORITY_DESCRIPTION)] = "speed",
):
    recording = _resolve_recording(file_id)
    return await _transcribe_meeting_audio(recording.path, recording.filename, threshold, allowed_speaker_ids, priority)


@app.post("/v1/meeting/recordings/{file_id}/transcribe/stream", summary="SSE 获取 MOSS 会后复核稿")
async def transcribe_meeting_recording_stream(
    file_id: str, threshold: float = Query(default=None),
    allowed_speaker_ids: list[str] | None = Query(default=None),
    priority: Annotated[str, Query(description=OFFLINE_TRANSCRIPTION_PRIORITY_DESCRIPTION)] = "speed",
    job_id: Annotated[str | None, Query()] = None,
):
    recording = _resolve_recording(file_id)
    return await _stream_meeting_transcription(
        None, os.path.splitext(recording.filename)[1] or ".wav", threshold, allowed_speaker_ids,
        priority=priority, source_path=recording.path, source_label=recording.filename, job_id=job_id,
    )


@app.get("/v1/meeting/recordings/{file_id}")
async def download_meeting_recording(file_id: str):
    recording = _resolve_recording(file_id)
    return FileResponse(recording.path, media_type="audio/wav", filename=recording.filename)


@app.websocket("/ws/meeting/live")
async def websocket_live(websocket: WebSocket):
    """两遍流式接口；保留 priority 作为断句/热词策略，所有模式都有 revision/isFinal。"""
    await websocket.accept()
    try:
        allowed = _normalize_allowed_speaker_ids(websocket.query_params.getlist("allowed_speaker_ids"))
        values = websocket.query_params.getlist("priority")
        priority = _normalize_transcription_priority(values[-1] if values else None)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "message": exc.detail})
        await websocket.close(code=1008)
        return
    await run_live_session(websocket, service, recording_store, priority=priority, allowed_speaker_ids=allowed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voiceprint Meeting System Server")
    parser.add_argument("--device", "-d", default="cpu", help="实时 ASR/CAM++ 设备；MOSS 使用独立 CUDA worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8000)
    args = parser.parse_args()
    DEVICE = args.device
    uvicorn.run(app, host=args.host, port=args.port)
