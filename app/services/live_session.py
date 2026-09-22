"""单一路径实时会话：FSMN-VAD → Paraformer 增量首遍 → Nano 句末精修。

复用已验证的 streaming 缓存/收敛协议；没有 legacy 能量切分或重复整段 interim。
"""

import asyncio
from datetime import datetime
import json
import math

import anyio

from app.core import CONFIG
from .recording_store import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH_BYTES
from .streaming import FsmnVadSegmenter, IncrementalParaformer

ACCURACY_HOTWORDS = ("生命体征",)


def final_token_budget(duration):
    return min(8192, max(200, math.ceil(duration * 10) + 64))


async def run_live_session(websocket, service, recording_store, *, priority, allowed_speaker_ids):
    matching = bool(allowed_speaker_ids)
    scope = service.build_matching_scope(allowed_speaker_ids) if matching else None
    vad = FsmnVadSegmenter(
        service.vad_stream,
        silence_ms=int((max(1.5, CONFIG["silence_duration"]) if priority == "accuracy" else CONFIG["silence_duration"]) * 1000),
        max_segment_ms=60000 if priority == "accuracy" else 12000,
    )
    writer = recording_store.begin_pcm_wav()
    queue = asyncio.Queue(maxsize=8)
    decoders, latest_revision, partial_texts = {}, {}, {}
    failed_streams = set()
    sequence, segment_id, revision = 0, None, 0
    segment_time = None
    interval_bytes = 9600 * SAMPLE_WIDTH_BYTES
    next_interim = interval_bytes
    paused = False
    stop_requested = False
    committed = False
    transcription_failed = False
    disconnected = False

    async def send(payload):
        if disconnected:
            return
        try:
            await websocket.send_json(payload)
        except Exception:
            # 发送失败不撤销显式 stop_recording 已保存的原始录音。
            pass

    async def enqueue(audio, bounds, is_final):
        nonlocal sequence, segment_id, revision, segment_time, next_interim
        if segment_id is None:
            sequence += 1
            segment_id = f"segment-{sequence}"
            segment_time = datetime.now().strftime("%H:%M:%S")
        revision += 1
        if not is_final:
            latest_revision[segment_id] = revision
        await queue.put({
            "audio": audio, "segmentId": segment_id, "revision": revision,
            "isFinal": is_final, "time": segment_time,
            "start_ms": bounds[0], "end_ms": bounds[1],
        })
        next_interim = len(audio) + interval_bytes
        if is_final:
            segment_id, segment_time, revision = None, None, 0
            next_interim = interval_bytes

    async def enqueue_segments(segments):
        for segment in segments:
            await enqueue(segment.audio, (segment.start_ms, segment.end_ms), True)

    async def flush():
        await enqueue_segments(await asyncio.to_thread(vad.flush))

    async def fail_final(work):
        nonlocal transcription_failed
        transcription_failed = True
        sid = work["segmentId"]
        text = partial_texts.get(sid, "")
        if text:
            await send({
                **{key: value for key, value in work.items() if key != "audio"},
                "type": "transcript", "text": text, "speakerId": None,
                "speaker": "未知", "confidence": 0.0, "degraded": True,
            })
        await send({
            "type": "error", "segmentId": sid, "revision": work["revision"], "isFinal": True,
            "message": "最终识别失败，已保留临时结果和原始录音" if text else "最终识别失败，请使用已保存录音重试",
        })

    async def worker():
        while True:
            work = await queue.get()
            if work is None:
                queue.task_done()
                return
            sid, final = work["segmentId"], work["isFinal"]
            try:
                if disconnected or (not final and work["revision"] < latest_revision.get(sid, 0)):
                    continue
                text = ""
                if sid not in failed_streams:
                    decoder = decoders.setdefault(sid, IncrementalParaformer(service.transcribe_stream_chunk))
                    try:
                        text = await asyncio.to_thread(decoder.update, work["audio"], is_final=final)
                        if text:
                            partial_texts[sid] = text
                    except Exception:
                        failed_streams.add(sid)
                        decoders.pop(sid, None)
                        await send({
                            "type": "status", "phase": "fallback", "segmentId": sid,
                            "message": "流式首遍识别失败，继续句末精修",
                        })
                if final:
                    text = await asyncio.to_thread(
                        service.transcribe_segment, work["audio"],
                        max_length=final_token_budget(len(work["audio"]) / 32000),
                        hotwords=ACCURACY_HOTWORDS if priority == "accuracy" else None,
                    )
                    if not text:
                        await fail_final(work)
                        continue
                if not text:
                    continue
                speaker_id, speaker, score = None, "未知", 0.0
                if matching and final:
                    try:
                        embedding = await asyncio.to_thread(service.extract_embedding, work["audio"])
                        if embedding is not None:
                            speaker_id, score = service.match_registered_speaker_guarded(
                                embedding, duration_ms=len(work["audio"]) * 1000 // 32000,
                                match_scope=scope, allowed_speaker_ids=allowed_speaker_ids,
                            )
                            if speaker_id is not None and score >= CONFIG["min_confidence"]:
                                speaker = service.get_speaker_name(speaker_id)
                            else:
                                speaker_id, score = None, 0.0
                    except Exception:
                        speaker_id, speaker, score = None, "未知", 0.0
                await send({
                    **{key: value for key, value in work.items() if key != "audio"},
                    "type": "transcript", "text": text,
                    "speakerId": speaker_id, "speaker": speaker, "confidence": round(score, 2),
                })
            except Exception:
                if final:
                    await fail_final(work)
            finally:
                if final:
                    decoders.pop(sid, None)
                    partial_texts.pop(sid, None)
                    latest_revision.pop(sid, None)
                    failed_streams.discard(sid)
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                disconnected = True
                break
            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                    if not isinstance(control, dict):
                        raise ValueError()
                except ValueError:
                    await send({"type": "error", "message": "控制消息必须是合法 JSON 对象"})
                    continue
                kind = control.get("type")
                if kind == "stop_recording":
                    stop_requested = True
                    break
                if kind == "pause_recording":
                    if not paused:
                        await flush()
                    paused = True
                    await send({"type": "recording_paused"})
                elif kind == "resume_recording":
                    # flush 已重置缓存并保留绝对采样原点，不能重复计入暂停时长。
                    paused = False
                    await send({"type": "recording_resumed"})
                else:
                    await send({"type": "error", "message": "未知控制消息"})
                continue
            data = message.get("bytes")
            if data is None or paused:
                continue
            if len(data) > 2 * 1024 * 1024:
                raise ValueError("单个 PCM 包超过限制")
            writer.write_pcm(data)
            await enqueue_segments(await asyncio.to_thread(vad.feed, data))
            if vad.current_segment_size >= next_interim:
                await enqueue(vad.current_segment(), vad.current_bounds, False)
    except Exception:
        transcription_failed = True
        await send({"type": "error", "message": "实时音频处理失败"})
    finally:
        try:
            if stop_requested and not paused:
                try:
                    await flush()
                except Exception:
                    transcription_failed = True
                    await send({"type": "error", "message": "尾段处理失败"})
            if not worker_task.done():
                await queue.put(None)
                await worker_task
            if stop_requested:
                try:
                    file_id = writer.commit()
                    committed = True
                    await send({
                        "type": "recording_saved", "fileId": file_id,
                        "format": "wav", "sampleRate": SAMPLE_RATE, "channels": CHANNELS,
                        "transcriptionStatus": "failed" if transcription_failed else "complete",
                    })
                except Exception:
                    writer.abort()
                    await send({"type": "error", "message": "录音保存失败"})
            else:
                writer.abort()
        finally:
            if not worker_task.done():
                worker_task.cancel()
                with anyio.CancelScope(shield=True):
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
            if not committed:
                writer.abort()
