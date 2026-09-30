"""单一路径实时会话：FSMN-VAD → Paraformer 增量首遍 → Nano 句末精修。

复用已验证的 streaming 缓存/收敛协议；没有 legacy 能量切分或重复整段 interim。
各模型在独立的单线程 FIFO 阶段执行；积压时只合并/作废首遍，最终片段不丢弃。
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import threading
import time

import anyio
import torch

from app.core import CONFIG
from .recording_store import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH_BYTES
from .streaming import FsmnVadSegmenter, IncrementalParaformer

ACCURACY_HOTWORDS = ("生命体征",)


def final_token_budget(duration):
    return min(8192, max(200, math.ceil(duration * 10) + 64))


def _percentiles_ms(values):
    if not values:
        return None
    values = sorted(values)
    pick = lambda q: round(values[max(0, math.ceil(q * len(values)) - 1)] * 1000, 2)
    return {"p50": pick(0.5), "p95": pick(0.95), "max": pick(1.0)}


class InferenceStage:
    """单线程 FIFO 推理阶段。

    与模型锁同样串行，但等待者不占默认线程池，按提交顺序执行，并记录排队/执行耗时。
    torch_threads 为 None 时继承进程默认；否则只作用于本阶段线程。
    """

    def __init__(self, name, torch_threads=None, *, recent=512):
        self.name = name
        self._torch_threads = torch_threads
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"live-{name}", initializer=self._init_thread,
        )
        self._stats_lock = threading.Lock()
        self._pending = 0
        self._calls = 0
        self._wait_total = 0.0
        self._run_total = 0.0
        self._recent = deque(maxlen=recent)

    def _init_thread(self):
        if self._torch_threads is None:
            return
        inherited = torch.get_num_threads()
        torch.set_num_threads(self._torch_threads)
        # set_num_threads 同时改写新线程继承的全局默认；本线程的值已固化，在临时线程里恢复全局。
        restore = threading.Thread(target=torch.set_num_threads, args=(inherited,))
        restore.start()
        restore.join()

    def start(self):
        self._executor.submit(lambda: None).result()

    def _finish(self, _future):
        with self._stats_lock:
            self._pending -= 1

    async def run(self, fn, *args, **kwargs):
        submitted = time.perf_counter()

        def call():
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                wait, elapsed = started - submitted, time.perf_counter() - started
                with self._stats_lock:
                    self._calls += 1
                    self._wait_total += wait
                    self._run_total += elapsed
                    self._recent.append((wait, elapsed))

        with self._stats_lock:
            self._pending += 1
        future = self._executor.submit(call)
        future.add_done_callback(self._finish)
        return await asyncio.wrap_future(future)

    def snapshot(self):
        with self._stats_lock:
            recent = list(self._recent)
            data = {
                "pending": self._pending, "calls": self._calls,
                "wait_seconds_total": round(self._wait_total, 3),
                "run_seconds_total": round(self._run_total, 3),
            }
        data["recent"] = {
            "samples": len(recent),
            "wait_ms": _percentiles_ms([wait for wait, _ in recent]),
            "run_ms": _percentiles_ms([run for _, run in recent]),
        }
        return data


# FSMN-VAD 固定在 CPU，每 200ms 一次小算子，多线程调度开销大于计算本身；
# 其余阶段保持进程原有的 torch 线程数。
VAD_STAGE = InferenceStage("vad", CONFIG["live_vad_cpu_threads"])
PARAFORMER_STAGE = InferenceStage("paraformer")
NANO_STAGE = InferenceStage("nano")
CAMPP_STAGE = InferenceStage("campp")
STAGES = {stage.name: stage for stage in (VAD_STAGE, PARAFORMER_STAGE, NANO_STAGE, CAMPP_STAGE)}
_stages_started = False
_active_sessions = 0


def _start_stages():
    global _stages_started
    if not _stages_started:
        for stage in STAGES.values():
            stage.start()
        _stages_started = True


def live_metrics():
    return {
        "active_sessions": _active_sessions,
        "stages": {name: stage.snapshot() for name, stage in STAGES.items()},
    }


class SegmentWorkQueue:
    """最终片段逐条保序、从不丢弃；同段未处理的 interim 只保留最新快照，句末入队即作废。"""

    def __init__(self, maxsize):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._interim = None

    async def put(self, work):
        interim = work is not None and not work["isFinal"]
        if interim and self._interim is not None:
            self._interim.update(work)
            return
        if work is not None and work["isFinal"] and self._interim is not None:
            # 句末精修会覆盖本段首遍文本，排队中的首遍不再占用 Paraformer。
            self._interim["superseded"] = True
            self._interim = None
        await self._queue.put(work)
        if interim:
            self._interim = work

    async def get(self):
        work = await self._queue.get()
        if work is self._interim:
            self._interim = None
        return work

    def task_done(self):
        self._queue.task_done()


async def run_live_session(websocket, service, recording_store, *, priority, allowed_speaker_ids):
    global _active_sessions
    _start_stages()
    _active_sessions += 1
    try:
        await _run_live_session(
            websocket, service, recording_store,
            priority=priority, allowed_speaker_ids=allowed_speaker_ids,
        )
    finally:
        _active_sessions -= 1


async def _run_live_session(websocket, service, recording_store, *, priority, allowed_speaker_ids):
    matching = bool(allowed_speaker_ids)
    scope = service.build_matching_scope(allowed_speaker_ids) if matching else None
    vad = FsmnVadSegmenter(
        service.vad_stream,
        silence_ms=int((max(1.5, CONFIG["silence_duration"]) if priority == "accuracy" else CONFIG["silence_duration"]) * 1000),
        max_segment_ms=60000 if priority == "accuracy" else 12000,
    )
    writer = recording_store.begin_pcm_wav()
    queue = SegmentWorkQueue(maxsize=8)
    decoders, partial_texts = {}, {}
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
        await enqueue_segments(await VAD_STAGE.run(vad.flush))

    async def fail_final(work):
        nonlocal transcription_failed
        transcription_failed = True
        sid = work["segmentId"]
        if sid not in failed_streams:
            # 仅在精修失败时补完首遍尾部作为降级文本；成功时首遍会被覆盖，不在关键路径上排空。
            decoder = decoders.pop(sid, None) or IncrementalParaformer(service.transcribe_stream_chunk)
            try:
                text = await PARAFORMER_STAGE.run(decoder.update, work["audio"], is_final=True)
                if text:
                    partial_texts[sid] = text
            except Exception:
                pass
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
                if disconnected or work.get("superseded"):
                    continue
                if final:
                    text = await NANO_STAGE.run(
                        service.transcribe_segment, work["audio"],
                        max_length=final_token_budget(len(work["audio"]) / 32000),
                        hotwords=ACCURACY_HOTWORDS if priority == "accuracy" else None,
                    )
                    if not text:
                        await fail_final(work)
                        continue
                else:
                    if sid in failed_streams:
                        continue
                    decoder = decoders.setdefault(sid, IncrementalParaformer(service.transcribe_stream_chunk))
                    try:
                        text = await PARAFORMER_STAGE.run(decoder.update, work["audio"])
                    except Exception:
                        failed_streams.add(sid)
                        decoders.pop(sid, None)
                        await send({
                            "type": "status", "phase": "fallback", "segmentId": sid,
                            "message": "流式首遍识别失败，继续句末精修",
                        })
                        continue
                    if not text:
                        continue
                    partial_texts[sid] = text
                speaker_id, speaker, score = None, "未知", 0.0
                if matching and final:
                    try:
                        embedding = await CAMPP_STAGE.run(service.extract_embedding, work["audio"])
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
            if vad.pending_bytes + len(data) < vad.chunk_bytes:
                vad.feed(data)  # 不足一帧只缓存，不调用模型也不切换线程
            else:
                await enqueue_segments(await VAD_STAGE.run(vad.feed, data))
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
