"""有界的进程内转写任务；HTTP/SSE 订阅不拥有推理或临时输入。

仅按不可猜测的 jobId 访问，不提供任务列表。刷新可恢复；进程重启不做推理断点续跑。
"""

import asyncio
from collections import OrderedDict
import json
import logging
import threading
import time
import uuid

from fastapi import HTTPException

# 使用服务已有日志通道，确保正常断连/显式取消也留下非敏感原因记录。
logger = logging.getLogger("uvicorn.error")


def normalize_job_id(value=None):
    if value is None:
        return str(uuid.uuid4())
    try:
        parsed = uuid.UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError()
        return value
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="非法 jobId") from None


class MeetingJob:
    MAX_EVENT_BYTES = 32 * 1024 * 1024
    MAX_SUBSCRIBERS = 4

    def __init__(self, job_id, source_label):
        self.id = job_id
        self.source_label = source_label
        self.state = "running"
        self.created_at = time.time()
        self.finished_at = None
        self.events = []
        self.event_bytes = 0
        self.segment_count = 0
        self.cancelled = threading.Event()
        self.changed = asyncio.Event()
        self.subscribers = 0
        self.overflow = False
        self.task = None
        self.expiry = None
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()

    @property
    def finished(self):
        return self.finished_at is not None

    def publish(self, event):
        # 推理回调来自工作线程，所有历史写入都串行交给事件循环。
        if threading.get_ident() == self.loop_thread:
            self._append(event)
        else:
            self.loop.call_soon_threadsafe(self._append, event.copy())

    def _append(self, event):
        if self.finished or self.overflow:
            return
        event = {**event, "eventId": len(self.events) + 1}
        wire = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        size = len(wire.encode())
        if self.event_bytes + size > self.MAX_EVENT_BYTES:
            self.overflow = True
            self.cancelled.set()
            return
        self.events.append(wire)
        self.event_bytes += size
        if event["type"] == "segment":
            self.segment_count += 1
        self.changed.set()

    def finish(self, state, event):
        # 为终态留出空间；超限请求不能把截断的历史作为成功稿导出。
        self.overflow = False
        limit = self.MAX_EVENT_BYTES
        self.MAX_EVENT_BYTES += 4096
        self._append(event)
        self.MAX_EVENT_BYTES = limit
        self.state = state
        self.finished_at = time.time()
        self.changed.set()

    def describe(self):
        return {"jobId": self.id, "state": self.state, "sourceLabel": self.source_label,
                "segments": self.segment_count, "eventCount": len(self.events),
                "createdAt": self.created_at, "finishedAt": self.finished_at}

    async def stream(self, after=0):
        # HTTP 参数在发送响应头前验证；finally 只释放订阅，绝不取消 job.task。
        if self.subscribers >= self.MAX_SUBSCRIBERS:
            yield 'data: {"type":"error","message":"同一任务的订阅连接过多"}\n\n'
            return
        self.subscribers += 1
        try:
            cursor = after
            while True:
                while cursor < len(self.events):
                    event = self.events[cursor]
                    cursor += 1
                    yield event
                if self.finished:
                    return
                self.changed.clear()
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": processing\n\n"
        finally:
            self.subscribers -= 1
            logger.info("meeting subscriber detached: state=%s segments=%s", self.state, self.segment_count)


class MeetingJobStore:
    def __init__(self, *, retention_seconds=3600, max_jobs=8):
        self.retention_seconds = retention_seconds
        self.max_jobs = max_jobs
        self.jobs = OrderedDict()
        self.closing = False

    @property
    def active_count(self):
        return sum(not job.finished for job in self.jobs.values())

    def _remove(self, job_id):
        job = self.jobs.get(job_id)
        if job is not None and job.finished and not job.subscribers:
            self.jobs.pop(job_id)
            if job.expiry is not None:
                job.expiry.cancel()

    def _prune(self):
        for job in list(self.jobs.values()):
            if job.finished and time.time() - job.finished_at >= self.retention_seconds:
                self._remove(job.id)

    def get(self, job_id):
        job_id = normalize_job_id(job_id)
        self._prune()
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="未找到转写任务：可能上传尚未完成、结果已过期或服务已重启")
        job = self.jobs[job_id]
        if job.finished and time.time() - job.finished_at >= self.retention_seconds:
            raise HTTPException(status_code=410, detail="转写任务结果已过期")
        return job

    def start(self, job_id, source_label, runner, cleanup):
        job_id = normalize_job_id(job_id)
        self._prune()
        if self.closing:
            raise HTTPException(status_code=503, detail="服务正在关闭，未启动转写")
        if job_id in self.jobs:
            raise HTTPException(status_code=409, detail="任务已存在，请恢复订阅，不要重复上传")
        if self.active_count:
            raise HTTPException(status_code=503, detail="已有离线录音正在处理，请稍后重试")
        while len(self.jobs) >= self.max_jobs:
            removable = next((job.id for job in self.jobs.values() if job.finished and not job.subscribers), None)
            if removable is None:
                raise HTTPException(status_code=503, detail="任务结果仍被读取，请稍后重试")
            self._remove(removable)
        job = MeetingJob(job_id, source_label)
        self.jobs[job_id] = job
        job.publish({"type": "status", "phase": "queued", "message": "准备 MOSS 全文识别...", "jobId": job.id})
        job.task = asyncio.create_task(self._run(job, runner, cleanup))
        return job

    async def _run(self, job, runner, cleanup):
        try:
            result = await runner(job.publish, job.cancelled)
            if job.cancelled.is_set():
                raise HTTPException(status_code=409, detail="转写已取消")
            job.finish("done", {"type": "done", **result})
        except asyncio.CancelledError:
            job.cancelled.set()
            job.finish("cancelled", {"type": "error", "message": "服务关闭，转写已停止", "cancelled": True})
        except Exception as exc:
            if job.overflow:
                state, message = "error", "转写结果超过任务缓存上限，未生成成功稿"
            elif job.cancelled.is_set() and isinstance(exc, HTTPException) and exc.status_code == 409:
                state, message = "cancelled", "转写已取消，原始保存录音未删除"
            else:
                state = "error"
                message = exc.detail if isinstance(exc, HTTPException) else "离线转写失败，请检查服务日志"
            logger.warning("meeting task ended: state=%s exception=%s", state, type(exc).__name__)
            job.finish(state, {"type": "error", "message": message, "cancelled": state == "cancelled"})
        finally:
            cleanup()
            if job.finished:
                job.expiry = asyncio.get_running_loop().call_later(self.retention_seconds, self._remove, job.id)
                logger.info("meeting task finished: state=%s segments=%s elapsed=%.3fs", job.state,
                            job.segment_count, job.finished_at - job.created_at)

    def cancel(self, job_id):
        job = self.get(job_id)
        if not job.finished:
            job.state = "cancelling"
            job.cancelled.set()
            logger.info("meeting cancellation requested explicitly")
        return job.describe()

    async def close(self):
        self.closing = True
        active = [job for job in self.jobs.values() if not job.finished]
        for job in active:
            job.cancelled.set()
        # 真正停服才取消计算；先等工作线程退出，再让 ModelService 关闭模型。
        if active:
            await asyncio.gather(*(job.task for job in active))
        for job in self.jobs.values():
            if job.expiry is not None:
                job.expiry.cancel()
        self.jobs.clear()
