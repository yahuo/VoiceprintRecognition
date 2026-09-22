"""MOSS 离线适配：本机隔离 worker、完整输入、严格输出契约。

不导入 vLLM/Torch；业务环境通过 MOSS_PYTHON 调用独立的 CUDA 环境。
模型不可用、超限、截断或解析失败均显式报错，不回退其他识别模型。
"""

import atexit
import json
import logging
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import threading
import time


logger = logging.getLogger("uvicorn.error")

MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
# 与官方 compact transcript 格式一致；只接受完整覆盖原始输出的片段。
_NUMBER = r"\d+(?:\.\d+)?"
_SEGMENT = re.compile(
    rf"\[({_NUMBER})\]\s*\[(S\d+)\](.*?)\[({_NUMBER})\]"
    rf"(?=\s*(?:\[{_NUMBER}\]\s*\[S\d+\]|$))", re.DOTALL,
)
# 未到 EOS 时必须看到下一段头，不能把正文里的 [26] 或半截时间戳当作段末。
_STREAM_SEGMENT = re.compile(
    rf"\s*\[({_NUMBER})\]\s*\[(S\d+)\](.*?)\[({_NUMBER})\]"
    rf"(?=\s*\[{_NUMBER}\]\s*\[S\d+\])", re.DOTALL,
)


class MossError(RuntimeError):
    status_code = 502


class MossCancelled(MossError):
    status_code = 409


class MossUnavailable(MossError):
    status_code = 503


class MossBusy(MossUnavailable):
    pass


class MossTimeout(MossError):
    status_code = 504


class AudioTooLong(MossError):
    status_code = 413


class InvalidAudio(MossError):
    status_code = 422


def max_audio_seconds():
    # 覆盖已验证的完整 43m11s MDT；更大上下文必须另行验证后调整。
    return float(os.environ.get("MOSS_MAX_AUDIO_SECONDS", "2600"))


def _parse_segment(match, duration_seconds, last_start):
    start, label, text, end = match.groups()
    start, end = float(start), float(end)
    if (not all(math.isfinite(x) for x in (start, end))
            or not 0 <= start < end <= duration_seconds + 0.1
            or start < last_start or not text.strip()):
        raise MossError("MOSS 返回无效时间戳或空片段")
    start_ms = round(start * 1000)
    end_ms = min(round(end * 1000), round(duration_seconds * 1000))
    if end_ms <= start_ms:
        raise MossError("MOSS 片段超出有效音频范围")
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "diarizationSpeaker": label, "text": text.strip(),
    }, start


def parse_result(result: dict, duration_seconds: float) -> list[dict]:
    """拒绝 token 上限截断、文本丢失和非法时间；保留真实重叠，不重排片段。"""
    if result.get("finish_reason") != "stop" or result.get("eos_reached") is not True:
        raise MossError("MOSS 未正常结束，结果可能被截断，未生成成功转写")
    raw = result.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        raise MossError("MOSS 未返回可解析的转写文本")
    segments = []
    cursor = 0
    last_start = 0
    for match in _SEGMENT.finditer(raw):
        if raw[cursor:match.start()].strip():
            raise MossError("MOSS 输出格式异常，拒绝丢弃未解析文本")
        segment, last_start = _parse_segment(match, duration_seconds, last_start)
        segments.append(segment)
        cursor = match.end()
    if not segments or raw[cursor:].strip():
        raise MossError("MOSS 输出不完整，拒绝丢弃未解析文本")
    return segments


class MossStreamParser:
    """只发布边界已闭合的前缀；全文成功仍必须通过正常 EOS 和最终一致性校验。"""

    def __init__(self, duration_seconds):
        self.duration = duration_seconds
        self.raw = ""
        self.pending = ""
        self.segments = []
        self.last_start = 0.0

    def feed(self, delta):
        if not isinstance(delta, str):
            raise MossError("MOSS worker 返回无效文本增量")
        if len(self.raw) + len(delta) > 4 * 1024 * 1024:
            raise MossError("MOSS worker 转写超过大小限制")
        self.raw += delta
        self.pending += delta
        ready = []
        while (match := _STREAM_SEGMENT.match(self.pending)) is not None:
            segment, self.last_start = _parse_segment(match, self.duration, self.last_start)
            self.segments.append(segment)
            ready.append(segment)
            self.pending = self.pending[match.end():]
        return ready

    def finish(self, result):
        segments = parse_result(result, self.duration)
        if result["raw"] != self.raw or segments[:len(self.segments)] != self.segments:
            raise MossError("MOSS 流式结果与最终转写不一致")
        return segments


class MossTranscriber:
    """每个业务进程一个懒加载 worker；忙时拒绝排队，超时/取消回收整个进程组。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._reader = None
        self._buffer = bytearray()
        self._request_sequence = 0
        atexit.register(self.close)

    def _receive(self, timeout, cancelled):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self._reader, selectors.EVENT_READ)
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise MossCancelled("离线转写已取消")
                if b"\n" in self._buffer:
                    line, _, rest = self._buffer.partition(b"\n")
                    self._buffer = bytearray(rest)
                    try:
                        message = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise MossError("MOSS worker 返回无效响应") from exc
                    if not isinstance(message, dict):
                        raise MossError("MOSS worker 返回无效响应")
                    if message.get("error"):
                        raise MossError(str(message["error"]))
                    return message
                if self._process is not None and self._process.poll() is not None:
                    raise MossUnavailable("MOSS worker 已退出，请检查运行日志")
                if time.monotonic() >= deadline:
                    raise MossTimeout("MOSS 推理超时，worker 已回收，请检查资源或重试")
                if selector.select(min(0.2, max(0, deadline - time.monotonic()))):
                    chunk = os.read(self._reader, 65536)
                    if not chunk:
                        raise MossUnavailable("MOSS worker 已退出，请检查独立运行环境和日志")
                    self._buffer.extend(chunk)
                    if len(self._buffer) > 16 * 1024 * 1024:
                        raise MossError("MOSS worker 响应超过大小限制")

    def _start(self, cancelled):
        if self._process is not None and self._process.poll() is None:
            return
        self._stop("restart")
        python = os.environ.get("MOSS_PYTHON", "")
        model = os.environ.get("MOSS_MODEL_PATH", "")
        if not python or not Path(python).is_file() or not Path(model).is_dir() or not model:
            raise MossUnavailable("请配置本机 MOSS_PYTHON 和 MOSS_MODEL_PATH；不支持自动回退或云端上传")
        reader, writer = os.pipe()
        self._reader = reader
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0", "TOKENIZERS_PARALLELISM": "false",
            "MOSS_RESULT_FD": str(writer),
        })
        try:
            self._process = subprocess.Popen(
                [python, "-u", str(Path(__file__).with_name("moss_worker.py"))],
                stdin=subprocess.PIPE, stdout=None, stderr=None,
                env=env, pass_fds=(writer,), start_new_session=True,
            )
        except OSError as exc:
            raise MossUnavailable("无法启动 MOSS_PYTHON，请检查路径及执行权限") from exc
        finally:
            os.close(writer)
        ready = self._receive(float(os.environ.get("MOSS_START_TIMEOUT_SECONDS", "600")), cancelled)
        if ready != {"ready": True}:
            raise MossUnavailable("MOSS worker 初始化失败")

    def transcribe(self, wav_path: str, duration_seconds: float, *, cancelled=None, on_segment=None):
        if duration_seconds > max_audio_seconds():
            raise AudioTooLong("音频超过 MOSS 完整上下文时长上限；不会截断或独立分块识别")
        if not self._lock.acquire(blocking=False):
            raise MossBusy("MOSS 正在处理另一个录音，请稍后重试")
        request_id, request_finished = None, False
        try:
            if cancelled is not None and cancelled.is_set():
                raise MossCancelled("离线转写已取消")
            # 加载不是请求本身；取消时允许当前加载完成，保留热模型供后续请求使用。
            self._start(None)
            if cancelled is not None and cancelled.is_set():
                raise MossCancelled("离线转写已取消")
            self._request_sequence += 1
            request_id = str(self._request_sequence)
            request = json.dumps({"audio_path": str(Path(wav_path).resolve()), "request_id": request_id}) + "\n"
            self._process.stdin.write(request.encode())
            self._process.stdin.flush()
            deadline = time.monotonic() + float(os.environ.get("MOSS_TIMEOUT_SECONDS", "1800"))
            parser = MossStreamParser(duration_seconds)
            streamed = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MossTimeout("MOSS 推理超时，worker 已回收，请检查资源或重试")
                result = self._receive(remaining, cancelled)
                if result.get("type") == "delta":
                    streamed = True
                    for segment in parser.feed(result.get("text")):
                        if on_segment is not None:
                            on_segment(segment.copy())
                    continue
                if result.get("type") not in (None, "result"):
                    raise MossError("MOSS worker 返回未知事件")
                request_finished = True
                segments = parser.finish(result) if streamed else parse_result(result, duration_seconds)
                # 最后一段只能在 EOS 后发布；已发布前缀不重复发送。
                if on_segment is not None:
                    for segment in segments[len(parser.segments):]:
                        on_segment(segment.copy())
                return segments
        except MossCancelled:
            try:
                if request_id is not None and not request_finished:
                    self._cancel_request(request_id)
                logger.info("MOSS request cancelled; worker retained pid=%s", self._process.pid if self._process else None)
            except BaseException:
                self._stop("cancel_ack_failed")
                raise
            raise
        except OSError as exc:
            self._stop("communication_error")
            raise MossUnavailable("MOSS worker 通信失败，请检查本机运行环境") from exc
        except BaseException as exc:
            self._stop(type(exc).__name__)
            raise
        finally:
            self._lock.release()

    def _cancel_request(self, request_id):
        self._process.stdin.write((json.dumps({"type": "cancel", "request_id": request_id}) + "\n").encode())
        self._process.stdin.flush()
        deadline = time.monotonic() + float(os.environ.get("MOSS_CANCEL_TIMEOUT_SECONDS", "30"))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MossTimeout("MOSS 取消确认超时，已回收异常 worker")
            message = self._receive(remaining, None)
            if message.get("type") in ("cancelled", "result") or (
                    message.get("type") is None and "raw" in message and "finish_reason" in message):
                return  # 完成与取消竞态时消费终态，避免旧响应污染下一次请求。
            if message.get("type") != "delta":
                raise MossError("MOSS worker 返回无效取消响应")

    def _stop(self, reason="shutdown"):
        process, self._process = self._process, None
        if process is not None:
            # 只记录退出类别与自有 PID，不记录临床文本、输入路径或任务凭据。
            logger.warning("MOSS worker stop: reason=%s pid=%s returncode=%s", reason, process.pid, process.poll())
            # 只终止本实例通过 start_new_session 创建的 worker 组，包含 vLLM 子进程。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            process.stdin.close()
        if self._reader is not None:
            os.close(self._reader)
            self._reader = None
        self._buffer.clear()

    def close(self):
        # 正常停服由请求取消/超时先结束持锁推理；不与活跃请求交叉操作管道。
        with self._lock:
            self._stop()
