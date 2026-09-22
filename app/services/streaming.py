"""实时音频适配：固定帧 FSMN-VAD、会话独立的 Paraformer 增量缓存。"""

from dataclasses import dataclass


SAMPLE_RATE = 16000
SAMPLE_BYTES = 2
VAD_CHUNK_MS = 200
ASR_CHUNK_SAMPLES = 9600  # Paraformer chunk_size=[0, 10, 5]，600ms


@dataclass(frozen=True)
class SpeechSegment:
    audio: bytes
    start_ms: int
    end_ms: int


class FsmnVadSegmenter:
    """按模型端点切分 PCM；保留起点回看，丢弃静音历史，限制连续讲话长度。

    infer(audio, cache, is_final=..., silence_ms=...) 返回 FSMN 的
    [start_ms, end_ms]，-1 表示该端点尚未检测到。缓存仅属于本实例。
    """

    def __init__(self, infer, *, silence_ms=500, max_segment_ms=12000):
        if silence_ms < 200 or max_segment_ms < VAD_CHUNK_MS:
            raise ValueError("invalid VAD duration")
        self.infer = infer
        self.silence_ms = silence_ms
        self.max_samples = max_segment_ms * SAMPLE_RATE // 1000
        self.chunk_bytes = VAD_CHUNK_MS * SAMPLE_RATE // 1000 * SAMPLE_BYTES
        self.lookback_samples = 2 * SAMPLE_RATE
        self._origin_samples = 0
        self._clear()

    def _clear(self):
        self.cache = {}
        self._pending = bytearray()
        self._buffer = bytearray()
        self._base_sample = 0
        self._samples = 0
        self._start_sample = None
        self._last_end = 0

    def reset(self):
        self._origin_samples += self._samples
        self._clear()

    def _slice(self, start, end):
        return bytes(self._buffer[
            (start - self._base_sample) * SAMPLE_BYTES:
            (end - self._base_sample) * SAMPLE_BYTES
        ])

    @property
    def current_segment_size(self):
        if self._start_sample is None:
            return 0
        return (self._samples - self._start_sample) * SAMPLE_BYTES

    def current_segment(self):
        if self._start_sample is None:
            return b""
        return self._slice(self._start_sample, self._samples)

    @property
    def current_bounds(self):
        start = self._start_sample if self._start_sample is not None else self._samples
        return (
            (self._origin_samples + start) * 1000 // SAMPLE_RATE,
            (self._origin_samples + self._samples) * 1000 // SAMPLE_RATE,
        )

    def _finish(self, end):
        start = self._start_sample
        self._start_sample = None
        end = min(self._samples, max(self._last_end, end))
        self._last_end = end
        if start is None or end <= start:
            return []
        return [SpeechSegment(
            self._slice(start, end),
            (self._origin_samples + start) * 1000 // SAMPLE_RATE,
            (self._origin_samples + end) * 1000 // SAMPLE_RATE,
        )]

    def _consume(self, audio, *, is_final=False):
        self._buffer.extend(audio)
        self._samples += len(audio) // SAMPLE_BYTES
        # 结束时补足一帧给模型排空；补零绝不写入录音或输出片段。
        model_audio = audio
        if is_final and len(model_audio) < self.chunk_bytes:
            model_audio += bytes(self.chunk_bytes - len(model_audio))
        endpoints = self.infer(
            model_audio, self.cache, is_final=is_final, silence_ms=self.silence_ms
        )
        segments = []
        for start_ms, end_ms in endpoints:
            if start_ms >= 0 and self._start_sample is None:
                self._start_sample = min(self._samples, max(
                    self._base_sample, self._last_end,
                    int(start_ms * SAMPLE_RATE / 1000),
                ))
            if end_ms >= 0:
                segments.extend(self._finish(int(end_ms * SAMPLE_RATE / 1000)))

        while (
            self._start_sample is not None
            and self._samples - self._start_sample >= self.max_samples
        ):
            end = self._start_sample + self.max_samples
            segments.extend(self._finish(end))
            self._start_sample = end

        if is_final:
            segments.extend(self._finish(self._samples))
        keep_from = (
            self._start_sample if self._start_sample is not None
            else max(0, self._samples - self.lookback_samples)
        )
        drop = max(0, keep_from - self._base_sample)
        del self._buffer[:drop * SAMPLE_BYTES]
        self._base_sample += drop
        return segments

    def feed(self, data):
        self._pending.extend(data)
        segments = []
        while len(self._pending) >= self.chunk_bytes:
            chunk = bytes(self._pending[:self.chunk_bytes])
            del self._pending[:self.chunk_bytes]
            segments.extend(self._consume(chunk))
        return segments

    def flush(self):
        if not self._samples and not self._pending:
            return []
        usable = len(self._pending) - len(self._pending) % SAMPLE_BYTES
        tail = bytes(self._pending[:usable])
        self._pending.clear()
        segments = self._consume(tail, is_final=True)
        self.reset()
        return segments


class IncrementalParaformer:
    """接收可合并的片段快照，但只向模型发送尚未处理的 PCM。

    排队的过期 interim 可以跳过，后续快照仍覆盖所有尚未消费的音频。
    FSMN 的最终端点可能回溯到已发送的尾部静音之前；最终排空不重放它。
    """

    def __init__(self, infer):
        self.infer = infer
        self.cache = {}
        self.consumed_bytes = 0
        self.text = ""
        self.finished = False

    def update(self, audio, *, is_final=False):
        if self.finished:
            raise RuntimeError("stream already finalized")
        if len(audio) < self.consumed_bytes and not is_final:
            raise ValueError("interim audio must grow monotonically")
        chunk_bytes = ASR_CHUNK_SAMPLES * SAMPLE_BYTES
        while len(audio) - self.consumed_bytes >= chunk_bytes:
            end = self.consumed_bytes + chunk_bytes
            self.text += self.infer(
                audio[self.consumed_bytes:end], self.cache, is_final=False
            )
            self.consumed_bytes = end
        if is_final:
            self.text += self.infer(
                audio[self.consumed_bytes:], self.cache, is_final=True
            )
            self.consumed_bytes = max(self.consumed_bytes, len(audio))
            self.finished = True
            self.cache.clear()
        return self.text
