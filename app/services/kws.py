#!/usr/bin/env python3
"""
实时 KWS 旁路与会话上下文。

KWS 默认关闭；KWS_ENABLED=1 时配置必须完整，否则启动失败，不静默降级。
keywords 文件每行的 @ 标签写成 `类别:值`，例如 `@bed:12床`、`@command:next_bed`。
KWS_CONTEXT_SLOTS 中的类别是上下文槽位：命中即切换会话上下文并标注到转写片段；
其余类别（指令、标记等）只推送事件，由业务端解释。
"""

from bisect import bisect_right
from dataclasses import dataclass
import re
from typing import Any

import numpy as np


LABEL_PATTERN = re.compile(r"^([a-z][a-z0-9_]*):(.+)$")
SLOT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_CONTEXT_VALUE_LENGTH = 64
# 片段起点之后这段时间内的切换视为本片段的上下文（VAD 起点通常略早于关键词）。
CONTEXT_LEAD_MS = 1000
REQUIRED_PATHS = ("kws_tokens_path", "kws_encoder_path", "kws_decoder_path", "kws_joiner_path", "kws_keywords_file")


@dataclass(frozen=True)
class KwsDetection:
    category: str
    value: str
    start_ms: int | None = None
    end_ms: int | None = None

    def to_event(self, session_id: str, seq: int, ts_ms: int) -> dict[str, Any]:
        return {
            "type": "kws.detected",
            "session_id": session_id,
            "seq": seq,
            "ts_ms": ts_ms,
            "source": "kws",
            "category": self.category,
            "value": self.value,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


class KwsService:
    """未启用时的占位：调用方先看 enabled，不创建会话。"""

    enabled = False
    context_slots: tuple[str, ...] = ()


class SherpaOnnxKwsService(KwsService):
    enabled = True

    def __init__(self, config: dict[str, Any]):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx 未安装，无法启用 KWS") from exc

        self.context_slots = parse_context_slots(config["kws_context_slots"])
        self.duplicate_cooldown_ms = int(config["kws_duplicate_cooldown_ms"])
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=config["kws_tokens_path"],
            encoder=config["kws_encoder_path"],
            decoder=config["kws_decoder_path"],
            joiner=config["kws_joiner_path"],
            num_threads=int(config["kws_num_threads"]),
            keywords_file=config["kws_keywords_file"],
            provider=config["kws_provider"],
        )

    def create_session(self):
        return SherpaOnnxKwsSession(self._spotter, duplicate_cooldown_ms=self.duplicate_cooldown_ms)


class SherpaOnnxKwsSession:
    def __init__(self, spotter, duplicate_cooldown_ms: int):
        self._spotter = spotter
        self._stream = spotter.create_stream()
        self._duplicate_cooldown_ms = duplicate_cooldown_ms
        self._last_detected: dict[tuple[str, str], int] = {}
        # sherpa 的 token 时间戳相对最近一次 reset，换算回会话时间轴。
        self._base_ms = 0

    def accept_pcm(self, pcm: bytes, sample_rate: int, end_ms: int) -> list[KwsDetection]:
        samples = _pcm16_to_float32(pcm)
        if samples.size == 0:
            return []

        self._stream.accept_waveform(sample_rate, samples)

        detections = []
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            # 只取一次底层结果：包装层的 get_result()/timestamps() 各取一次，第二次已拿不到时间戳。
            result = self._spotter.keyword_spotter.get_result(self._stream)
            if not result.keyword.strip():
                continue
            detection = _parse_result(result.keyword, result.timestamps, self._base_ms, end_ms)
            self._spotter.reset_stream(self._stream)
            self._base_ms = end_ms
            if self._should_emit(detection, end_ms):
                detections.append(detection)

        return detections

    def _should_emit(self, detection: KwsDetection, now_ms: int) -> bool:
        key = (detection.category, detection.value)
        last_ms = self._last_detected.get(key)
        if last_ms is not None and now_ms - last_ms < self._duplicate_cooldown_ms:
            return False
        self._last_detected[key] = now_ms
        return True


class ContextTimeline:
    """会话上下文槽位随音频时间的变化；按片段时间段查询，结果不受 KWS 检测延迟和入队时机影响。"""

    def __init__(self, slots):
        self.slots = tuple(slots)
        self._current = dict.fromkeys(self.slots)
        # 槽位 -> (有序生效时刻, 对应值)；同一时刻只保留最后一次设置，没有新音频时反复 set_context 不会累积。
        self._changes: dict[str, tuple[list[int], list[str | None]]] = {slot: ([], []) for slot in self.slots}

    def set(self, slot: str, value: str | None, at_ms: int) -> tuple[bool, str | None]:
        previous = self._current[slot]
        if value == previous:
            return False, previous
        self._current[slot] = value
        times, values = self._changes[slot]
        index = bisect_right(times, at_ms)
        if index and times[index - 1] == at_ms:
            values[index - 1] = value
        else:
            times.insert(index, at_ms)
            values.insert(index, value)
        return True, previous

    def at(self, start_ms: int, end_ms: int) -> tuple[dict[str, str | None], bool]:
        """返回片段上下文（起点 + 引导窗口内的最终取值）以及片段其余时段是否发生切换。"""
        lead_end = start_ms + CONTEXT_LEAD_MS
        context = {}
        changed = False
        for slot, (times, values) in self._changes.items():
            index = bisect_right(times, lead_end)
            context[slot] = values[index - 1] if index else None
            changed = changed or bisect_right(times, end_ms) > index
        return context, changed


def parse_context_slots(raw: str) -> tuple[str, ...]:
    slots = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    invalid = [slot for slot in slots if not SLOT_PATTERN.match(slot)]
    if invalid:
        raise ValueError(f"KWS_CONTEXT_SLOTS 含非法槽位名: {', '.join(invalid)}")
    return slots


def create_kws_service(config: dict[str, Any]) -> KwsService:
    if not config.get("kws_enabled"):
        return KwsService()
    missing = [key.upper() for key in REQUIRED_PATHS if not config.get(key)]
    if missing:
        raise RuntimeError(f"KWS_ENABLED=1 但配置缺失: {', '.join(missing)}")
    return SherpaOnnxKwsService(config)


def build_state_event(
    *,
    session_id: str,
    seq: int,
    ts_ms: int,
    source: str,
    slot: str,
    value: str | None,
    previous: str | None,
) -> dict[str, Any]:
    return {
        "type": "state.changed",
        "session_id": session_id,
        "seq": seq,
        "ts_ms": ts_ms,
        "source": source,
        "slot": slot,
        "value": value,
        "previous": previous,
    }


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    even_length = len(pcm) - (len(pcm) % 2)
    if even_length <= 0:
        return np.array([], dtype=np.float32)
    samples = np.frombuffer(pcm[:even_length], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _parse_result(keyword: str, timestamps, base_ms: int, detected_ms: int) -> KwsDetection:
    label = keyword.strip()
    match = LABEL_PATTERN.match(label)
    category, value = (match.group(1), match.group(2).strip()) if match else ("marker", label)
    if timestamps:
        start_ms = base_ms + round(timestamps[0] * 1000)
        end_ms = base_ms + round(timestamps[-1] * 1000)
    else:
        start_ms, end_ms = None, detected_ms
    return KwsDetection(category=category, value=value, start_ms=start_ms, end_ms=end_ms)
