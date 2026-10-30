#!/usr/bin/env python3
"""
实时 KWS 旁路与会话上下文。

KWS 默认关闭；KWS_ENABLED=1 时配置必须完整，否则启动失败，不静默降级。
关键词按类别维护在 KWS_KEYWORDS_STORE（JSON，见 kws_keywords.py）；该文件不存在时从 KWS_KEYWORDS_FILE 导入初始内容。
命中标签为 `类别:值`，例如 `bed:12床`、`command:next_bed`。
上下文槽位类别：命中即切换会话上下文并标注到转写片段；其余类别（指令、标记等）只推送事件，由业务端解释。
保存关键词时重建 KeywordSpotter：新开始的会话用新词表，进行中的会话保持开始时的词表和槽位。
"""

from bisect import bisect_right
from dataclasses import dataclass
import os
import re
import tempfile
from typing import Any

import numpy as np

from .kws_keywords import KeywordsError, KeywordStore, compile_categories, load_vocabulary, parse_raw_lines, phrase_tokens


LABEL_PATTERN = re.compile(r"^([a-z][a-z0-9_]*):(.+)$")
SLOT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_CONTEXT_VALUE_LENGTH = 64
# 片段起点之后这段时间内的切换视为本片段的上下文（VAD 起点通常略早于关键词）。
CONTEXT_LEAD_MS = 1000
# 末个关键词 token 到检测时刻的间隔（chunk-16 模型、40ms 收包实测约 400ms），用于倒推关键词起止时刻。
DETECTION_LAG_MS = 400
REQUIRED_PATHS = ("kws_tokens_path", "kws_encoder_path", "kws_decoder_path", "kws_joiner_path", "kws_keywords_store")


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


@dataclass(frozen=True)
class PreparedKeywords:
    """校验通过、已建好 spotter 但尚未生效的关键词表。"""

    categories: list[dict]
    spotter: Any


class SherpaOnnxKwsService(KwsService):
    enabled = True

    def __init__(self, config: dict[str, Any]):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx 未安装，无法启用 KWS") from exc
        try:
            import pypinyin  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pypinyin 未安装，无法把关键词转成发音") from exc

        self._sherpa = sherpa_onnx
        self._config = config
        self.duplicate_cooldown_ms = int(config["kws_duplicate_cooldown_ms"])
        self._vocabulary = load_vocabulary(config["kws_tokens_path"])
        self.store = KeywordStore(config["kws_keywords_store"])
        document = self.store.load() or _seed_document(config)
        try:
            prepared = self.prepare(document["categories"])
        except KeywordsError as exc:
            raise RuntimeError(f"KWS 关键词表无效: {exc}") from exc
        self.activate(prepared, int(document.get("version") or 0), document.get("updated_at"))

    def preview(self, phrases) -> list[dict]:
        """说法对应的发音 token；供网页在保存前展示和标出无法识别的说法。"""
        result = []
        for phrase in phrases:
            tokens, error = phrase_tokens(" ".join(str(phrase).split()), self._vocabulary)
            result.append({"phrase": phrase, "tokens": " ".join(tokens), "error": error})
        return result

    def prepare(self, categories) -> PreparedKeywords:
        """校验并建好新 spotter；耗时，放在线程里执行。校验失败抛出 KeywordsError。"""
        normalized, lines = compile_categories(categories, self._vocabulary)
        fd, path = tempfile.mkstemp(prefix="kws-keywords-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            spotter = self._sherpa.KeywordSpotter(
                tokens=self._config["kws_tokens_path"],
                encoder=self._config["kws_encoder_path"],
                decoder=self._config["kws_decoder_path"],
                joiner=self._config["kws_joiner_path"],
                num_threads=int(self._config["kws_num_threads"]),
                keywords_file=path,
                provider=self._config["kws_provider"],
            )
        finally:
            os.unlink(path)
        return PreparedKeywords(normalized, spotter)

    def activate(self, prepared: PreparedKeywords, version: int, updated_at: str | None) -> None:
        """切换到新词表；在事件循环中调用，与 create_session、context_slots 的读取不交错。"""
        self._spotter = prepared.spotter
        self.categories = prepared.categories
        self.context_slots = tuple(c["code"] for c in prepared.categories if c["slot"])
        self.version = version
        self.updated_at = updated_at

    def create_session(self):
        return SherpaOnnxKwsSession(self._spotter, duplicate_cooldown_ms=self.duplicate_cooldown_ms)


def _seed_document(config: dict[str, Any]) -> dict:
    path = config.get("kws_keywords_file")
    if not path or not os.path.exists(path):
        raise RuntimeError(f"关键词库 {config['kws_keywords_store']} 不存在，且未提供可导入的 KWS_KEYWORDS_FILE")
    with open(path, encoding="utf-8") as f:
        try:
            categories = parse_raw_lines(f, parse_context_slots(config["kws_context_slots"]))
        except KeywordsError as exc:
            raise RuntimeError(f"KWS_KEYWORDS_FILE 无法导入: {exc}") from exc
    return {"version": 0, "updated_at": None, "categories": categories}


class SherpaOnnxKwsSession:
    def __init__(self, spotter, duplicate_cooldown_ms: int):
        self._spotter = spotter
        self._stream = spotter.create_stream()
        self._duplicate_cooldown_ms = duplicate_cooldown_ms
        self._last_detected: dict[tuple[str, str], int] = {}

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
            detection = _parse_result(result.keyword, result.timestamps, end_ms)
            self._spotter.reset_stream(self._stream)
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
        # 槽位 -> (有序生效时刻, 对应值)；同一时刻只保留最后一次设置，没有新音频时反复 set_context 不会累积。
        self._changes: dict[str, tuple[list[int], list[str | None]]] = {slot: ([], []) for slot in self.slots}

    def set(self, slot: str, value: str | None, at_ms: int) -> tuple[bool, str | None]:
        """在 at_ms 生效；返回 (当前值是否改变, 原当前值)。当前值是时间上最晚的一次设置。

        早于已有设置的插入（KWS 检测延迟期间客户端已 set_context）只修正该时段的片段上下文，不改变当前值。
        """
        times, values = self._changes[slot]
        previous = values[-1] if values else None
        index = bisect_right(times, at_ms)
        if (values[index - 1] if index else None) == value:
            return False, previous
        if index and times[index - 1] == at_ms:
            values[index - 1] = value
        else:
            times.insert(index, at_ms)
            values.insert(index, value)
        return values[-1] != previous, previous

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


def _parse_result(keyword: str, timestamps, detected_ms: int) -> KwsDetection:
    label = keyword.strip()
    match = LABEL_PATTERN.match(label)
    category, value = (match.group(1), match.group(2).strip()) if match else ("marker", label)
    if timestamps:
        # sherpa 连续约 1.5s 无关键词时会在内部 reset，token 时间戳从该处重新计数，而 Python 绑定拿不到这个偏移；
        # 只用 token 跨度，从检测时刻倒推起止。
        end_ms = max(0, detected_ms - DETECTION_LAG_MS)
        start_ms = max(0, end_ms - round((timestamps[-1] - timestamps[0]) * 1000))
    else:
        start_ms, end_ms = None, detected_ms
    return KwsDetection(category=category, value=value, start_ms=start_ms, end_ms=end_ms)
