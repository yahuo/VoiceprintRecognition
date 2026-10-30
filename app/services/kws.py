#!/usr/bin/env python3
"""
实时 KWS 服务。

KWS 是实时链路的旁路能力：默认关闭，只有显式配置模型路径后才加载
sherpa-onnx。这样本模块不会影响现有 ASR / 声纹服务启动。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


COMMAND_NORMALIZATION = {
    "下一床": "next_bed",
    "上一床": "previous_bed",
    "开始记录": "start_recording",
    "暂停记录": "pause_recording",
    "继续记录": "resume_recording",
    "结束记录": "stop_recording",
}


@dataclass(frozen=True)
class KwsDetection:
    keyword: str
    normalized: str
    category: str
    score: float | None = None
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
            "keyword": self.keyword,
            "normalized": self.normalized,
            "score": self.score,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


class KwsService:
    def __init__(self, enabled: bool, reason: str = ""):
        self.enabled = enabled
        self.reason = reason

    def create_session(self):
        return DisabledKwsSession(self.reason)


class DisabledKwsSession:
    enabled = False

    def __init__(self, reason: str = ""):
        self.reason = reason

    def accept_pcm(self, pcm: bytes, sample_rate: int, end_ms: int) -> list[KwsDetection]:
        return []


class SherpaOnnxKwsService(KwsService):
    def __init__(self, config: dict[str, Any]):
        super().__init__(enabled=True)
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx 未安装，无法启用 KWS") from exc

        self._sherpa_onnx = sherpa_onnx
        self.sample_rate = int(config["kws_sample_rate"])
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
        return SherpaOnnxKwsSession(
            self._spotter,
            sample_rate=self.sample_rate,
            duplicate_cooldown_ms=self.duplicate_cooldown_ms,
        )


class SherpaOnnxKwsSession:
    enabled = True

    def __init__(self, spotter, sample_rate: int, duplicate_cooldown_ms: int):
        self._spotter = spotter
        self._stream = spotter.create_stream()
        self._sample_rate = sample_rate
        self._duplicate_cooldown_ms = duplicate_cooldown_ms
        self._last_detected: dict[str, int] = {}

    def accept_pcm(self, pcm: bytes, sample_rate: int, end_ms: int) -> list[KwsDetection]:
        if not pcm:
            return []
        if sample_rate != self._sample_rate:
            raise ValueError(f"KWS 仅支持 {self._sample_rate}Hz PCM，收到 {sample_rate}Hz")

        samples = _pcm16_to_float32(pcm)
        if samples.size == 0:
            return []

        self._stream.accept_waveform(sample_rate, samples)

        detections = []
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                detection = _parse_result(result, end_ms=end_ms)
                if self._should_emit(detection, end_ms):
                    detections.append(detection)
                self._spotter.reset_stream(self._stream)

        return detections

    def _should_emit(self, detection: KwsDetection, now_ms: int) -> bool:
        last_ms = self._last_detected.get(detection.normalized)
        if last_ms is not None and now_ms - last_ms < self._duplicate_cooldown_ms:
            return False
        self._last_detected[detection.normalized] = now_ms
        return True


def create_kws_service(config: dict[str, Any]) -> KwsService:
    if not config.get("kws_enabled"):
        return KwsService(enabled=False, reason="KWS_ENABLED 未启用")

    required_paths = [
        "kws_tokens_path",
        "kws_encoder_path",
        "kws_decoder_path",
        "kws_joiner_path",
        "kws_keywords_file",
    ]
    missing = [key for key in required_paths if not config.get(key)]
    if missing:
        return KwsService(enabled=False, reason=f"KWS 配置缺失: {', '.join(missing)}")

    try:
        return SherpaOnnxKwsService(config)
    except Exception as exc:
        return KwsService(enabled=False, reason=str(exc))


def build_state_change_event(
    *,
    session_id: str,
    seq: int,
    current_bed: str,
    detection: KwsDetection,
    ts_ms: int,
) -> dict[str, Any]:
    return {
        "type": "state.current_bed_changed",
        "session_id": session_id,
        "seq": seq,
        "ts_ms": ts_ms,
        "source": "state_machine",
        "current_bed": current_bed,
        "reason": "kws.detected",
        "confidence": detection.score,
    }


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    even_length = len(pcm) - (len(pcm) % 2)
    if even_length <= 0:
        return np.array([], dtype=np.float32)
    samples = np.frombuffer(pcm[:even_length], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def _parse_result(result: str, end_ms: int) -> KwsDetection:
    keyword = result.strip()
    normalized = _normalize_keyword(keyword)
    category = _categorize_keyword(normalized)
    return KwsDetection(
        keyword=keyword,
        normalized=normalized,
        category=category,
        score=None,
        end_ms=end_ms,
    )


def _normalize_keyword(keyword: str) -> str:
    keyword = keyword.strip()
    if "@" in keyword:
        return keyword.rsplit("@", 1)[1].strip() or keyword
    return COMMAND_NORMALIZATION.get(keyword, keyword)


def _categorize_keyword(normalized: str) -> str:
    if normalized in COMMAND_NORMALIZATION.values():
        return "command"
    if "床" in normalized:
        return "bed"
    return "marker"
