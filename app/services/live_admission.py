"""实时会话准入：按 Nano 句末精修阶段的负载，决定新会话是否实时转写。

每个会话只在开始时判定一次，整场不变：
- off：不评估。
- observe（默认）：照常实时转写，只累计"本应降级"的次数、原因和当时的信号，并写日志，用于校准阈值。
- enforce：超限的新会话只录音、不送任何模型；停止后可用 MOSS 生成会后复核稿。

窗口利用率和排队 p95 统计的是已完成的调用，属于滞后信号；多人同时加入时两者仍接近 0，
这种突发由实时转写会话数的硬上限兜底。只录音会话不占用模型，不计入上限。
"""

from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

MODES = ("off", "observe", "enforce")
REASONS = ("max_sessions", "nano_utilization", "nano_wait_p95")


class LiveAdmission:
    """判定与计数只在事件循环线程上执行，不加锁。"""

    def __init__(self, *, mode, max_sessions, nano_utilization, nano_wait_p95_seconds,
                 window_seconds, recent=32):
        if mode not in MODES:
            raise ValueError(f"LIVE_ADMISSION_MODE 仅支持 {'/'.join(MODES)}，当前为 {mode!r}")
        if max_sessions < 0 or window_seconds <= 0 or nano_utilization <= 0 or nano_wait_p95_seconds <= 0:
            raise ValueError("实时准入阈值必须为正数，硬上限不能为负")
        self.mode = mode
        self.limits = {
            "max_sessions": max_sessions,
            "nano_utilization": nano_utilization,
            "nano_wait_p95_seconds": nano_wait_p95_seconds,
            "window_seconds": window_seconds,
        }
        self._counts = {"evaluated": 0, "within_limits": 0, "would_degrade": 0, "degraded": 0}
        self._reasons = dict.fromkeys(REASONS, 0)
        self._recent = deque(maxlen=recent)

    def _signals(self, transcribing_sessions, nano_stage):
        nano = nano_stage.window(self.limits["window_seconds"])
        reasons = []
        if transcribing_sessions >= self.limits["max_sessions"]:
            reasons.append("max_sessions")
        if nano["utilization"] >= self.limits["nano_utilization"]:
            reasons.append("nano_utilization")
        if nano["wait_p95_seconds"] is not None and nano["wait_p95_seconds"] >= self.limits["nano_wait_p95_seconds"]:
            reasons.append("nano_wait_p95")
        return {
            "transcribing_sessions": transcribing_sessions,
            "nano_calls": nano["calls"],
            "nano_utilization": nano["utilization"],
            "nano_wait_p95_seconds": nano["wait_p95_seconds"],
            "reasons": reasons,
        }

    def decide(self, transcribing_sessions, nano_stage):
        """返回新会话是否实时转写；transcribing_sessions 不含本会话。"""
        if self.mode == "off":
            return True
        signals = self._signals(transcribing_sessions, nano_stage)
        reasons = signals["reasons"]
        outcome = "within_limits" if not reasons else "degraded" if self.mode == "enforce" else "would_degrade"
        self._counts["evaluated"] += 1
        self._counts[outcome] += 1
        for reason in reasons:
            self._reasons[reason] += 1
        self._recent.append({"time": round(time.time(), 3), "outcome": outcome, **signals})
        if reasons:
            logger.warning(
                "live admission %s: transcribing_sessions=%s nano_utilization=%.3f nano_wait_p95=%s reasons=%s",
                outcome, transcribing_sessions, signals["nano_utilization"],
                signals["nano_wait_p95_seconds"], ",".join(reasons),
            )
        return outcome != "degraded"

    def snapshot(self, transcribing_sessions, nano_stage):
        data = {
            "mode": self.mode, "limits": dict(self.limits),
            "counts": dict(self._counts), "reasons": dict(self._reasons),
            "recent": list(self._recent),
        }
        if self.mode != "off":
            # 此刻若有新会话加入会得到的判定；不计数。
            data["current"] = self._signals(transcribing_sessions, nano_stage)
        return data
