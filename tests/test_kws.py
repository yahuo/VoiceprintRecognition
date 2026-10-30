import asyncio
import tempfile
import unittest
from unittest.mock import patch

from app import server
from app.services import live_session
from app.services.kws import (
    KwsDetection,
    _categorize_keyword,
    _normalize_keyword,
    _pcm16_to_float32,
    build_state_change_event,
    create_kws_service,
)
from app.services.recording_store import RecordingStore
from tests.test_live_admission import admission
from tests.test_streaming_pipeline import FakeStreamingService, pcm
from tests.test_transcription_priority import _WebSocket, _packetize


class KwsServiceTest(unittest.TestCase):
    def test_create_kws_service_is_disabled_by_default(self):
        service = create_kws_service({"kws_enabled": False})

        self.assertFalse(service.enabled)
        self.assertFalse(service.create_session().enabled)

    def test_missing_model_paths_disable_kws_without_raising(self):
        service = create_kws_service({"kws_enabled": True})

        self.assertFalse(service.enabled)
        self.assertIn("KWS 配置缺失", service.reason)

    def test_keyword_normalization_supports_alias_and_commands(self):
        self.assertEqual(_normalize_keyword("shí èr chuáng @12床"), "12床")
        self.assertEqual(_normalize_keyword("下一床"), "next_bed")

    def test_keyword_category(self):
        self.assertEqual(_categorize_keyword("12床"), "bed")
        self.assertEqual(_categorize_keyword("next_bed"), "command")
        self.assertEqual(_categorize_keyword("重点"), "marker")

    def test_pcm16_to_float32_trims_odd_byte(self):
        samples = _pcm16_to_float32(b"\x00\x40\xff")

        self.assertEqual(samples.shape[0], 1)
        self.assertAlmostEqual(float(samples[0]), 0.5, places=5)

    def test_state_change_event(self):
        detection = KwsDetection(
            keyword="十二床",
            normalized="12床",
            category="bed",
            score=0.86,
        )

        event = build_state_change_event(
            session_id="s1",
            seq=2,
            current_bed="12床",
            detection=detection,
            ts_ms=1000,
        )

        self.assertEqual(event["type"], "state.current_bed_changed")
        self.assertEqual(event["current_bed"], "12床")
        self.assertEqual(event["confidence"], 0.86)


class FakeKwsService:
    enabled = True
    reason = ""

    def __init__(self, beds_by_ms=(), fail=False):
        self.beds_by_ms = list(beds_by_ms)
        self.fail = fail
        self.calls = []

    def create_session(self):
        return self

    def accept_pcm(self, pcm_bytes, sample_rate, end_ms):
        self.calls.append(end_ms)
        if self.fail:
            raise RuntimeError("kws broken")
        if self.beds_by_ms and end_ms >= self.beds_by_ms[0][0]:
            _, bed = self.beds_by_ms.pop(0)
            return [KwsDetection(keyword=bed, normalized=bed, category="bed", score=0.9, end_ms=end_ms)]
        return []


class KwsWebSocketTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(server, "service", FakeStreamingService()),
            patch.object(server, "recording_store", RecordingStore(self.tmpdir.name)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmpdir.cleanup()

    def run_session(self, kws, control=None):
        # 暂停会 flush，得到两个最终片段：0-2100ms 与 2100-3200ms。
        events = [{"type": "websocket.receive", "bytes": c} for c in _packetize(pcm(2.1), 683)]
        events.append({"type": "websocket.receive", "text": '{"type":"pause_recording"}'})
        events.append({"type": "websocket.receive", "text": '{"type":"resume_recording"}'})
        events += [{"type": "websocket.receive", "bytes": c} for c in _packetize(pcm(1.1), 683)]
        events.append({"type": "websocket.receive", "text": '{"type":"stop_recording"}'})
        ws = _WebSocket(events, {"session_id": ["round-1"]})
        with patch.object(server, "kws_service", kws), \
                patch.object(live_session, "ADMISSION", control or admission("off")):
            asyncio.run(server.websocket_live(ws))
        return ws.sent_json

    def test_bed_events_and_segment_bed_follow_audio_order(self):
        sent = self.run_session(FakeKwsService([(0, "12床"), (2200, "15床")]))

        kws_events = [e for e in sent if e["type"] in ("kws.detected", "state.current_bed_changed")]
        self.assertEqual([e["seq"] for e in kws_events], [1, 2, 3, 4])
        self.assertTrue(all(e["session_id"] == "round-1" for e in kws_events))
        beds = [e["current_bed"] for e in kws_events if e["type"] == "state.current_bed_changed"]
        self.assertEqual(beds, ["12床", "15床"])
        finals = [e for e in sent if e["type"] == "transcript" and e["isFinal"]]
        self.assertEqual([e["bed"] for e in finals], ["12床", "15床"])
        self.assertEqual(sent[-1]["transcriptionStatus"], "complete")

    def test_kws_failure_keeps_transcription_and_bed_field(self):
        kws = FakeKwsService(fail=True)
        sent = self.run_session(kws)

        self.assertEqual(len(kws.calls), 1)
        self.assertEqual([e.get("phase") for e in sent if e["type"] == "status"], ["kws_unavailable"])
        finals = [e for e in sent if e["type"] == "transcript" and e["isFinal"]]
        self.assertEqual([e["bed"] for e in finals], [None, None])
        self.assertEqual(sent[-1]["transcriptionStatus"], "complete")

    def test_recording_only_session_skips_kws(self):
        kws = FakeKwsService([(0, "12床")])
        sent = self.run_session(kws, admission("enforce", max_sessions=0))

        self.assertEqual(kws.calls, [])
        self.assertFalse(any(e["type"].startswith(("kws.", "state.")) for e in sent))

    def test_disabled_kws_leaves_transcript_protocol_unchanged(self):
        sent = self.run_session(create_kws_service({"kws_enabled": False}))

        finals = [e for e in sent if e["type"] == "transcript" and e["isFinal"]]
        self.assertEqual(len(finals), 2)
        self.assertTrue(all("bed" not in e for e in finals))


if __name__ == "__main__":
    unittest.main()
