import asyncio
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

from app import server
from app.services import live_session
from app.services.kws import (
    ContextTimeline,
    KwsDetection,
    _parse_result,
    _pcm16_to_float32,
    build_state_event,
    create_kws_service,
    parse_context_slots,
)
from app.services.recording_store import RecordingStore
from scripts.build_kws_keywords import build
from tests.test_live_admission import admission
from tests.test_streaming_pipeline import FakeStreamingService, pcm
from tests.test_transcription_priority import _WebSocket, _packetize


class KwsServiceTest(unittest.TestCase):
    def test_create_kws_service_is_disabled_by_default(self):
        self.assertFalse(create_kws_service({"kws_enabled": False}).enabled)

    def test_enabled_with_missing_paths_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "KWS_TOKENS_PATH"):
            create_kws_service({"kws_enabled": True})

    def test_label_is_category_and_value_timed_back_from_detection(self):
        # token 时间戳相对 sherpa 内部 reset，只取跨度：起止由检测时刻倒推。
        detection = _parse_result("bed:12床", [0.08, 0.36, 0.64], detected_ms=27040)
        self.assertEqual(detection, KwsDetection("bed", "12床", start_ms=26080, end_ms=26640))
        self.assertEqual(_parse_result("command:next_bed", [], 500),
                         KwsDetection("command", "next_bed", start_ms=None, end_ms=500))
        self.assertEqual(_parse_result("体温", [0.1], 500).category, "marker")

    def test_context_slots_are_validated(self):
        self.assertEqual(parse_context_slots(" bed, ward ,bed"), ("bed", "ward"))
        with self.assertRaises(ValueError):
            parse_context_slots("bed,Bad-Slot")

    def test_pcm16_to_float32_trims_odd_byte(self):
        samples = _pcm16_to_float32(b"\x00\x40\xff")
        self.assertEqual(samples.shape[0], 1)
        self.assertAlmostEqual(float(samples[0]), 0.5, places=5)

    def test_state_event(self):
        event = build_state_event(session_id="s1", seq=2, ts_ms=1000, source="kws",
                                  slot="bed", value="12床", previous=None)
        self.assertEqual((event["type"], event["slot"], event["value"]), ("state.changed", "bed", "12床"))


class ContextTimelineTest(unittest.TestCase):
    def test_segment_takes_context_at_start_with_lead_window(self):
        context = ContextTimeline(["bed", "ward"])
        context.set("ward", "三病区", 0)
        context.set("bed", "12床", 2300)  # 片段起点后 300ms 才说出床号
        self.assertEqual(context.at(2000, 5000), ({"bed": "12床", "ward": "三病区"}, False))

    def test_change_inside_segment_is_flagged_not_applied(self):
        context = ContextTimeline(["bed"])
        context.set("bed", "12床", 0)
        context.set("bed", "15床", 4000)
        self.assertEqual(context.at(1000, 6000), ({"bed": "12床"}, True))
        self.assertEqual(context.at(3500, 6000), ({"bed": "15床"}, False))

    def test_same_value_is_not_a_change_and_late_insert_keeps_time_order(self):
        context = ContextTimeline(["bed"])
        self.assertEqual(context.set("bed", "12床", 5000), (True, None))
        self.assertEqual(context.set("bed", "12床", 6000), (False, "12床"))
        # KWS 起点可能早于先到达的客户端设置：只修正该时段，当前值不变。
        self.assertEqual(context.set("bed", "15床", 3000), (False, "12床"))
        self.assertEqual(context.at(0, 2000), ({"bed": None}, False))
        self.assertEqual(context.at(2500, 4000), ({"bed": "15床"}, False))
        self.assertEqual(context.at(4500, 7000), ({"bed": "12床"}, False))
        self.assertEqual(context.set("bed", "16床", 7000), (True, "12床"))

    def test_repeated_sets_at_same_time_do_not_accumulate(self):
        context = ContextTimeline(["bed"])
        for i in range(1000):
            context.set("bed", f"{i % 2}床", 2000)
        self.assertEqual(len(context._changes["bed"][0]), 1)
        self.assertEqual(context.at(2000, 3000), ({"bed": "1床"}, False))


@unittest.skipUnless(importlib.util.find_spec("pypinyin"), "需要 pypinyin")
class KeywordsBuilderTest(unittest.TestCase):
    VOCABULARY = {"sh", "í", "èr", "ch", "uáng", "x", "ià", "y", "ī"}

    def test_builds_tokens_and_keeps_options_and_label(self):
        output, errors = build(["# 注释", "", "十二床 @bed:12床", "下一床 :1.5 #0.3 @command:next_bed"], self.VOCABULARY)
        self.assertEqual(errors, [])
        self.assertEqual(output, ["sh í èr ch uáng @bed:12床", "x ià y ī ch uáng :1.5 #0.3 @command:next_bed"])

    def test_rejects_missing_label_and_unknown_tokens(self):
        output, errors = build(["十二床", "12床 @bed:12床"], self.VOCABULARY)
        self.assertEqual(output, [])
        self.assertEqual(len(errors), 2)


class FakeKwsService:
    enabled = True
    context_slots = ("bed",)

    def __init__(self, detections=(), fail=False):
        self.detections = list(detections)  # (检测时刻, KwsDetection)
        self.fail = fail
        self.calls = []

    def create_session(self):
        return self

    def accept_pcm(self, pcm_bytes, sample_rate, end_ms):
        self.calls.append(end_ms)
        if self.fail:
            raise RuntimeError("kws broken")
        if self.detections and end_ms >= self.detections[0][0]:
            return [self.detections.pop(0)[1]]
        return []


def bed(value, start_ms):
    return KwsDetection("bed", value, start_ms=start_ms, end_ms=start_ms + 500)


def control(payload):
    return {"type": "websocket.receive", "text": json.dumps(payload)}


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

    def run_session(self, kws, admission_control=None, before=(), between=()):
        # 暂停会 flush，得到两个最终片段：0-2100ms 与 2100-3200ms。
        events = list(before)
        events += [{"type": "websocket.receive", "bytes": c} for c in _packetize(pcm(2.1), 683)]
        events.append(control({"type": "pause_recording"}))
        events += list(between)
        events.append(control({"type": "resume_recording"}))
        events += [{"type": "websocket.receive", "bytes": c} for c in _packetize(pcm(1.1), 683)]
        events.append(control({"type": "stop_recording"}))
        ws = _WebSocket(events, {"session_id": ["round-1"]})
        with patch.object(server, "kws_service", kws), \
                patch.object(live_session, "ADMISSION", admission_control or admission("off")):
            asyncio.run(server.websocket_live(ws))
        return ws.sent_json

    @staticmethod
    def finals(sent):
        return [e for e in sent if e["type"] == "transcript" and e["isFinal"]]

    def test_slot_detections_update_context_by_audio_time(self):
        kws = FakeKwsService([(500, bed("12床", 100)), (2700, bed("15床", 2300))])
        sent = self.run_session(kws)

        events = [e for e in sent if e["type"] in ("kws.detected", "state.changed")]
        self.assertEqual([e["seq"] for e in events], [1, 2, 3, 4])
        self.assertTrue(all(e["session_id"] == "round-1" for e in events))
        states = [(e["value"], e["previous"], e["ts_ms"], e["source"]) for e in events if e["type"] == "state.changed"]
        self.assertEqual(states, [("12床", None, 100, "kws"), ("15床", "12床", 2300, "kws")])
        self.assertEqual([(e["context"], e["contextChanged"]) for e in self.finals(sent)],
                         [({"bed": "12床"}, False), ({"bed": "15床"}, False)])
        self.assertEqual(sent[-1]["transcriptionStatus"], "complete")

    def test_non_slot_category_is_event_only_and_mid_segment_change_is_flagged(self):
        kws = FakeKwsService([
            (300, KwsDetection("command", "next_bed", start_ms=100, end_ms=300)),
            (1900, bed("15床", 1500)),
        ])
        sent = self.run_session(kws)

        self.assertEqual([e["value"] for e in sent if e["type"] == "state.changed"], ["15床"])
        self.assertEqual([(e["context"], e["contextChanged"]) for e in self.finals(sent)],
                         [({"bed": None}, True), ({"bed": "15床"}, False)])

    def test_set_context_corrects_state_and_validates_input(self):
        sent = self.run_session(
            FakeKwsService([(500, bed("12床", 100))]),
            between=[
                control({"type": "set_context", "slot": "bed", "value": " 15床 "}),
                control({"type": "set_context", "slot": "bed", "value": "15床"}),
                control({"type": "set_context", "slot": "ward", "value": "三病区"}),
                control({"type": "set_context", "slot": "bed", "value": ""}),
            ],
        )

        states = [(e["value"], e["source"], e["ts_ms"]) for e in sent if e["type"] == "state.changed"]
        self.assertEqual(states, [("12床", "kws", 100), ("15床", "client", 2100)])
        self.assertEqual(len([e for e in sent if e["type"] == "error"]), 2)
        self.assertEqual([e["context"] for e in self.finals(sent)], [{"bed": "12床"}, {"bed": "15床"}])

    def test_late_kws_detection_does_not_override_newer_client_context(self):
        # 15床 在 2000ms 说出，但到 2200ms 才检出；客户端已在 2100ms 设为 12床。
        sent = self.run_session(
            FakeKwsService([(2200, bed("15床", 2000))]),
            between=[control({"type": "set_context", "slot": "bed", "value": "12床"})],
        )

        self.assertEqual([e["value"] for e in sent if e["type"] == "kws.detected"], ["15床"])
        states = [(e["value"], e["previous"], e["source"]) for e in sent if e["type"] == "state.changed"]
        self.assertEqual(states, [("12床", None, "client")])
        self.assertEqual([e["context"] for e in self.finals(sent)], [{"bed": None}, {"bed": "12床"}])

    def test_kws_failure_keeps_transcription_and_client_context(self):
        kws = FakeKwsService(fail=True)
        sent = self.run_session(kws, between=[control({"type": "set_context", "slot": "bed", "value": "15床"})])

        self.assertEqual(len(kws.calls), 1)
        self.assertEqual([e.get("phase") for e in sent if e["type"] == "status"], ["kws_unavailable"])
        self.assertEqual([e["context"] for e in self.finals(sent)], [{"bed": None}, {"bed": "15床"}])
        self.assertEqual(sent[-1]["transcriptionStatus"], "complete")

    def test_recording_only_session_keeps_kws(self):
        kws = FakeKwsService([(500, bed("12床", 100))])
        sent = self.run_session(kws, admission("enforce", max_sessions=0))

        self.assertEqual(sent[0]["phase"], "recording_only")
        self.assertGreater(len(kws.calls), 0)
        self.assertEqual([e["type"] for e in sent if e["type"].startswith(("kws.", "state."))],
                         ["kws.detected", "state.changed"])
        self.assertFalse(any(e["type"] == "transcript" for e in sent))
        self.assertEqual(sent[-1]["transcriptionStatus"], "recording_only")

    def test_disabled_kws_leaves_transcript_protocol_unchanged(self):
        sent = self.run_session(
            create_kws_service({"kws_enabled": False}),
            before=[control({"type": "set_context", "slot": "bed", "value": "12床"})],
        )

        self.assertEqual(len(self.finals(sent)), 2)
        self.assertTrue(all("context" not in e and "contextChanged" not in e for e in self.finals(sent)))
        self.assertEqual([e["type"] for e in sent if e["type"] == "error"], ["error"])


if __name__ == "__main__":
    unittest.main()
