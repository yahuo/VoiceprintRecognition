import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

from app import server
from app.services import live_session
from app.services.live_admission import LiveAdmission
from app.services.live_session import InferenceStage
from app.services.recording_store import RecordingStore
from tests.test_streaming_pipeline import FakeStreamingService, pcm
from tests.test_transcription_priority import _WebSocket, _packetize


class FakeNanoStage:
    def __init__(self, utilization=0.0, wait_p95_seconds=None, calls=0):
        self.result = {"calls": calls, "utilization": utilization, "wait_p95_seconds": wait_p95_seconds}
        self.windows = []

    def window(self, seconds):
        self.windows.append(seconds)
        return dict(self.result)


def admission(mode="observe", **limits):
    return LiveAdmission(**{
        "mode": mode, "max_sessions": 16, "nano_utilization": 0.85,
        "nano_wait_p95_seconds": 3.0, "window_seconds": 30, **limits,
    })


class StageWindowTest(unittest.TestCase):
    def test_window_counts_only_calls_finished_inside_it(self):
        stage = InferenceStage("test-window", 1)

        async def run():
            for _ in range(3):
                await stage.run(time.sleep, 0.01)

        asyncio.run(run())
        current = stage.window(30)
        self.assertEqual(current["calls"], 3)
        self.assertGreater(current["utilization"], 0)
        self.assertLessEqual(current["utilization"], 1.0)
        self.assertIsNotNone(current["wait_p95_seconds"])
        later = stage.window(1, now=time.perf_counter() + 10)
        self.assertEqual(later, {"calls": 0, "utilization": 0.0, "wait_p95_seconds": None})


class LiveAdmissionTest(unittest.TestCase):
    def test_invalid_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            admission("strict")
        with self.assertRaises(ValueError):
            admission(window_seconds=0)
        with self.assertRaises(ValueError):
            admission(max_sessions=-1)

    def test_off_never_reads_signals(self):
        control, nano = admission("off"), FakeNanoStage(utilization=1.0)
        self.assertTrue(control.decide(100, nano))
        self.assertNotIn("current", control.snapshot(100, nano))
        self.assertEqual(nano.windows, [])
        self.assertEqual(control.snapshot(0, nano)["counts"]["evaluated"], 0)

    def test_each_limit_is_reported_as_its_own_reason(self):
        control = admission("enforce")
        cases = [
            (16, FakeNanoStage(), ["max_sessions"]),
            (0, FakeNanoStage(utilization=0.85), ["nano_utilization"]),
            (0, FakeNanoStage(wait_p95_seconds=3.0), ["nano_wait_p95"]),
            (15, FakeNanoStage(utilization=0.84, wait_p95_seconds=2.9), []),
            (0, FakeNanoStage(wait_p95_seconds=None), []),
        ]
        for sessions, nano, reasons in cases:
            with self.subTest(sessions=sessions, nano=nano.result):
                with self.assertNoLogs("app.services.live_admission") if not reasons else \
                        self.assertLogs("app.services.live_admission", "WARNING"):
                    self.assertEqual(control.decide(sessions, nano), not reasons)
                self.assertEqual(control.snapshot(0, FakeNanoStage())["recent"][-1]["reasons"], reasons)
                self.assertEqual(nano.windows, [30])
        counts = control.snapshot(0, FakeNanoStage())["counts"]
        self.assertEqual(counts, {"evaluated": 5, "within_limits": 2, "would_degrade": 0, "degraded": 3})

    def test_observe_admits_but_records_would_degrade(self):
        control = admission("observe")
        with self.assertLogs("app.services.live_admission", "WARNING") as logs:
            self.assertTrue(control.decide(20, FakeNanoStage(utilization=0.95, wait_p95_seconds=4.0, calls=200)))
        self.assertIn("would_degrade", logs.output[0])
        snapshot = control.snapshot(3, FakeNanoStage(utilization=0.2))
        self.assertEqual(snapshot["counts"]["would_degrade"], 1)
        self.assertEqual(snapshot["reasons"], {"max_sessions": 1, "nano_utilization": 1, "nano_wait_p95": 1})
        self.assertEqual(snapshot["recent"][0]["outcome"], "would_degrade")
        self.assertEqual(snapshot["recent"][0]["nano_calls"], 200)
        self.assertEqual(snapshot["current"]["reasons"], [])
        self.assertEqual(snapshot["current"]["transcribing_sessions"], 3)
        # 快照只看不计数。
        self.assertEqual(snapshot["counts"]["evaluated"], 1)


class AdmissionWebSocketTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = RecordingStore(self.tmpdir.name)
        self.service = FakeStreamingService()
        self.patches = [
            patch.object(server, "service", self.service),
            patch.object(server, "recording_store", self.store),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmpdir.cleanup()

    def run_session(self, control):
        audio = [pcm(2.1), pcm(1.1, amplitude=200)]
        events = [{"type": "websocket.receive", "bytes": c} for c in _packetize(audio[0], 683)]
        events += [
            {"type": "websocket.receive", "text": '{"type":"pause_recording"}'},
            {"type": "websocket.receive", "bytes": pcm(0.5, amplitude=300)},  # 暂停期间不入录音
            {"type": "websocket.receive", "text": '{"type":"resume_recording"}'},
        ]
        events += [{"type": "websocket.receive", "bytes": c} for c in _packetize(audio[1], 683)]
        events.append({"type": "websocket.receive", "text": '{"type":"stop_recording"}'})
        ws = _WebSocket(events, {"allowed_speaker_ids": ["speaker-a"]})
        with patch.object(live_session, "ADMISSION", control):
            asyncio.run(server.websocket_live(ws))
            metrics = live_session.live_metrics()
        return ws.sent_json, b"".join(audio), metrics

    def saved_pcm(self, event):
        with wave.open(self.store.resolve(event["fileId"]).path, "rb") as wav:
            return wav.readframes(wav.getnframes())

    def test_enforce_degrades_to_recording_only_without_any_model_call(self):
        events, sent, metrics = self.run_session(admission("enforce", max_sessions=0))
        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[0]["phase"], "recording_only")
        self.assertIn("MOSS", events[0]["message"])
        self.assertFalse(any(e["type"] in ("transcript", "error") for e in events))
        self.assertEqual([e["type"] for e in events[1:3]], ["recording_paused", "recording_resumed"])
        self.assertEqual((self.service.vad_caches, self.service.stream_chunks, self.service.refinements), ([], [], []))
        self.assertEqual(self.service.guarded_calls, [])
        saved = events[-1]
        self.assertEqual((saved["type"], saved["transcriptionStatus"]), ("recording_saved", "recording_only"))
        self.assertEqual(self.saved_pcm(saved), sent)
        self.assertEqual((metrics["active_sessions"], metrics["recording_only_sessions"]), (0, 0))
        self.assertEqual(metrics["admission"]["counts"]["degraded"], 1)

    def test_observe_keeps_transcribing_and_counts_would_degrade(self):
        events, sent, metrics = self.run_session(admission("observe", max_sessions=0))
        self.assertFalse(any(e.get("phase") == "recording_only" for e in events))
        finals = [e for e in events if e["type"] == "transcript" and e["isFinal"]]
        self.assertEqual(len(finals), 2)
        self.assertEqual(events[-1]["transcriptionStatus"], "complete")
        self.assertEqual(self.saved_pcm(events[-1]), sent)
        self.assertEqual(metrics["admission"]["counts"]["would_degrade"], 1)
        self.assertEqual(metrics["admission"]["recent"][-1]["reasons"], ["max_sessions"])

    def test_recording_only_sessions_do_not_count_toward_the_cap(self):
        control = admission("enforce", max_sessions=1)

        async def run():
            entered = asyncio.Event()
            release = asyncio.Event()

            class HeldWebSocket(_WebSocket):
                async def receive(self):
                    entered.set()
                    await release.wait()
                    return {"type": "websocket.disconnect"}

            first, second, third = (HeldWebSocket([]) for _ in range(3))
            with patch.object(live_session, "ADMISSION", control):
                tasks = [asyncio.create_task(server.websocket_live(ws)) for ws in (first, second)]
                await entered.wait()
                await asyncio.sleep(0)
                metrics = live_session.live_metrics()
                tasks.append(asyncio.create_task(server.websocket_live(third)))
                await asyncio.sleep(0)
                release.set()
                await asyncio.gather(*tasks)
            return first, second, third, metrics

        first, second, third, metrics = asyncio.run(run())
        self.assertEqual(first.sent_json, [])
        self.assertEqual(second.sent_json[0]["phase"], "recording_only")
        # 只录音会话不占实时名额，第三路仍按 1 路实时转写计数判定。
        self.assertEqual(third.sent_json[0]["phase"], "recording_only")
        self.assertEqual((metrics["active_sessions"], metrics["recording_only_sessions"]), (2, 1))
        self.assertEqual(metrics["admission"]["current"]["transcribing_sessions"], 1)
        self.assertEqual([r["transcribing_sessions"] for r in control.snapshot(0, FakeNanoStage())["recent"]], [0, 1, 1])


class AdmissionMetricsTest(unittest.TestCase):
    def test_metrics_include_admission_without_content(self):
        response = asyncio.run(server.get_live_metrics())
        admission_block = response["admission"]
        self.assertEqual(set(admission_block), {"mode", "limits", "counts", "reasons", "recent", "current"})
        self.assertEqual(set(admission_block["current"]),
                         {"transcribing_sessions", "nano_calls", "nano_utilization", "nano_wait_p95_seconds", "reasons"})
        self.assertIsInstance(response["recording_only_sessions"], int)
        json.dumps(response)


if __name__ == "__main__":
    unittest.main()
