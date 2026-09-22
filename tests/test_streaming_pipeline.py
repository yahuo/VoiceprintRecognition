import asyncio
import tempfile
import threading
from pathlib import Path
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from app import server
from app.core import ModelService
from app.services.recording_store import RecordingStore
from app.services.streaming import FsmnVadSegmenter, IncrementalParaformer
from tests.test_transcription_priority import _WebSocket, _packetize


def pcm(seconds, amplitude=100):
    return np.full(round(seconds * 16000), amplitude, dtype=np.int16).tobytes()


class StreamingVadTest(unittest.TestCase):
    @staticmethod
    def endpoint_model(audio, cache, *, is_final, silence_ms):
        cache["ticks"] = cache.get("ticks", 0) + 1
        if cache["ticks"] == 1:
            return [[150, -1]]
        if cache["ticks"] == 4:
            return [[-1, 750]]
        return []

    def test_packet_boundaries_and_quiet_pcm_do_not_control_detection(self):
        audio = pcm(1, amplitude=10)
        for packet_bytes in (1, 86, 1366, 4096, len(audio)):
            with self.subTest(packet_bytes=packet_bytes):
                vad = FsmnVadSegmenter(self.endpoint_model)
                segments = []
                for offset in range(0, len(audio), packet_bytes):
                    segments.extend(vad.feed(audio[offset:offset + packet_bytes]))
                segments.extend(vad.flush())
                self.assertEqual(len(segments), 1)
                self.assertEqual((segments[0].start_ms, segments[0].end_ms), (150, 750))
                self.assertEqual(segments[0].audio, audio[150 * 32:750 * 32])

    @staticmethod
    def speech_model(audio, cache, *, is_final, silence_ms):
        first = not cache
        cache["samples"] = cache.get("samples", 0) + len(audio) // 2
        if is_final:
            return [[0 if first else -1, cache["samples"] // 16]]
        return [[0, -1]] if first else []

    def test_maximum_duration_has_no_missing_or_duplicate_samples(self):
        vad = FsmnVadSegmenter(self.speech_model, max_segment_ms=1000)
        audio = pcm(3.45)
        segments = vad.feed(audio) + vad.flush()
        self.assertEqual([len(s.audio) for s in segments], [32000] * 3 + [14400])
        self.assertEqual(b"".join(s.audio for s in segments), audio)
        self.assertEqual([(s.start_ms, s.end_ms) for s in segments],
                         [(0, 1000), (1000, 2000), (2000, 3000), (3000, 3450)])

    def test_flush_does_not_include_padding_and_resets_cache_and_origin(self):
        vad = FsmnVadSegmenter(self.speech_model)
        cache_a = vad.cache
        first = vad.feed(pcm(0.25)) + vad.flush()
        cache_b = vad.cache
        second = vad.feed(pcm(0.35)) + vad.flush()
        self.assertIsNot(cache_a, cache_b)
        self.assertEqual(first[0].audio, pcm(0.25))
        self.assertEqual(second[0].audio, pcm(0.35))
        self.assertEqual((second[0].start_ms, second[0].end_ms), (250, 600))
        self.assertEqual(vad.flush(), [])

    def test_silence_buffer_is_bounded(self):
        vad = FsmnVadSegmenter(lambda *_args, **_kwargs: [])
        self.assertEqual(vad.feed(pcm(60, 0)), [])
        self.assertLessEqual(len(vad._buffer), 2 * 16000 * 2)
        self.assertEqual(vad.current_segment_size, 0)

    def test_endpoint_can_retract_trailing_silence(self):
        def model(_audio, cache, **_kwargs):
            cache["ticks"] = cache.get("ticks", 0) + 1
            return [[0, -1]] if cache["ticks"] == 1 else [[-1, 100]]
        vad = FsmnVadSegmenter(model)
        vad.feed(pcm(0.2))
        self.assertEqual(vad.current_segment_size, len(pcm(0.2)))
        segment = vad.feed(pcm(0.2))[0]
        self.assertEqual(segment.audio, pcm(0.1))

    def test_odd_final_byte_is_not_sent_to_model(self):
        vad = FsmnVadSegmenter(self.speech_model)
        segments = vad.feed(pcm(0.21) + b"\x01") + vad.flush()
        self.assertEqual(segments[0].audio, pcm(0.21))


class IncrementalAsrTest(unittest.TestCase):
    def test_skipped_snapshots_do_not_drop_or_repeat_pcm(self):
        calls = []
        def infer(audio, cache, *, is_final):
            calls.append((audio, cache, is_final))
            cache["count"] = cache.get("count", 0) + 1
            return "末" if is_final else "字"
        decoder = IncrementalParaformer(infer)
        audio = np.arange(40000, dtype=np.int16).tobytes()
        self.assertEqual(decoder.update(audio[:19200]), "字")
        self.assertEqual(decoder.update(audio[:76800]), "字字字字")
        self.assertEqual(decoder.update(audio, is_final=True), "字字字字末")
        self.assertEqual(b"".join(c[0] for c in calls), audio)
        self.assertTrue(calls[-1][2])
        self.assertTrue(all(c[1] is calls[0][1] for c in calls))
        self.assertEqual(decoder.cache, {})
        with self.assertRaises(RuntimeError):
            decoder.update(audio, is_final=True)

    def test_final_may_retract_silence_but_interim_may_not(self):
        calls = []
        decoder = IncrementalParaformer(
            lambda audio, _cache, **_kwargs: calls.append(audio) or "字"
        )
        decoder.update(pcm(0.6))
        with self.assertRaises(ValueError):
            decoder.update(pcm(0.5))
        decoder.update(pcm(0.5), is_final=True)
        self.assertEqual(calls, [pcm(0.6), b""])

    def test_model_service_serializes_calls_and_preserves_session_caches(self):
        class FakeModel:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.seen = []
            def generate(self, **kwargs):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                cache = kwargs["cache"]
                self.seen.append(cache)
                cache["used"] = True
                time.sleep(0.01)
                self.active -= 1
                return [{"text": "字"}]
        service = ModelService()
        service.streaming_asr_model = FakeModel()
        caches = [{}, {}]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(service.transcribe_stream_chunk, pcm(0.6), cache,
                                   is_final=False) for cache in caches]
            self.assertEqual([f.result() for f in futures], ["字", "字"])
        self.assertEqual(service.streaming_asr_model.max_active, 1)
        self.assertIsNot(caches[0], caches[1])
        self.assertTrue(all(c["used"] for c in caches))


class FakeStreamingService:
    registered_embeddings = {"speaker-a": np.ones(2)}

    def __init__(self):
        self.refinements = []
        self.stream_chunks = []
        self.vad_caches = []
        self.fail_stream = False
        self.final_text = "句末精修结果"
        self.guarded_calls = []

    def build_matching_scope(self, _ids):
        return None

    def vad_stream(self, audio, cache, **kwargs):
        if not cache:
            self.vad_caches.append(cache)
        return StreamingVadTest.speech_model(audio, cache, **kwargs)

    def transcribe_stream_chunk(self, audio, cache, *, is_final):
        if self.fail_stream:
            raise RuntimeError("first pass failed")
        self.stream_chunks.append((audio, cache, is_final))
        cache["seen"] = True
        return "首遍" if audio else ""

    def transcribe_segment(self, audio, **kwargs):
        self.refinements.append((audio, kwargs))
        return self.final_text

    def extract_embedding(self, _audio):
        return np.ones(2)

    def match_registered_speaker_guarded(self, _emb, **kwargs):
        self.guarded_calls.append(kwargs)
        return None, 0.2

    def match_speaker_fast(self, *_args, **_kwargs):
        raise AssertionError("new pipeline must use guarded matching")

    def get_speaker_name(self, _speaker_id):
        return "未知"


class StreamingWebSocketTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.service = FakeStreamingService()
        self.patches = [
            patch.object(server, "service", self.service),
            patch.object(server, "recording_store", RecordingStore(self.tmpdir.name)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmpdir.cleanup()

    def run_audio(self, *, pause=False, priority="speed"):
        def messages(audio):
            return [{"type": "websocket.receive", "bytes": chunk}
                    for chunk in _packetize(audio, 683)]
        events = messages(pcm(2.1))
        if pause:
            events += [
                {"type": "websocket.receive", "text": '{"type":"pause_recording"}'},
                {"type": "websocket.receive", "text": '{"type":"resume_recording"}'},
            ] + messages(pcm(1.1))
        events.append({"type": "websocket.receive", "text": '{"type":"stop_recording"}'})
        ws = _WebSocket(events, {"priority": priority, "allowed_speaker_ids": ["speaker-a"]})
        asyncio.run(server.websocket_live(ws))
        return ws.sent_json

    def test_native_interims_and_one_final_refinement(self):
        events = self.run_audio()
        transcripts = [e for e in events if e.get("type") == "transcript"]
        self.assertTrue(any(not e["isFinal"] for e in transcripts))
        self.assertEqual(len(self.service.refinements), 1)
        self.assertEqual(self.service.refinements[0][1]["hotwords"], None)
        self.assertEqual(b"".join(c[0] for c in self.service.stream_chunks), pcm(2.1))
        self.assertEqual(len({e["segmentId"] for e in transcripts}), 1)
        self.assertEqual(transcripts[-1]["text"], "句末精修结果")
        self.assertTrue(transcripts[-1]["isFinal"])
        self.assertEqual((transcripts[-1]["start_ms"], transcripts[-1]["end_ms"]), (0, 2100))
        self.assertIsNone(transcripts[-1]["speakerId"])
        self.assertEqual(len(self.service.guarded_calls), 1)
        self.assertEqual(events[-1]["transcriptionStatus"], "complete")

    def test_pause_finalizes_then_starts_new_vad_and_asr_caches(self):
        events = self.run_audio(pause=True, priority="accuracy")
        finals = [e for e in events if e.get("isFinal") and e["type"] == "transcript"]
        self.assertEqual([e["segmentId"] for e in finals], ["segment-1", "segment-2"])
        self.assertEqual((finals[1]["start_ms"], finals[1]["end_ms"]), (2100, 3200))
        self.assertEqual(len(self.service.refinements), 2)
        self.assertEqual(self.service.refinements[0][1]["hotwords"], server.ACCURACY_HOTWORDS)
        self.assertEqual(len(self.service.vad_caches), 2)
        self.assertIsNot(self.service.vad_caches[0], self.service.vad_caches[1])
        self.assertIsNot(self.service.stream_chunks[0][1], self.service.stream_chunks[-1][1])

    def test_empty_refinement_retains_first_pass_with_explicit_degradation(self):
        self.service.final_text = ""
        events = self.run_audio()
        finals = [e for e in events if e.get("isFinal") and e["type"] == "transcript"]
        self.assertTrue(finals[-1]["degraded"])
        self.assertIn("首遍", finals[-1]["text"])
        self.assertEqual(events[-1]["transcriptionStatus"], "failed")

    def test_cancelling_during_stop_does_not_leave_partial_recording(self):
        entered, release = threading.Event(), threading.Event()
        def refine(*_args, **_kwargs):
            entered.set()
            release.wait(5)
            return "未提交"
        self.service.transcribe_segment = refine
        async def run():
            ws = _WebSocket([
                {"type": "websocket.receive", "bytes": pcm(2.1)},
                {"type": "websocket.receive", "text": '{"type":"stop_recording"}'},
            ])
            task = asyncio.create_task(server.websocket_live(ws))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
                self.assertEqual(list(Path(self.tmpdir.name).iterdir()), [])
                self.assertFalse(any(x["type"] == "recording_saved" for x in ws.sent_json))
            finally:
                release.set()
        asyncio.run(run())

    def test_first_pass_failure_does_not_prevent_final_refinement(self):
        self.service.fail_stream = True
        events = self.run_audio()
        finals = [e for e in events if e.get("isFinal") and e["type"] == "transcript"]
        self.assertEqual(finals[-1]["text"], "句末精修结果")
        self.assertTrue(any(e.get("phase") == "fallback" for e in events))
        self.assertEqual(events[-1]["transcriptionStatus"], "complete")


class SafetyTest(unittest.TestCase):
    def test_health_does_not_expose_credentials(self):
        with patch.object(server, "service", ModelService()), patch.dict(
            server.CONFIG, {"llm_api_key": "secret-test-value", "llm_base_url": "private-url"}
        ):
            response = asyncio.run(server.root())
        self.assertNotIn("llm_api_key", response["config"])
        self.assertNotIn("secret-test-value", str(response))
        self.assertNotIn("private-url", str(response))


if __name__ == "__main__":
    unittest.main()
