import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import HTTPException
from app import server
from app.services.meeting_jobs import MeetingJobStore
from app.services.moss import MossError, MossUnavailable


class _QueryParams:
    def __init__(self, values=None):
        self.values = values or {}

    def getlist(self, name):
        value = self.values.get(name, [])
        return value if isinstance(value, list) else [value]


class _WebSocket:
    def __init__(self, messages, query_params=None):
        self.messages = list(messages)
        self.query_params = _QueryParams(query_params)
        self.sent_json = []

    async def accept(self):
        pass

    async def receive(self):
        if not self.messages:
            return {"type": "websocket.disconnect"}
        return self.messages.pop(0)

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def close(self, code=1000):
        pass


def _packetize(pcm, packet_samples):
    size = packet_samples * 2
    return [pcm[i:i + size] for i in range(0, len(pcm), size)]


async def _read_streaming_response(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def events(body):
    return [json.loads(x[6:]) for x in body.split("\n\n") if x.startswith("data: ")]


ITEM = {"time": "00:00", "start_ms": 0, "end_ms": 1000, "speakerId": None,
        "speaker": "陌生人1", "diarizationSpeaker": "S01", "confidence": 0.0, "text": "测试文本"}


class TranscriptionPriorityTest(unittest.TestCase):
    def setUp(self):
        self.jobs = MeetingJobStore()
        self.patch = patch.object(server, 'meeting_jobs', self.jobs)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_priority_validation_and_documented_deprecation(self):
        self.assertEqual(server._normalize_transcription_priority(None), "speed")
        self.assertEqual(server._normalize_transcription_priority(" accuracy "), "accuracy")
        with self.assertRaises(HTTPException):
            server._normalize_transcription_priority("nano")
        schema = json.dumps(server.app.openapi(), ensure_ascii=False)
        self.assertIn("processingMode 固定为 unified", schema)
        self.assertNotIn("812.8", schema)

    def test_json_sse_and_both_priorities_share_exactly_one_pipeline(self):
        calls = []
        def process(service, path, threshold, ids, scope, matching, **kwargs):
            self.assertTrue(os.path.exists(path))
            calls.append((ids, matching))
            if kwargs["progress"]:
                kwargs["progress"]({"type": "info", "method": "moss", "total_segments": 1})
                kwargs["progress"]({"type": "segment", "index": 0, **ITEM})
            return [ITEM]
        async def run(path, priority):
            payload = await server._transcribe_meeting_audio(path, "test.wav", None, None, priority)
            response = await server._stream_meeting_transcription(None, ".wav", None, None,
                                                                  priority=priority, source_path=path)
            data = events(await _read_streaming_response(response))
            segment = next(x for x in data if x["type"] == "segment")
            self.assertEqual({k: v for k, v in segment.items() if k not in {"type", "index", "eventId"}}, payload["transcript"][0])
            self.assertEqual(data[-1]["effectivePriority"], priority)
            self.assertEqual(payload["effectivePriority"], priority)
            self.assertEqual(data[-1]["processingMode"], "unified")
            self.assertEqual(payload["processingMode"], "unified")
            self.assertEqual(payload["requestedPriority"], priority)
            self.assertTrue(payload["priorityDeprecated"])
            self.assertIsNone(payload["fallbackReason"])
        with tempfile.NamedTemporaryFile() as f, patch("app.services.meeting.process_meeting", side_effect=process):
            for priority in ("speed", "accuracy"):
                asyncio.run(run(f.name, priority))
        self.assertEqual(calls, [(None, False)] * 4)

    def test_failure_never_emits_done_and_does_not_delete_saved_source(self):
        async def run(path):
            with self.assertRaises(HTTPException) as caught:
                await server._transcribe_meeting_audio(path, "test.wav", None, None)
            self.assertEqual(caught.exception.status_code, 503)
            response = await server._stream_meeting_transcription(None, ".wav", None, None, source_path=path)
            data = events(await _read_streaming_response(response))
            self.assertEqual(data[-1]["type"], "error")
            self.assertNotIn("done", [x["type"] for x in data])
            self.assertTrue(os.path.exists(path))
        with tempfile.NamedTemporaryFile() as f, patch("app.services.meeting.process_meeting", side_effect=MossUnavailable("配置缺失")):
            asyncio.run(run(f.name))

    def test_explicit_job_cancel_waits_for_inference_cleanup_before_removing_temp_source(self):
        started, finished = threading.Event(), threading.Event()
        paths = []
        def process(service, path, *args, cancelled, **kwargs):
            paths.append(path)
            started.set()
            cancelled.wait(2)
            self.assertTrue(os.path.exists(path))
            finished.set()
            raise MossError("cancelled")
        async def run():
            response = await server._stream_meeting_transcription(b"test", ".wav", None, None)
            iterator = response.body_iterator
            await anext(iterator)
            await asyncio.to_thread(started.wait, 2)
            await iterator.aclose()
            self.assertFalse(finished.is_set())
            self.assertTrue(os.path.exists(paths[0]))
            job = next(iter(self.jobs.jobs.values()))
            self.jobs.cancel(job.id)
            await job.task
            self.assertTrue(finished.is_set())
            self.assertFalse(os.path.exists(paths[0]))
        with patch("app.services.meeting.process_meeting", side_effect=process):
            asyncio.run(run())

    def test_live_rejects_unknown_priority(self):
        ws = _WebSocket([], {"priority": "invalid"})
        asyncio.run(server.websocket_live(ws))
        self.assertEqual(ws.sent_json[0]["type"], "error")

    def test_model_service_forwards_and_resets_hotwords(self):
        class Model:
            def __init__(self):
                self.calls = []
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [{"text": "结果"}]
        service = server.ModelService()
        service.asr_model = Model()
        service._prepare_asr_input = lambda x: x
        service.transcribe_segment(b"audio", hotwords=("生命体征",), max_length=4096)
        service.transcribe_segment(b"audio")
        first, second = service.asr_model.calls
        self.assertEqual(first["hotwords"], ["生命体征"])
        self.assertEqual(second["hotwords"], [])
        self.assertEqual(first["max_length"], 4096)
        self.assertEqual(second["max_length"], 200)
        self.assertTrue(first["itn"])
        self.assertNotIn("use_itn", first)

    def test_model_service_serializes_nano_requests_to_isolate_hotwords(self):
        class Model:
            def __init__(self):
                self.active = self.max_active = 0
                self.calls = []
            def generate(self, **kwargs):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(tuple(kwargs["hotwords"]))
                time.sleep(0.02)
                self.active -= 1
                return [{"text": "结果"}]
        service = server.ModelService()
        service.asr_model = Model()
        service._prepare_asr_input = lambda x: x
        with ThreadPoolExecutor(max_workers=2) as pool:
            work = [pool.submit(service.transcribe_segment, b"audio", hotwords=x) for x in [("生命体征",), None]]
            self.assertEqual([f.result() for f in work], ["结果", "结果"])
        self.assertEqual(service.asr_model.max_active, 1)
        self.assertCountEqual(service.asr_model.calls, [("生命体征",), ()])


if __name__ == "__main__":
    unittest.main()
