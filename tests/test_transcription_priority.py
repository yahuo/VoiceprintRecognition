import asyncio
import io
import json
import os
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from app import server
from app.services.recording_store import RecordingStore


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


class _AccuracyService:
    registered_embeddings = {}

    def __init__(self):
        self.transcribed_audio = []
        self.transcribe_max_lengths = []
        self.transcribe_hotwords = []

    def build_matching_scope(self, _allowed_speaker_ids):
        return None

    def transcribe_segment(
        self,
        audio,
        language=None,
        max_length=200,
        hotwords=None,
    ):
        self.transcribed_audio.append(audio)
        self.transcribe_max_lengths.append(max_length)
        self.transcribe_hotwords.append(hotwords)
        if len(self.transcribed_audio) == 1:
            return "给遗传患者录生命体征"
        return "给一床患者录生命体征"


def _packetize(pcm: bytes, packet_samples: int):
    packet_bytes = packet_samples * 2
    return [pcm[offset:offset + packet_bytes] for offset in range(0, len(pcm), packet_bytes)]


def _long_utterance_pcm(duration_seconds=5):
    speech = np.full(duration_seconds * 16000, 1000, dtype=np.int16)
    trailing_silence = np.zeros(int(0.51 * 16000), dtype=np.int16)
    return np.concatenate([speech, trailing_silence]).tobytes()


def _paused_utterance_pcm():
    speech = np.full(16000, 1000, dtype=np.int16)
    pause = np.zeros(int(0.8 * 16000), dtype=np.int16)
    trailing_silence = np.zeros(int(1.51 * 16000), dtype=np.int16)
    return np.concatenate([speech, pause, speech, trailing_silence]).tobytes()


def _wav_bytes(pcm: bytes):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


async def _read_streaming_response(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


class TranscriptionPriorityTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_store = server.recording_store
        self.original_service = server.service
        server.recording_store = RecordingStore(self.tmpdir.name)

    def tearDown(self):
        server.recording_store = self.original_store
        server.service = self.original_service
        self.tmpdir.cleanup()

    def test_priority_defaults_to_speed_and_rejects_unknown_value(self):
        self.assertEqual(server._normalize_transcription_priority(None), "speed")
        self.assertEqual(server._normalize_transcription_priority(""), "speed")
        self.assertEqual(server._normalize_transcription_priority("accuracy"), "accuracy")

        with self.assertRaisesRegex(Exception, "priority"):
            server._normalize_transcription_priority("nano")

    def test_openapi_documents_priority_for_all_offline_endpoints(self):
        schema = server.app.openapi()
        paths = schema["paths"]

        for path in (
            "/v1/meeting/transcribe",
            "/v1/meeting/transcribe/stream",
        ):
            request_schema = paths[path]["post"]["requestBody"]["content"][
                "multipart/form-data"
            ]["schema"]
            component_name = request_schema["$ref"].rsplit("/", 1)[-1]
            priority_schema = schema["components"]["schemas"][component_name][
                "properties"
            ]["priority"]
            self.assertEqual(priority_schema["default"], "speed")
            self.assertIn("812.8", priority_schema["description"])

        for path in (
            "/v1/meeting/recordings/{file_id}/transcribe",
            "/v1/meeting/recordings/{file_id}/transcribe/stream",
        ):
            parameters = paths[path]["post"]["parameters"]
            priority_parameter = next(
                parameter for parameter in parameters if parameter["name"] == "priority"
            )
            self.assertEqual(priority_parameter["schema"]["default"], "speed")
            self.assertIn("812.8", priority_parameter["description"])

    def test_accuracy_output_budget_scales_with_audio_duration(self):
        self.assertEqual(server._accuracy_max_length(13.417), 200)
        self.assertGreater(server._accuracy_max_length(60), 200)
        self.assertEqual(server._accuracy_max_length(100_000), 8192)

        self.assertEqual(
            server._resolve_offline_transcription_priority("accuracy", 812),
            ("accuracy", None),
        )
        self.assertEqual(
            server._resolve_offline_transcription_priority("accuracy", 813),
            ("speed", "accuracy_duration_limit"),
        )
        self.assertEqual(
            server._resolve_offline_transcription_priority("speed", 10_000),
            ("speed", None),
        )

    def test_model_service_forwards_hotwords_with_the_supported_itn_key(self):
        class _AsrModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [{"text": "识别结果"}]

        service = server.ModelService()
        service.asr_model = _AsrModel()
        service._prepare_asr_input = lambda audio: audio

        text = service.transcribe_segment(
            b"audio",
            max_length=4096,
            hotwords=("生命体征",),
        )

        self.assertEqual(text, "识别结果")
        service.transcribe_segment(b"speed audio")

        accuracy_kwargs, speed_kwargs = service.asr_model.calls
        self.assertEqual(accuracy_kwargs["hotwords"], ["生命体征"])
        self.assertEqual(speed_kwargs["hotwords"], [])
        self.assertEqual(accuracy_kwargs["max_length"], 4096)
        self.assertEqual(speed_kwargs["max_length"], 200)
        self.assertTrue(accuracy_kwargs["itn"])
        self.assertNotIn("use_itn", accuracy_kwargs)

    def test_model_service_serializes_nano_requests_to_isolate_hotwords(self):
        class _AsrModel:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = []
                self.guard = threading.Lock()

            def generate(self, **kwargs):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    self.calls.append(tuple(kwargs["hotwords"]))
                time.sleep(0.02)
                with self.guard:
                    self.active -= 1
                return [{"text": "识别结果"}]

        service = server.ModelService()
        service.asr_model = _AsrModel()
        service._prepare_asr_input = lambda audio: audio

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    service.transcribe_segment,
                    b"accuracy audio",
                    hotwords=("生命体征",),
                ),
                pool.submit(service.transcribe_segment, b"speed audio"),
            ]
            self.assertEqual([future.result() for future in futures], ["识别结果"] * 2)

        self.assertEqual(service.asr_model.max_active, 1)
        self.assertCountEqual(service.asr_model.calls, [("生命体征",), ()])

    def test_fixed_frame_vad_forces_a_maximum_segment_duration(self):
        segmenter = server._FixedFrameVadSegmenter(
            sample_rate=16000,
            sample_width_bytes=2,
            frame_ms=10,
            energy_threshold=500,
            silence_duration=1.5,
            min_segment_duration=0.1,
            max_segment_duration=1.0,
        )
        pcm = np.full(int(2.2 * 16000), 1000, dtype=np.int16).tobytes()

        segments = segmenter.feed(pcm)
        segments.extend(segmenter.flush())

        self.assertEqual(len(segments), 3)
        self.assertLessEqual(max(map(len, segments)), 16000 * 2)

    def test_non_stream_accuracy_uses_one_whole_audio_decode(self):
        service = _AccuracyService()
        server.service = service
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 16000 * 30)
        audio_path = server.recording_store.resolve(file_id).path

        with patch("app.services.meeting.process_meeting") as process_meeting:
            payload = asyncio.run(
                server._transcribe_meeting_audio(
                    audio_path,
                    os.path.basename(audio_path),
                    None,
                    None,
                    "accuracy",
                )
            )

        process_meeting.assert_not_called()
        self.assertEqual(len(service.transcribed_audio), 1)
        self.assertGreater(service.transcribe_max_lengths[0], 200)
        self.assertEqual(service.transcribe_hotwords, [server.ACCURACY_HOTWORDS])
        self.assertEqual(payload["segments"], 1)
        self.assertEqual(payload["transcript"][0]["text"], "给遗传患者录生命体征")
        self.assertEqual(payload["requestedPriority"], "accuracy")
        self.assertEqual(payload["effectivePriority"], "accuracy")
        self.assertIsNone(payload["fallbackReason"])

    def test_non_stream_long_accuracy_falls_back_to_speed_explicitly(self):
        service = _AccuracyService()
        server.service = service
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 16000)
        audio_path = server.recording_store.resolve(file_id).path
        speed_transcript = [{"speaker": "未知", "text": "分段识别结果", "time": "00:00"}]

        with (
            patch(
                "app.server.librosa.get_duration",
                return_value=server.ACCURACY_MAX_AUDIO_SECONDS + 1,
            ),
            patch(
                "app.services.meeting.process_meeting",
                return_value=speed_transcript,
            ) as process_meeting,
        ):
            payload = asyncio.run(
                server._transcribe_meeting_audio(
                    audio_path,
                    os.path.basename(audio_path),
                    None,
                    None,
                    "accuracy",
                )
            )

        process_meeting.assert_called_once()
        self.assertEqual(service.transcribed_audio, [])
        self.assertEqual(payload["transcript"], speed_transcript)
        self.assertEqual(payload["requestedPriority"], "accuracy")
        self.assertEqual(payload["effectivePriority"], "speed")
        self.assertEqual(payload["fallbackReason"], "accuracy_duration_limit")

    def test_non_stream_accuracy_rejects_an_empty_decode(self):
        service = _AccuracyService()
        service.transcribe_segment = lambda *_args, **_kwargs: ""
        server.service = service
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 16000)
        audio_path = server.recording_store.resolve(file_id).path

        with self.assertRaises(server.HTTPException) as raised:
            asyncio.run(
                server._transcribe_meeting_audio(
                    audio_path,
                    os.path.basename(audio_path),
                    None,
                    None,
                    "accuracy",
                )
            )

        self.assertEqual(raised.exception.status_code, 502)

    def test_live_accuracy_revises_the_same_segment(self):
        service = _AccuracyService()
        server.service = service
        messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(_long_utterance_pcm(), 683)
        ]
        messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        websocket = _WebSocket(messages, {"priority": "accuracy"})

        asyncio.run(server.websocket_live(websocket))

        revisions = [
            payload for payload in websocket.sent_json if payload.get("segmentId")
        ]
        self.assertEqual(len(service.transcribed_audio), 2)
        self.assertEqual(len(revisions), 2)
        self.assertEqual(revisions[0]["segmentId"], revisions[1]["segmentId"])
        self.assertEqual([item["revision"] for item in revisions], [1, 2])
        self.assertEqual([item["isFinal"] for item in revisions], [False, True])
        self.assertIn("遗传", revisions[0]["text"])
        self.assertIn("一床", revisions[1]["text"])

    def test_live_accuracy_keeps_short_pauses_in_one_context_window(self):
        pcm = _paused_utterance_pcm()

        speed_service = _AccuracyService()
        server.service = speed_service
        speed_messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(pcm, 683)
        ]
        speed_messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        asyncio.run(server.websocket_live(_WebSocket(speed_messages)))

        accuracy_service = _AccuracyService()
        server.service = accuracy_service
        accuracy_messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(pcm, 683)
        ]
        accuracy_messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        accuracy_websocket = _WebSocket(
            accuracy_messages,
            {"priority": "accuracy"},
        )
        asyncio.run(server.websocket_live(accuracy_websocket))

        self.assertEqual(len(speed_service.transcribed_audio), 2)
        self.assertEqual(
            max(len(audio) for audio in accuracy_service.transcribed_audio),
            len(pcm),
        )
        accuracy_finals = [
            payload
            for payload in accuracy_websocket.sent_json
            if payload.get("segmentId") and payload.get("isFinal")
        ]
        self.assertEqual(len(accuracy_finals), 1)

    def test_sse_accuracy_emits_one_whole_audio_segment(self):
        service = _AccuracyService()
        server.service = service
        response = asyncio.run(
            server._stream_meeting_transcription(
                _wav_bytes(_long_utterance_pcm()),
                ".wav",
                None,
                None,
                priority="accuracy",
            )
        )

        body = asyncio.run(_read_streaming_response(response))
        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        segments = [event for event in events if event.get("type") == "segment"]

        self.assertEqual(len(service.transcribed_audio), 1)
        self.assertEqual(service.transcribe_hotwords, [server.ACCURACY_HOTWORDS])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "给遗传患者录生命体征")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["requestedPriority"], "accuracy")
        self.assertEqual(events[-1]["effectivePriority"], "accuracy")
        self.assertIsNone(events[-1]["fallbackReason"])

    def test_sse_long_accuracy_emits_fallback_and_finishes_as_speed(self):
        class _FallbackService(_AccuracyService):
            upload_asr_backend = "nano"

            def diarize(self, *_args, **_kwargs):
                return []

            def vad_segment(self, _audio):
                return [[0, 1000]]

        service = _FallbackService()
        server.service = service
        duration = int(server.ACCURACY_MAX_AUDIO_SECONDS) + 1

        with patch(
            "app.server.librosa.load",
            return_value=(np.ones(duration, dtype=np.float32), 1),
        ):
            response = asyncio.run(
                server._stream_meeting_transcription(
                    b"audio",
                    ".wav",
                    None,
                    None,
                    priority="accuracy",
                )
            )
            body = asyncio.run(_read_streaming_response(response))

        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        fallback = next(
            event
            for event in events
            if event.get("type") == "status" and event.get("phase") == "fallback"
        )

        self.assertEqual(service.transcribe_hotwords, [None])
        self.assertEqual(fallback["requestedPriority"], "accuracy")
        self.assertEqual(fallback["effectivePriority"], "speed")
        self.assertEqual(fallback["fallbackReason"], "accuracy_duration_limit")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["effectivePriority"], "speed")

    def test_sse_accuracy_emits_error_instead_of_done_for_empty_decode(self):
        for outcome in ("", RuntimeError("decode failed")):
            with self.subTest(outcome=repr(outcome)):
                service = _AccuracyService()

                def transcribe(*_args, **_kwargs):
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome

                service.transcribe_segment = transcribe
                server.service = service
                response = asyncio.run(
                    server._stream_meeting_transcription(
                        _wav_bytes(_long_utterance_pcm()),
                        ".wav",
                        None,
                        None,
                        priority="accuracy",
                    )
                )

                body = asyncio.run(_read_streaming_response(response))
                events = [
                    json.loads(line[6:])
                    for line in body.splitlines()
                    if line.startswith("data: ")
                ]

                self.assertEqual(events[-1]["type"], "error")
                self.assertNotIn("done", [event.get("type") for event in events])

    def test_live_accuracy_degrades_explicitly_when_final_decode_fails(self):
        for final_outcome in ("", RuntimeError("decode failed")):
            with self.subTest(final_outcome=repr(final_outcome)):
                service = _AccuracyService()
                outcomes = iter(["临时文本", final_outcome])

                def transcribe(audio, language=None, max_length=200, hotwords=None):
                    service.transcribed_audio.append(audio)
                    service.transcribe_max_lengths.append(max_length)
                    service.transcribe_hotwords.append(hotwords)
                    outcome = next(outcomes)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome

                service.transcribe_segment = transcribe
                server.service = service
                messages = [
                    {"type": "websocket.receive", "bytes": chunk}
                    for chunk in _packetize(_long_utterance_pcm(), 683)
                ]
                messages.append(
                    {
                        "type": "websocket.receive",
                        "text": '{"type":"stop_recording"}',
                    }
                )
                websocket = _WebSocket(messages, {"priority": "accuracy"})

                asyncio.run(server.websocket_live(websocket))

                revisions = [
                    payload
                    for payload in websocket.sent_json
                    if payload.get("segmentId") and payload.get("type") == "transcript"
                ]
                errors = [
                    payload
                    for payload in websocket.sent_json
                    if payload.get("type") == "error"
                ]
                recording_saved = next(
                    payload
                    for payload in websocket.sent_json
                    if payload.get("type") == "recording_saved"
                )
                self.assertEqual([item["revision"] for item in revisions], [1, 2])
                self.assertTrue(revisions[-1]["isFinal"])
                self.assertTrue(revisions[-1]["degraded"])
                self.assertEqual(revisions[-1]["text"], "临时文本")
                self.assertEqual(errors[-1]["segmentId"], revisions[-1]["segmentId"])
                self.assertEqual(recording_saved["transcriptionStatus"], "failed")

    def test_live_accuracy_low_speaker_confidence_still_sends_final_text(self):
        service = _AccuracyService()
        service.registered_embeddings = {"speaker-a": np.ones(2)}
        service.embedding_calls = 0

        def extract_embedding(_audio):
            service.embedding_calls += 1
            return np.ones(2)

        service.extract_embedding = extract_embedding
        service.match_speaker_fast = lambda *_args, **_kwargs: ("speaker-a", 0.01)
        service.get_speaker_name = lambda _speaker_id: "测试人员"
        server.service = service
        messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(_long_utterance_pcm(), 683)
        ]
        messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        websocket = _WebSocket(
            messages,
            {
                "priority": "accuracy",
                "allowed_speaker_ids": ["speaker-a"],
            },
        )

        asyncio.run(server.websocket_live(websocket))

        revisions = [
            payload for payload in websocket.sent_json if payload.get("segmentId")
        ]
        self.assertEqual(service.embedding_calls, 1)
        self.assertTrue(revisions[-1]["isFinal"])
        self.assertEqual(revisions[-1]["speakerId"], None)
        self.assertEqual(revisions[-1]["speaker"], "未知")
        self.assertIn("一床", revisions[-1]["text"])

    def test_live_accuracy_speaker_failure_keeps_the_final_text(self):
        for fail_at in ("match", "name"):
            with self.subTest(fail_at=fail_at):
                service = _AccuracyService()
                service.registered_embeddings = {"speaker-a": np.ones(2)}
                service.extract_embedding = lambda _audio: np.ones(2)

                def match_speaker(*_args, **_kwargs):
                    if fail_at == "match":
                        raise RuntimeError("speaker match failed")
                    return "speaker-a", 0.99

                def get_speaker_name(_speaker_id):
                    if fail_at == "name":
                        raise RuntimeError("speaker name failed")
                    return "测试人员"

                service.match_speaker_fast = match_speaker
                service.get_speaker_name = get_speaker_name
                server.service = service
                messages = [
                    {"type": "websocket.receive", "bytes": chunk}
                    for chunk in _packetize(_long_utterance_pcm(), 683)
                ]
                messages.append(
                    {
                        "type": "websocket.receive",
                        "text": '{"type":"stop_recording"}',
                    }
                )
                websocket = _WebSocket(
                    messages,
                    {
                        "priority": "accuracy",
                        "allowed_speaker_ids": ["speaker-a"],
                    },
                )

                asyncio.run(server.websocket_live(websocket))

                revisions = [
                    payload
                    for payload in websocket.sent_json
                    if payload.get("segmentId") and payload.get("type") == "transcript"
                ]
                recording_saved = next(
                    payload
                    for payload in websocket.sent_json
                    if payload.get("type") == "recording_saved"
                )
                self.assertTrue(revisions[-1]["isFinal"])
                self.assertIn("一床", revisions[-1]["text"])
                self.assertEqual(revisions[-1]["speaker"], "未知")
                self.assertEqual(recording_saved["transcriptionStatus"], "complete")

    def test_live_speed_does_not_claim_accuracy_transcription_status(self):
        service = _AccuracyService()
        server.service = service
        messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(_long_utterance_pcm(1), 683)
        ]
        messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        websocket = _WebSocket(messages)

        asyncio.run(server.websocket_live(websocket))

        recording_saved = next(
            payload
            for payload in websocket.sent_json
            if payload.get("type") == "recording_saved"
        )
        self.assertEqual(service.transcribe_hotwords, [None])
        self.assertEqual(service.transcribe_max_lengths, [200])
        self.assertNotIn("transcriptionStatus", recording_saved)

    def test_live_accuracy_coalesces_stale_interims_for_fast_upload(self):
        service = _AccuracyService()
        server.service = service
        messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(_long_utterance_pcm(13), 683)
        ]
        messages.append(
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'}
        )
        websocket = _WebSocket(messages, {"priority": "accuracy"})

        asyncio.run(server.websocket_live(websocket))

        revisions = [
            payload for payload in websocket.sent_json if payload.get("segmentId")
        ]
        self.assertEqual(len(service.transcribed_audio), 2)
        self.assertEqual(len(revisions), 2)
        self.assertEqual([item["revision"] for item in revisions], [2, 3])
        self.assertEqual([item["isFinal"] for item in revisions], [False, True])

    def test_live_rejects_unknown_priority(self):
        service = _AccuracyService()
        server.service = service
        websocket = _WebSocket([], {"priority": "nano"})

        asyncio.run(server.websocket_live(websocket))

        self.assertEqual(len(service.transcribed_audio), 0)
        self.assertEqual(websocket.sent_json[0]["type"], "error")
        self.assertIn("priority", websocket.sent_json[0]["message"])


if __name__ == "__main__":
    unittest.main()
