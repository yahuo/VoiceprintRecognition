import asyncio
import os
import tempfile
import unittest
import wave
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app import server
from app.services.recording_store import (
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    RecordingStore,
)


class RecordingStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = RecordingStore(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_save_pcm_wav_writes_valid_wav(self):
        file_id = self.store.save_pcm_wav(b"\x01\x00\x02\x00" * 100)
        recording = self.store.resolve(file_id)

        with wave.open(recording.path, "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav_file.getnchannels(), CHANNELS)
            self.assertEqual(wav_file.getsampwidth(), SAMPLE_WIDTH_BYTES)
            self.assertEqual(wav_file.getnframes(), 200)

    def test_save_empty_pcm_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save_pcm_wav(b"")

    def test_cleanup_temp_files_removes_orphaned_wav_tmp(self):
        temp_path = os.path.join(self.tmpdir.name, "orphan.wav.tmp")
        final_path = os.path.join(self.tmpdir.name, "keep.wav")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(b"tmp")
        with open(final_path, "wb") as final_file:
            final_file.write(b"wav")

        removed = self.store.cleanup_temp_files()

        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(temp_path))
        self.assertTrue(os.path.exists(final_path))

    def test_odd_pcm_byte_is_trimmed(self):
        file_id = self.store.save_pcm_wav(b"\x01\x00\x02")
        recording = self.store.resolve(file_id)

        with wave.open(recording.path, "rb") as wav_file:
            self.assertEqual(wav_file.getnframes(), 1)

    def test_delete_many_is_idempotent(self):
        file_id = self.store.save_pcm_wav(b"\x00\x00" * 10)

        first = self.store.delete_many([file_id])
        second = self.store.delete_many([file_id])

        self.assertEqual(first["deleted"], [file_id])
        self.assertEqual(first["missing"], [])
        self.assertEqual(second["deleted"], [])
        self.assertEqual(second["missing"], [file_id])


class RecordingApiTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_store = server.recording_store
        self.original_service = server.service
        server.recording_store = RecordingStore(self.tmpdir.name)
        server.service = FakeService()

    def tearDown(self):
        server.recording_store = self.original_store
        server.service = self.original_service
        self.tmpdir.cleanup()

    def test_download_recording(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)

        response = asyncio.run(server.download_meeting_recording(file_id))

        self.assertIn("audio/wav", response.media_type)
        self.assertIn(
            f"{file_id}.wav",
            response.headers["content-disposition"],
        )

    def test_download_missing_and_invalid_recording(self):
        missing_id = "00000000-0000-0000-0000-000000000000"

        with self.assertRaises(HTTPException) as missing:
            asyncio.run(server.download_meeting_recording(missing_id))
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(server.download_meeting_recording("not-a-file-id"))

        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_transcribe_recording_by_file_id(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)
        transcript = [
            {
                "time": "00:00",
                "speaker": "未知",
                "confidence": 0.0,
                "text": "测试内容",
            }
        ]

        with patch("app.services.meeting.process_meeting", return_value=transcript) as process_meeting:
            payload = asyncio.run(
                server.transcribe_meeting_recording(
                    file_id,
                    threshold=0.42,
                    allowed_speaker_ids=None,
                )
            )

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["segments"], 1)
        self.assertEqual(payload["transcript"], transcript)
        self.assertIn("测试内容", payload["markdown"])
        self.assertIn(f"{file_id}.wav", payload["markdown"])
        args = process_meeting.call_args.args
        self.assertIs(args[0], server.service)
        self.assertEqual(args[1], os.path.join(self.tmpdir.name, f"{file_id}.wav"))
        self.assertEqual(args[2], 0.42)

    def test_transcribe_recording_missing_and_invalid_file_id(self):
        missing_id = "00000000-0000-0000-0000-000000000000"

        with self.assertRaises(HTTPException) as missing:
            asyncio.run(server.transcribe_meeting_recording(missing_id))
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(server.transcribe_meeting_recording("bad-file-id"))

        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_transcribe_recording_stream_by_file_id(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)

        async def fake_stream(content, suffix, threshold, allowed_speaker_ids, *, source_path=None):
            async def events():
                yield b"data: {\"type\":\"done\"}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        with patch("app.server._stream_meeting_transcription", side_effect=fake_stream) as stream:
            response = asyncio.run(
                server.transcribe_meeting_recording_stream(
                    file_id,
                    threshold=0.42,
                    allowed_speaker_ids=["spk-a"],
                )
            )

        self.assertEqual(response.media_type, "text/event-stream")
        args = stream.call_args.args
        kwargs = stream.call_args.kwargs
        self.assertIsNone(args[0])
        self.assertEqual(args[1], ".wav")
        self.assertEqual(args[2], 0.42)
        self.assertEqual(args[3], ["spk-a"])
        self.assertEqual(kwargs["source_path"], os.path.join(self.tmpdir.name, f"{file_id}.wav"))

    def test_transcribe_recording_stream_missing_and_invalid_file_id(self):
        missing_id = "00000000-0000-0000-0000-000000000000"

        with self.assertRaises(HTTPException) as missing:
            asyncio.run(server.transcribe_meeting_recording_stream(missing_id))
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(server.transcribe_meeting_recording_stream("bad-file-id"))

        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_delete_recordings(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)
        missing_id = "00000000-0000-0000-0000-000000000000"

        payload = asyncio.run(
            server.delete_meeting_recordings(
                server.DeleteRecordingsRequest(
                    fileIds=[file_id, missing_id],
                    reason="patient_discharged",
                    requestId="req-1",
                )
            )
        )

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["deleted"], [file_id])
        self.assertEqual(payload["missing"], [missing_id])
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir.name, f"{file_id}.wav")))

    def test_delete_recordings_deduplicates_file_ids(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)

        payload = asyncio.run(
            server.delete_meeting_recordings(
                server.DeleteRecordingsRequest(fileIds=[file_id, file_id])
            )
        )

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["deleted"], [file_id])
        self.assertEqual(payload["missing"], [])

    def test_delete_recordings_returns_partial_when_delete_fails(self):
        file_id = server.recording_store.save_pcm_wav(b"\x00\x00" * 10)

        with patch("app.services.recording_store.os.remove", side_effect=OSError("busy")):
            payload = asyncio.run(
                server.delete_meeting_recordings(
                    server.DeleteRecordingsRequest(fileIds=[file_id])
                )
            )

        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["deleted"], [])
        self.assertEqual(payload["failed"], [file_id])

    def test_delete_empty_and_invalid_recordings(self):
        with self.assertRaises(HTTPException) as empty:
            asyncio.run(
                server.delete_meeting_recordings(
                    server.DeleteRecordingsRequest(fileIds=[])
                )
            )
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(
                server.delete_meeting_recordings(
                    server.DeleteRecordingsRequest(fileIds=["../../bad"])
                )
            )

        self.assertEqual(empty.exception.status_code, 400)
        self.assertEqual(invalid.exception.status_code, 400)

    def test_live_websocket_stop_returns_recording_file_id(self):
        websocket = FakeWebSocket([
            {"type": "websocket.receive", "bytes": b"\x00\x00" * 1600},
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'},
        ])

        asyncio.run(server.websocket_live(websocket))
        payload = websocket.sent_json[-1]

        self.assertEqual(payload["type"], "recording_saved")
        self.assertEqual(payload["format"], "wav")
        self.assertEqual(payload["sampleRate"], SAMPLE_RATE)
        self.assertEqual(payload["channels"], CHANNELS)
        self.assertTrue(
            os.path.exists(os.path.join(self.tmpdir.name, f"{payload['fileId']}.wav"))
        )

    def test_live_websocket_pause_resume_skips_paused_audio(self):
        websocket = FakeWebSocket([
            {"type": "websocket.receive", "bytes": b"\x01\x00" * 10},
            {"type": "websocket.receive", "text": '{"type":"pause_recording"}'},
            {"type": "websocket.receive", "bytes": b"\x02\x00" * 10},
            {"type": "websocket.receive", "text": '{"type":"resume_recording"}'},
            {"type": "websocket.receive", "bytes": b"\x03\x00" * 10},
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'},
        ])

        asyncio.run(server.websocket_live(websocket))

        message_types = [payload["type"] for payload in websocket.sent_json]
        self.assertIn("recording_paused", message_types)
        self.assertIn("recording_resumed", message_types)
        recording_saved = websocket.sent_json[-1]
        self.assertEqual(recording_saved["type"], "recording_saved")

        wav_path = os.path.join(self.tmpdir.name, f"{recording_saved['fileId']}.wav")
        with wave.open(wav_path, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            samples = [
                int.from_bytes(frames[i:i + 2], "little", signed=True)
                for i in range(0, len(frames), 2)
            ]

        self.assertEqual(len(samples), 20)
        self.assertEqual(samples[:10], [1] * 10)
        self.assertEqual(samples[10:], [3] * 10)

    def test_live_websocket_pause_then_stop_saves_audio_before_pause(self):
        websocket = FakeWebSocket([
            {"type": "websocket.receive", "bytes": b"\x04\x00" * 10},
            {"type": "websocket.receive", "text": '{"type":"pause_recording"}'},
            {"type": "websocket.receive", "bytes": b"\x05\x00" * 10},
            {"type": "websocket.receive", "text": '{"type":"stop_recording"}'},
        ])

        asyncio.run(server.websocket_live(websocket))

        recording_saved = websocket.sent_json[-1]
        self.assertEqual(recording_saved["type"], "recording_saved")
        wav_path = os.path.join(self.tmpdir.name, f"{recording_saved['fileId']}.wav")
        with wave.open(wav_path, "rb") as wav_file:
            self.assertEqual(wav_file.getnframes(), 10)

    def test_live_websocket_stop_send_failure_does_not_raise(self):
        websocket = FakeWebSocket(
            [
                {"type": "websocket.receive", "bytes": b"\x00\x00" * 1600},
                {"type": "websocket.receive", "text": '{"type":"stop_recording"}'},
            ],
            fail_send=True,
        )

        asyncio.run(server.websocket_live(websocket))

        wav_files = [name for name in os.listdir(self.tmpdir.name) if name.endswith(".wav")]
        self.assertEqual(len(wav_files), 1)

    def test_live_websocket_disconnect_discards_temp_recording(self):
        websocket = FakeWebSocket([
            {"type": "websocket.receive", "bytes": b"\x00\x00" * 1600},
            {"type": "websocket.disconnect"},
        ])

        asyncio.run(server.websocket_live(websocket))

        self.assertEqual(os.listdir(self.tmpdir.name), [])


class FakeQueryParams:
    def getlist(self, name):
        return []


class FakeWebSocket:
    def __init__(self, messages, fail_send=False):
        self.messages = list(messages)
        self.query_params = FakeQueryParams()
        self.sent_json = []
        self.accepted = False
        self.fail_send = fail_send

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if not self.messages:
            return {"type": "websocket.disconnect"}
        return self.messages.pop(0)

    async def send_json(self, payload):
        if self.fail_send:
            raise RuntimeError("client disconnected")
        self.sent_json.append(payload)

    async def close(self, code=1000):
        pass


class FakeService:
    registered_embeddings = {}

    def build_matching_scope(self, allowed_speaker_ids):
        return None


if __name__ == "__main__":
    unittest.main()
