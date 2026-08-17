import asyncio
import hashlib
import tempfile
import unittest

import numpy as np

from app import server
from app.services.recording_store import RecordingStore


class _QueryParams:
    def getlist(self, _name):
        return []


class _WebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.query_params = _QueryParams()
        self.sent_json = []

    async def accept(self):
        pass

    async def receive(self):
        if not self.messages:
            return {"type": "websocket.disconnect"}
        return self.messages.pop(0)

    async def send_json(self, payload):
        self.sent_json.append(payload)
        return True

    async def close(self, code=1000):
        pass


class _CapturingService:
    registered_embeddings = {}

    def __init__(self):
        self.transcribed_audio = []

    def build_matching_scope(self, _allowed_speaker_ids):
        return None

    def transcribe_segment(self, audio):
        self.transcribed_audio.append(bytes(audio))
        return "测试"


def _packetize(pcm: bytes, packet_samples: int):
    packet_bytes = packet_samples * 2
    return [pcm[offset:offset + packet_bytes] for offset in range(0, len(pcm), packet_bytes)]


def _fingerprint(segments):
    return [
        (len(segment), hashlib.sha256(segment).hexdigest())
        for segment in segments
    ]


def _boundary_sensitive_pcm():
    sample_rate = 16000
    silence = np.zeros(int(0.36 * sample_rate), dtype=np.int16)
    speech_a = np.full(3 * sample_rate, 1000, dtype=np.int16)
    short_bursts = np.tile(
        np.concatenate([
            np.full(int(0.01 * sample_rate), 1000, dtype=np.int16),
            np.zeros(int(0.04 * sample_rate), dtype=np.int16),
        ]),
        12,
    )
    speech_b = np.full(2 * sample_rate, 1000, dtype=np.int16)
    trailing_silence = np.zeros(int(0.51 * sample_rate), dtype=np.int16)
    return np.concatenate([
        silence,
        speech_a,
        short_bursts,
        speech_b,
        trailing_silence,
    ]).tobytes()


class LivePcmSegmentationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_store = server.recording_store
        self.original_service = server.service
        server.recording_store = RecordingStore(self.tmpdir.name)

    def tearDown(self):
        server.recording_store = self.original_store
        server.service = self.original_service
        self.tmpdir.cleanup()

    def _transcribe_with_packet_size(self, packet_samples):
        service = _CapturingService()
        server.service = service
        messages = [
            {"type": "websocket.receive", "bytes": chunk}
            for chunk in _packetize(_boundary_sensitive_pcm(), packet_samples)
        ]
        messages.append({"type": "websocket.receive", "text": '{"type":"stop_recording"}'})

        asyncio.run(server.websocket_live(_WebSocket(messages)))
        return service.transcribed_audio

    def test_segmentation_is_independent_of_websocket_packet_size(self):
        expected = self._transcribe_with_packet_size(43)

        self.assertEqual(len(expected), 1)
        for packet_samples in (128, 683, 2048):
            self.assertEqual(
                _fingerprint(self._transcribe_with_packet_size(packet_samples)),
                _fingerprint(expected),
                f"packet size {packet_samples} changed VAD segmentation",
            )


if __name__ == "__main__":
    unittest.main()
