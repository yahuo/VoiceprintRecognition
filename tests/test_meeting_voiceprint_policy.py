import unittest

import numpy as np

from app.services.meeting import _process_with_diarization, _process_with_vad


class NoVoiceprintService:
    def __init__(self):
        self.extract_embedding_calls = 0

    def transcribe_segment(self, _audio):
        return "测试文本"

    def extract_embedding(self, _audio):
        self.extract_embedding_calls += 1
        raise AssertionError("extract_embedding should not be called")

    def match_registered_speaker_guarded(self, *args, **kwargs):
        raise AssertionError("registered speaker matching should not be called")

    def match_registered_speaker_consensus(self, *args, **kwargs):
        raise AssertionError("registered speaker consensus should not be called")

    def vad_segment(self, _audio_path):
        return [[0, 1000]]


class MeetingVoiceprintPolicyTest(unittest.TestCase):
    def test_upload_diarization_without_selected_speakers_skips_voiceprint(self):
        service = NoVoiceprintService()
        speech = np.ones(16000, dtype=np.float32)

        transcript = _process_with_diarization(
            service,
            "unused.wav",
            speech,
            16000,
            [(0, 1000, "SPEAKER_00")],
            0.3,
            None,
            None,
            False,
        )

        self.assertEqual(service.extract_embedding_calls, 0)
        self.assertIsNone(transcript[0]["speakerId"])
        self.assertEqual(transcript[0]["speaker"], "陌生人1")

    def test_upload_vad_without_selected_speakers_skips_voiceprint(self):
        service = NoVoiceprintService()
        speech = np.ones(16000, dtype=np.float32)

        transcript = _process_with_vad(
            service,
            "unused.wav",
            speech,
            16000,
            0.3,
            None,
            None,
            False,
        )

        self.assertEqual(service.extract_embedding_calls, 0)
        self.assertIsNone(transcript[0]["speakerId"])
        self.assertEqual(transcript[0]["speaker"], "未知")


if __name__ == "__main__":
    unittest.main()
