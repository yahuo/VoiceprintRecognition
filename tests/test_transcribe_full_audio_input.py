import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from app.core import ModelService


class DummyParaformerModel:
    def __init__(self):
        self.last_input = None

    def generate(self, **kwargs):
        self.last_input = kwargs.get("input")
        return [{"text": "ok", "sentence_info": []}]


class TranscribeFullAudioInputTest(unittest.TestCase):
    def test_live_and_full_transcription_share_paraformer_model(self):
        service = ModelService()
        dummy_model = DummyParaformerModel()
        service.asr_backend = "paraformer"
        service.asr_model = dummy_model

        audio = np.zeros(1600, dtype=np.float32)

        self.assertEqual(service.transcribe_live_segment(audio), "ok")
        self.assertIsInstance(dummy_model.last_input, np.ndarray)
        self.assertIs(service.asr_model, dummy_model)

    def test_paraformer_path_input_is_loaded_to_numpy_array(self):
        service = ModelService()
        dummy_model = DummyParaformerModel()
        service.asr_backend = "paraformer"
        service.asr_model = dummy_model

        audio = np.zeros(1600, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            audio_path = tmp.name

        try:
            result = service.transcribe_full_audio(
                audio_path,
                return_timestamps=True,
            )
        finally:
            os.unlink(audio_path)

        self.assertEqual(result["text"], "ok")
        self.assertIsInstance(dummy_model.last_input, np.ndarray)
        self.assertEqual(dummy_model.last_input.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
