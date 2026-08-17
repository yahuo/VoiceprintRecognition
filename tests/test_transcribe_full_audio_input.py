import os
import tempfile
import unittest
from unittest.mock import patch

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
    def test_paraformer_path_input_is_loaded_to_numpy_array(self):
        service = ModelService()
        dummy_model = DummyParaformerModel()
        service.upload_asr_backend = "paraformer"
        service.upload_asr_model = dummy_model

        audio = np.zeros(1600, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            audio_path = tmp.name

        try:
            result = service.transcribe_full_audio(
                audio_path,
                backend="paraformer",
                return_timestamps=True,
            )
        finally:
            os.unlink(audio_path)

        self.assertEqual(result["text"], "ok")
        self.assertIsInstance(dummy_model.last_input, np.ndarray)
        self.assertEqual(dummy_model.last_input.dtype, np.float32)

    def test_nano_backend_resets_request_scoped_generation_kwargs(self):
        class DummyNanoModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [{"text": "ok"}]

        service = ModelService()
        dummy_model = DummyNanoModel()
        service.upload_asr_backend = "nano"
        service.upload_asr_model = dummy_model
        service.asr_model = dummy_model

        audio = np.zeros(1600, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            audio_path = tmp.name

        try:
            with patch.object(
                service,
                "_generate_nano",
                wraps=service._generate_nano,
            ) as generate_nano:
                result = service.transcribe_full_audio(
                    audio_path,
                    backend="nano",
                )
        finally:
            os.unlink(audio_path)

        generate_nano.assert_called_once()
        self.assertEqual(result["text"], "ok")
        self.assertEqual(dummy_model.kwargs["input"], audio_path)
        self.assertEqual(dummy_model.kwargs["hotwords"], [])
        self.assertEqual(dummy_model.kwargs["max_length"], 200)
        self.assertTrue(dummy_model.kwargs["itn"])
        self.assertNotIn("use_itn", dummy_model.kwargs)


if __name__ == "__main__":
    unittest.main()
