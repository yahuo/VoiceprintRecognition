import unittest
from unittest.mock import patch

from app.core import CONFIG, ModelService


class AsrConfigurationTest(unittest.TestCase):
    def test_paraformer_loads_one_shared_asr_model(self):
        config = {
            "asr_backend": "paraformer",
            "asr_model_path": "/models/asr/paraformer",
            "vad_model_path": "",
            "punc_model_path": "",
        }
        with (
            patch.dict(CONFIG, config),
            patch("app.core.os.path.exists", return_value=True),
            patch("app.core.AutoModel") as auto_model,
        ):
            loaded_model = object()
            auto_model.return_value = loaded_model

            service = ModelService()
            service._load_asr_model(device="cpu")

        auto_model.assert_called_once()
        self.assertEqual(auto_model.call_args.kwargs["model_path"], config["asr_model_path"])
        self.assertIs(service.asr_model, loaded_model)
        self.assertEqual(service.asr_backend, "paraformer")
        self.assertNotIn("upload_asr_backend", CONFIG)
        self.assertNotIn("upload_asr_model_path", CONFIG)
        self.assertFalse(hasattr(service, "upload_asr_model"))

    def test_nano_loads_one_shared_asr_model(self):
        config = {
            "asr_backend": "nano",
            "asr_model_path": "/models/asr/nano",
        }
        with (
            patch.dict(CONFIG, config),
            patch("app.core.os.path.exists", return_value=True),
            patch("app.core.AutoModel") as auto_model,
        ):
            loaded_model = object()
            auto_model.return_value = loaded_model

            service = ModelService()
            service._load_asr_model(device="cpu")

        auto_model.assert_called_once()
        self.assertEqual(auto_model.call_args.kwargs["model_path"], config["asr_model_path"])
        self.assertIs(service.asr_model, loaded_model)
        self.assertEqual(service.asr_backend, "nano")
        self.assertFalse(hasattr(service, "upload_asr_model"))

    def test_unknown_backend_fails_fast(self):
        with patch.dict(CONFIG, {"asr_backend": "unknown"}):
            with self.assertRaisesRegex(ValueError, "ASR_BACKEND"):
                ModelService()


if __name__ == "__main__":
    unittest.main()
