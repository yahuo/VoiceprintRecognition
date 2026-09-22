"""配置回归：实时模型分工固定，离线 MOSS 独立懒加载，不恢复旧后端切换。"""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dotenv import dotenv_values

from app.core import CONFIG, ModelService
from app.services.moss import MossTranscriber
from scripts.download_all_models import MODEL_SPECS, MOSS_SPEC


class AsrConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.service = ModelService()
        self.addCleanup(self.service.close)

    def test_realtime_loads_configured_models_without_starting_moss(self):
        config = {
            "asr_model_path": "/models/nano",
            "streaming_asr_model_path": "/models/streaming-paraformer",
            "vad_model_path": "/models/vad",
            "spk_model_path": "/models/campplus",
        }
        nano, vad, streaming, speaker = [object() for _ in range(4)]
        with (
            patch.dict(CONFIG, config),
            patch("app.core.os.path.exists", return_value=True),
            patch("app.core.AutoModel", side_effect=[nano, vad, streaming, speaker]) as auto_model,
            patch.object(self.service, "reload_voiceprints"),
            patch.object(self.service.moss, "_start") as start_moss,
        ):
            self.service.load_models(device="cpu")

        self.assertEqual(auto_model.call_count, 4)
        calls = [call.kwargs for call in auto_model.call_args_list]
        self.assertEqual(calls[0]["model"], "FunAudioLLM/Fun-ASR-Nano-2512")
        self.assertEqual(calls[0]["model_path"], config["asr_model_path"])
        self.assertEqual(calls[1]["model"], config["vad_model_path"])
        self.assertEqual(calls[2]["model"], config["streaming_asr_model_path"])
        self.assertEqual(calls[3]["model"], "iic/speech_campplus_sv_zh-cn_16k-common")
        self.assertEqual(calls[3]["model_path"], config["spk_model_path"])
        self.assertIs(self.service.asr_model, nano)
        self.assertIs(self.service.streaming_vad_model, vad)
        self.assertIs(self.service.streaming_asr_model, streaming)
        self.assertIs(self.service.spk_model, speaker)
        self.assertIsInstance(self.service.moss, MossTranscriber)
        start_moss.assert_not_called()
        self.assertIsNone(self.service.moss._process)

    def test_offline_only_loads_speaker_without_realtime_or_moss_startup(self):
        with (
            patch.dict(CONFIG, {"spk_model_path": ""}),
            patch("app.core.AutoModel") as auto_model,
            patch.object(self.service, "reload_voiceprints"),
            patch.object(self.service.moss, "_start") as start_moss,
        ):
            self.service.load_models(device="cpu", load_live=False)

        auto_model.assert_called_once_with(
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            device="cpu", disable_update=True,
        )
        self.assertIsNone(self.service.asr_model)
        self.assertIsNone(self.service.streaming_vad_model)
        self.assertIsNone(self.service.streaming_asr_model)
        start_moss.assert_not_called()

    def test_legacy_backend_switches_do_not_replace_realtime_pipeline(self):
        for backend in ("paraformer", "nano", "unknown"):
            with (
                self.subTest(backend=backend),
                patch.dict(os.environ, {key: backend for key in
                           ("ASR_BACKEND", "LIVE_ASR_BACKEND", "UPLOAD_ASR_BACKEND")}),
                patch.dict(CONFIG, {"asr_model_path": "", "spk_model_path": ""}),
                patch("app.core.AutoModel") as auto_model,
                patch.object(self.service, "_load_streaming_models") as streaming,
                patch.object(self.service, "reload_voiceprints"),
            ):
                self.service.load_models(device="cpu")
                self.assertEqual(auto_model.call_args_list[0].kwargs["model"],
                                 "FunAudioLLM/Fun-ASR-Nano-2512")
                streaming.assert_called_once_with("cpu")
        self.assertNotIn("asr_backend", CONFIG)
        self.assertNotIn("upload_asr_backend", CONFIG)
        self.assertFalse(hasattr(self.service, "upload_asr_model"))

    def test_env_example_matches_current_model_download_layout(self):
        example = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example", interpolate=False)
        for spec in [*MODEL_SPECS, MOSS_SPEC]:
            with self.subTest(model=spec.name):
                self.assertEqual(example[f"{spec.name}_MODEL_PATH"], f"/app/models/{spec.subdir}")
        for key in ("ASR_BACKEND", "LIVE_ASR_BACKEND", "UPLOAD_ASR_BACKEND",
                    "UPLOAD_ASR_MODEL_PATH", "PUNC_MODEL_PATH"):
            self.assertNotIn(key, example)
        self.assertEqual(example["MOSS_PYTHON"], "")
        self.assertEqual(example["LLM_API_KEY"], "")


if __name__ == "__main__":
    unittest.main()
