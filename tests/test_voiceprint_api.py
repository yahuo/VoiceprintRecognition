import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app.core as core
import app.server as server
import app.utils.voiceprint as voiceprint
from scripts import verification
from tests.test_meeting_http import request


class FakeUpload:
    filename = "sample.wav"

    def __init__(self, content=b"fake wav"):
        self.file = io.BytesIO(content)

    async def read(self, size):
        return self.file.read(size)


class FakeVoiceprintService:
    def __init__(self):
        self.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.reload_voiceprints()

    def extract_embedding(self, _audio_path):
        return self.embedding

    def reload_voiceprints(self):
        self.registered_speakers = core.load_voiceprint_index()
        self.registered_embeddings = core.load_voiceprint_embeddings()


class FakeVoiceprintModel:
    def generate(self, input):
        return [{"spk_embedding": [1.0, 0.0, 0.0]}]


class VoiceprintApiTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_paths = (
            core.VOICEPRINT_DB_DIR,
            core.VOICEPRINT_INDEX_FILE,
            core.VOICEPRINT_FILES_DIR,
        )
        self.original_service = server.service
        core.VOICEPRINT_DB_DIR = self.tmpdir.name
        core.VOICEPRINT_INDEX_FILE = os.path.join(self.tmpdir.name, "index.json")
        core.VOICEPRINT_FILES_DIR = os.path.join(self.tmpdir.name, "files")
        server.service = FakeVoiceprintService()

    def tearDown(self):
        server.service = self.original_service
        (
            core.VOICEPRINT_DB_DIR,
            core.VOICEPRINT_INDEX_FILE,
            core.VOICEPRINT_FILES_DIR,
        ) = self.original_paths
        self.tmpdir.cleanup()

    def test_register_without_id_generates_object_id_style_id(self):
        payload = asyncio.run(server.register_speaker("张三", FakeUpload(), voiceprint_id=None))

        speaker_id = payload["id"]
        self.assertRegex(speaker_id, r"^[0-9a-f]{24}$")
        self.assertEqual(payload["name"], "张三")
        self.assertTrue(os.path.exists(core.load_voiceprint_index()[speaker_id]["file"]))

        exists = asyncio.run(server.voiceprint_exists(speaker_id))
        self.assertTrue(exists["registered"])
        self.assertEqual(exists["name"], "张三")

    def test_custom_id_is_opaque_and_overwrites_same_id(self):
        custom_id = " external/user/中文 id "
        first = asyncio.run(server.register_speaker("张三", FakeUpload(), voiceprint_id=custom_id))
        first_file = core.load_voiceprint_index()[custom_id]["file"]

        server.service.embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        second = asyncio.run(server.register_speaker("李四", FakeUpload(), voiceprint_id=custom_id))
        index = core.load_voiceprint_index()

        self.assertEqual(first["id"], custom_id)
        self.assertEqual(second["id"], custom_id)
        self.assertEqual(len(index), 1)
        self.assertEqual(index[custom_id]["name"], "李四")
        self.assertEqual(index[custom_id]["file"], first_file)
        np.testing.assert_array_equal(np.load(first_file), server.service.embedding)

        exists = asyncio.run(server.voiceprint_exists(custom_id))
        trimmed_exists = asyncio.run(server.voiceprint_exists(custom_id.strip()))
        self.assertTrue(exists["registered"])
        self.assertFalse(trimmed_exists["registered"])

    def test_same_name_can_register_multiple_ids(self):
        asyncio.run(server.register_speaker("张三", FakeUpload(), voiceprint_id="a"))
        asyncio.run(server.register_speaker("张三", FakeUpload(), voiceprint_id="b"))

        payload = asyncio.run(server.list_speakers())

        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            payload["speakers"],
            [{"id": "a", "name": "张三"}, {"id": "b", "name": "张三"}],
        )

    def test_delete_by_id_removes_index_and_file(self):
        asyncio.run(server.register_speaker("张三", FakeUpload(), voiceprint_id="a"))
        embedding_file = core.load_voiceprint_index()["a"]["file"]

        payload = asyncio.run(server.delete_speaker("a"))

        self.assertEqual(payload["id"], "a")
        self.assertFalse(os.path.exists(embedding_file))
        self.assertEqual(core.load_voiceprint_index(), {})

    def test_old_index_format_migrates_on_load(self):
        legacy_file = os.path.join(self.tmpdir.name, "张三.npy")
        np.save(legacy_file, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        with open(core.VOICEPRINT_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"张三": legacy_file}, f, ensure_ascii=False)

        server.service.reload_voiceprints()
        listed = asyncio.run(server.list_speakers())
        index = core.load_voiceprint_index()
        speaker_id = next(iter(index))
        entry = index[speaker_id]
        exists = asyncio.run(server.voiceprint_exists(speaker_id))

        self.assertRegex(speaker_id, r"^[0-9a-f]{24}$")
        self.assertEqual(entry["name"], "张三")
        self.assertTrue(entry["file"].startswith(core.VOICEPRINT_FILES_DIR))
        self.assertTrue(os.path.exists(entry["file"]))
        self.assertFalse(os.path.exists(legacy_file))
        self.assertIn(speaker_id, server.service.registered_embeddings)
        self.assertEqual(listed["speakers"], [{"id": speaker_id, "name": "张三"}])
        self.assertTrue(exists["registered"])

    def test_cli_old_index_format_migrates_on_load(self):
        legacy_file = os.path.join(self.tmpdir.name, "张三.npy")
        query_file = os.path.join(self.tmpdir.name, "query.wav")
        np.save(legacy_file, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        with open(query_file, "wb") as f:
            f.write(b"fake wav")
        with open(core.VOICEPRINT_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"张三": legacy_file}, f, ensure_ascii=False)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            voiceprint.list_voiceprints()
            results = voiceprint.identify_speaker(FakeVoiceprintModel(), query_file)

        speaker_id = next(iter(core.load_voiceprint_index()))
        self.assertIn("张三", output.getvalue())
        self.assertEqual(results[0][0], speaker_id)
        self.assertEqual(results[0][1], "张三")

    def test_register_upload_limits_through_http(self):
        async def exercise():
            boundary = "voiceprint-registration-test"
            prefix = (
                f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\n测试\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\ntest-id\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n'
            ).encode()
            for content, expected in ((b"", 422), (b"123456789", 413), (b"12345678", 200)):
                body = prefix + content + f'\r\n--{boundary}--\r\n'.encode()
                status, response = await request(
                    "/v1/voiceprint/register", body, f"multipart/form-data; boundary={boundary}",
                )
                self.assertEqual(status, expected, response)
                self.assertEqual(len(core.load_voiceprint_index()), int(expected == 200))

        with tempfile.TemporaryDirectory() as uploads:
            factory = tempfile.NamedTemporaryFile
            with patch.object(server, "MAX_UPLOAD_BYTES", 8), patch.object(
                server.tempfile, "NamedTemporaryFile",
                side_effect=lambda **kwargs: factory(dir=uploads, **kwargs),
            ), patch.object(server.service, "extract_embedding", wraps=server.service.extract_embedding) as extract:
                asyncio.run(exercise())
                self.assertEqual(extract.call_count, 1)
                self.assertEqual(os.listdir(uploads), [])

    def test_register_removes_partial_upload_on_read_failure_or_cancellation(self):
        class BrokenUpload:
            filename = "sample.wav"

            def __init__(self, error):
                self.reads = 0
                self.error = error

            async def read(self, _size):
                self.reads += 1
                if self.reads == 1:
                    return b"partial"
                raise self.error

        for error in (OSError("read failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as uploads:
                factory = tempfile.NamedTemporaryFile
                with patch.object(
                    server.tempfile, "NamedTemporaryFile",
                    side_effect=lambda **kwargs: factory(dir=uploads, **kwargs),
                ), patch.object(server.service, "extract_embedding") as extract:
                    with self.assertRaises(type(error)):
                        asyncio.run(server.register_speaker("测试", BrokenUpload(error), voiceprint_id="test-id"))
                    extract.assert_not_called()
                    self.assertEqual(os.listdir(uploads), [])
                    self.assertEqual(core.load_voiceprint_index(), {})

    def test_register_removes_upload_if_embedding_extraction_fails(self):
        with tempfile.TemporaryDirectory() as uploads:
            factory = tempfile.NamedTemporaryFile
            with patch.object(
                server.tempfile, "NamedTemporaryFile",
                side_effect=lambda **kwargs: factory(dir=uploads, **kwargs),
            ), patch.object(server.service, "extract_embedding", return_value=None):
                with self.assertRaises(server.HTTPException) as raised:
                    asyncio.run(server.register_speaker("测试", FakeUpload(), voiceprint_id="test-id"))
                self.assertEqual(raised.exception.status_code, 400)
                self.assertEqual(os.listdir(uploads), [])
                self.assertEqual(core.load_voiceprint_index(), {})

    def test_cli_extract_flattens_numpy_and_detaches_cpu_tensor(self):
        import torch
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            for embedding in (np.array([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]], requires_grad=True)):
                with self.subTest(kind=type(embedding).__name__):
                    model = Mock()
                    model.generate.return_value = [{"spk_embedding": embedding}]
                    result = voiceprint.extract_embedding(model, audio.name)
                    np.testing.assert_array_equal(result, [1.0, 0.0])
                    self.assertEqual(result.shape, (2,))

    def test_cli_device_tensor_is_moved_to_cpu_before_numpy(self):
        # 模拟设备张量的转换约束，不是 GPU 实机验收。
        class DeviceTensor:
            def __init__(self):
                self.steps = []

            def detach(self):
                self.steps.append("detach")
                return self

            def cpu(self):
                self.steps.append("cpu")
                return self

            def numpy(self):
                self.steps.append("numpy")
                return np.array([[1.0, 0.0]])

        embedding = DeviceTensor()
        model = Mock()
        model.generate.return_value = [{"spk_embedding": embedding}]
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio, patch("torch.Tensor", DeviceTensor):
            np.testing.assert_array_equal(voiceprint.extract_embedding(model, audio.name), [1.0, 0.0])
        self.assertEqual(embedding.steps, ["detach", "cpu", "numpy"])

    def test_cli_model_honors_local_speaker_model_path(self):
        with patch.dict(voiceprint.CONFIG, {"spk_model_path": "/models/campplus"}), patch.object(
            voiceprint, "AutoModel"
        ) as factory:
            self.assertIs(voiceprint.create_model("mps"), factory.return_value)
            factory.assert_called_once_with(
                model="iic/speech_campplus_sv_zh-cn_16k-common", model_path="/models/campplus",
                device="mps", disable_update=True,
            )

    def test_verification_reuses_cli_and_handles_batched_embeddings(self):
        self.assertIs(verification.create_model, voiceprint.create_model)
        self.assertIs(verification.extract_embedding, voiceprint.extract_embedding)
        self.assertIs(verification.cosine_similarity, core.cosine_similarity)
        model = Mock()
        model.generate.return_value = [{"spk_embedding": [[1.0, 0.0]]}]
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            same, score = verification.verify_speakers(model, audio.name, audio.name)
        self.assertTrue(same)
        self.assertAlmostEqual(score, 1.0)

    def test_invalid_embeddings_cannot_produce_nan_or_a_threshold_zero_match(self):
        valid = np.array([1.0, 0.0])
        for invalid in (np.array([]), np.zeros(2), np.array([np.nan, 0.0]), np.array([np.inf, 0.0])):
            with self.subTest(invalid=invalid):
                for left, right in ((valid, invalid), (invalid, valid)):
                    with np.errstate(all="raise"), self.assertRaisesRegex(ValueError, "声纹向量"):
                        core.cosine_similarity(left, right)
        model = Mock()
        model.generate.return_value = [{"spk_embedding": [[0.0, 0.0]]}]
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with self.assertRaisesRegex(ValueError, "声纹向量"):
                verification.verify_speakers(model, audio.name, audio.name, threshold=0)

    def test_cosine_scores_for_valid_vectors_are_unchanged(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            a, b = rng.normal(size=(2, 192)).astype(np.float32)
            expected = float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
            self.assertEqual(core.cosine_similarity(a, b), expected)

    def test_cli_identify_returns_id_for_duplicate_names(self):
        query_file = os.path.join(self.tmpdir.name, "query.wav")
        with open(query_file, "wb") as f:
            f.write(b"fake wav")
        core.save_voiceprint_embedding("a", "张三", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        core.save_voiceprint_embedding("b", "张三", np.array([0.0, 1.0, 0.0], dtype=np.float32))

        results = voiceprint.identify_speaker(FakeVoiceprintModel(), query_file)

        self.assertEqual(results[0][0], "a")
        self.assertEqual(results[0][1], "张三")
        self.assertGreater(results[0][2], results[1][2])


if __name__ == "__main__":
    unittest.main()
