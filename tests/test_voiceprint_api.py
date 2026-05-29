import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

import app.core as core
import app.server as server
import app.utils.voiceprint as voiceprint


class FakeUpload:
    filename = "sample.wav"

    def __init__(self):
        self.file = io.BytesIO(b"fake wav")


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
