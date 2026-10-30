import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import server
from app.services.kws import create_kws_service
from app.services.kws_keywords import KeywordsError, KeywordStore, compile_categories, parse_raw_lines

# 不含汉字的说法按已分好的 token 处理，校验逻辑不依赖 pypinyin。
VOCABULARY = {"a", "b", "c", "d"}


def category(code, *phrases, slot=False, **extra):
    return {"code": code, "name": code, "slot": slot,
            "keywords": [{"phrase": p, "value": p.replace(" ", "_"), **extra} for p in phrases]}


class FakeSpotter:
    def __init__(self, **kwargs):
        with open(kwargs["keywords_file"], encoding="utf-8") as f:
            self.lines = f.read().splitlines()
        self.keywords_file = kwargs["keywords_file"]

    def create_stream(self):
        return object()


def fake_modules():
    return {"sherpa_onnx": types.SimpleNamespace(KeywordSpotter=FakeSpotter), "pypinyin": types.ModuleType("pypinyin")}


class CompileCategoriesTest(unittest.TestCase):
    def test_compiles_lines_with_options_and_labels(self):
        categories, lines = compile_categories([
            category("bed", "a b", slot=True),
            {"code": "command", "name": "指令", "keywords": [{"phrase": "c  d", "value": "next", "boost": "1.5", "threshold": 0.3}]},
        ], VOCABULARY)
        self.assertEqual(lines, ["a b @bed:a_b", "c d :1.5 #0.3 @command:next"])
        self.assertEqual(categories[1]["keywords"][0], {"phrase": "c d", "value": "next", "boost": 1.5, "threshold": 0.3, "tokens": "c d"})
        self.assertEqual([c["slot"] for c in categories], [True, False])

    def test_reports_every_invalid_keyword(self):
        with self.assertRaises(KeywordsError) as caught:
            compile_categories([
                category("bed", "a b", "a b", "x", slot=True),
                category("cmd", "a b"),
                {"code": "cmd2", "keywords": [{"phrase": "c", "value": "two words"}, {"phrase": "d", "boost": 20},
                                              {"phrase": "b a#0.1"}, {"phrase": "a  b"}]},
                category("Bad-Code", "c"),
                category("bed", "d"),
            ], VOCABULARY)
        messages = [e["message"] for e in caught.exception.errors]
        self.assertEqual(len(messages), 9, messages)
        self.assertIn("「a b」在 bed 和 bed 里重复", messages)
        self.assertTrue(any("x 不在模型词表中" in m for m in messages))
        self.assertTrue(any("不含空白" in m for m in messages))
        self.assertTrue(any("增益" in m for m in messages))
        self.assertTrue(any(": # @ /" in m for m in messages))
        self.assertTrue(any("Bad-Code" in m for m in messages))
        self.assertIn("类别代码 bed 重复", messages)

    def test_same_pronunciation_and_empty_table_are_rejected(self):
        with self.assertRaisesRegex(KeywordsError, "至少保留一个关键词"):
            compile_categories([category("bed", slot=True)], VOCABULARY)
        with self.assertRaisesRegex(KeywordsError, "categories 须为列表"):
            compile_categories({"bed": []}, VOCABULARY)

    @unittest.skipUnless(importlib.util.find_spec("pypinyin"), "需要 pypinyin")
    def test_chinese_phrases_with_same_pronunciation_are_rejected(self):
        vocabulary = {"sh", "í", "èr", "ch", "uáng"}
        with self.assertRaisesRegex(KeywordsError, "发音相同"):
            compile_categories([category("bed", "十二床", "拾贰床")], vocabulary)

    def test_raw_lines_seed_slot_categories(self):
        categories = parse_raw_lines(["# 注释", "a b @bed:12床", "c :1.5 #0.2 @command:next_bed"], ("bed", "ward"))
        self.assertEqual([(c["code"], c["slot"], len(c["keywords"])) for c in categories],
                         [("bed", True, 1), ("ward", True, 0), ("command", False, 1)])
        self.assertEqual(categories[2]["keywords"][0], {"phrase": "c", "value": "next_bed", "boost": "1.5", "threshold": "0.2"})
        with self.assertRaisesRegex(KeywordsError, "第 1 行"):
            parse_raw_lines(["a b"], ())


class KeywordStoreTest(unittest.TestCase):
    def test_round_trip_is_atomic_and_missing_store_is_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KeywordStore(os.path.join(tmpdir, "kws", "keywords.json"))
            self.assertIsNone(store.load())
            store.save({"version": 1, "categories": [category("bed", "a")]})
            self.assertEqual(store.load()["categories"][0]["keywords"][0]["phrase"], "a")
            self.assertEqual(os.listdir(os.path.dirname(store.path)), ["keywords.json"])


class KeywordServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        tokens = os.path.join(self.tmpdir.name, "tokens.txt")
        with open(tokens, "w", encoding="utf-8") as f:
            f.write("".join(f"{token} {i}\n" for i, token in enumerate(sorted(VOCABULARY))))
        self.seed = os.path.join(self.tmpdir.name, "keywords.txt")
        with open(self.seed, "w", encoding="utf-8") as f:
            f.write("a b @bed:12床\nc #0.3 @command:next_bed\n")
        self.config = {
            "kws_enabled": True, "kws_tokens_path": tokens, "kws_encoder_path": "e", "kws_decoder_path": "d",
            "kws_joiner_path": "j", "kws_keywords_file": self.seed, "kws_provider": "cpu", "kws_num_threads": 1,
            "kws_keywords_store": os.path.join(self.tmpdir.name, "data", "keywords.json"),
            "kws_context_slots": "bed", "kws_duplicate_cooldown_ms": 2000,
        }
        self.modules = patch.dict(sys.modules, fake_modules())
        self.modules.start()
        self.service = create_kws_service(self.config)

    def tearDown(self):
        self.modules.stop()
        self.tmpdir.cleanup()

    def put(self, version, categories):
        with patch.object(server, "kws_service", self.service):
            return asyncio.run(server.update_kws_keywords(server.KwsKeywordsUpdate(version=version, categories=categories)))

    def test_missing_store_config_fails_fast(self):
        config = dict(self.config, kws_keywords_store="")
        with self.assertRaisesRegex(RuntimeError, "KWS_KEYWORDS_STORE"):
            create_kws_service(config)
        config = dict(self.config, kws_keywords_file="")
        with self.assertRaisesRegex(RuntimeError, "KWS_KEYWORDS_FILE"):
            create_kws_service(config)

    def test_seeds_from_keywords_file_without_writing_store(self):
        self.assertEqual((self.service.version, self.service.context_slots), (0, ("bed",)))
        self.assertEqual(self.service._spotter.lines, ["a b @bed:12床", "c #0.3 @command:next_bed"])
        self.assertFalse(os.path.exists(self.service._spotter.keywords_file))
        self.assertFalse(os.path.exists(self.config["kws_keywords_store"]))
        with patch.object(server, "kws_service", self.service):
            payload = asyncio.run(server.get_kws_keywords())
        self.assertEqual(payload["categories"][0]["keywords"][0]["tokens"], "a b")
        self.assertEqual(payload["active_sessions"], 0)

    def test_save_swaps_spotter_for_new_sessions_and_persists(self):
        old_session = self.service.create_session()
        payload = self.put(0, [category("bed", "a b", "d", slot=True), category("ward", "c", slot=True)])

        self.assertEqual((payload["version"], self.service.context_slots), (1, ("bed", "ward")))
        self.assertEqual(self.service.create_session()._spotter.lines, ["a b @bed:a_b", "d @bed:d", "c @ward:c"])
        self.assertEqual(old_session._spotter.lines, ["a b @bed:12床", "c #0.3 @command:next_bed"])
        with open(self.config["kws_keywords_store"], encoding="utf-8") as f:
            stored = json.load(f)
        self.assertEqual(stored["version"], 1)
        self.assertNotIn("tokens", stored["categories"][0]["keywords"][0])
        # 重启以关键词库为准，不再读 KWS_KEYWORDS_FILE。
        reloaded = create_kws_service(dict(self.config, kws_keywords_file=""))
        self.assertEqual((reloaded.version, reloaded.context_slots), (1, ("bed", "ward")))

    def test_stale_version_and_invalid_table_change_nothing(self):
        with self.assertRaises(HTTPException) as caught:
            self.put(5, [category("bed", "a", slot=True)])
        self.assertEqual(caught.exception.status_code, 409)

        response = self.put(0, [category("bed", "a", "x", slot=True)])
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["errors"][0]["phrase"], "x")
        self.assertEqual(self.service.version, 0)
        self.assertFalse(os.path.exists(self.config["kws_keywords_store"]))

    def test_preview_reports_unknown_tokens(self):
        with patch.object(server, "kws_service", self.service):
            result = asyncio.run(server.preview_kws_keywords(server.KwsKeywordsPreview(phrases=["a  b", "x"])))
        self.assertEqual([(i["tokens"], i["error"] is None) for i in result["items"]], [("a b", True), ("", False)])

    def test_disabled_kws_has_no_keywords_to_maintain(self):
        with patch.object(server, "kws_service", create_kws_service({"kws_enabled": False})):
            self.assertEqual(asyncio.run(server.get_kws_keywords()), {"enabled": False})
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(server.preview_kws_keywords(server.KwsKeywordsPreview(phrases=["a"])))
        self.assertEqual(caught.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
