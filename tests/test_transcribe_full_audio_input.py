"""原整段 ASR 输入测试迁移为 MOSS 完整输入与输出契约测试。"""
import os
import tempfile
import unittest
from unittest.mock import patch

from app.services.moss import MossTranscriber, MossError, MossUnavailable, MossBusy, MossStreamParser, parse_result


class TranscribeFullAudioInputTest(unittest.TestCase):
    def result(self, raw, **kwargs):
        return {"raw": raw, "finish_reason": "stop", "eos_reached": True, **kwargs}

    def test_preserves_text_numbers_order_and_overlaps(self):
        raw = "[0.00][S01]血压118、26。[1.23]\n[1.00][S02]没有。[2.30]"
        rows = parse_result(self.result(raw), 3)
        self.assertEqual([r["text"] for r in rows], ["血压118、26。", "没有。"])
        self.assertEqual((rows[0]["end_ms"], rows[1]["start_ms"]), (1230, 1000))

    def test_bracketed_text_is_not_silently_removed(self):
        rows = parse_result(self.result("[0][S01]文本[26]还有[noise]符号。[1]"), 2)
        self.assertEqual(rows[0]["text"], "文本[26]还有[noise]符号。")

    def test_invalid_or_partial_outputs_fail_closed(self):
        for raw in ["", "free text", "junk[0][S01]内容[1]", "[0][S01]内容[1]garbage",
                    "[0][S01]内容[1][1][S02]未完", "[2][S01]逆序[1]", "[0][S01]超时[9]",
                    "[0][S01]空时间[0]", "[0][S01] [1]", "[1][S01]逆序[2][0][S02]文字[1]"]:
            with self.subTest(raw=raw), self.assertRaises(MossError):
                parse_result(self.result(raw), 3)
        for changes in [{"finish_reason": "length"}, {"eos_reached": False}]:
            with self.assertRaises(MossError):
                parse_result(self.result("[0][S01]已生成但截断[1]", **changes), 2)

    def test_incremental_parse_preserves_brackets_overlap_and_final_tail(self):
        raw = ' \n[0][S01]血压[118]、[26]，无发热。[1.23]\n[1][S02]没有。[2.3]'
        expected = parse_result(self.result(raw), 3)
        for size in (1, 3, 17, len(raw)):
            with self.subTest(size=size):
                parser = MossStreamParser(3)
                seen = []
                for offset in range(0, len(raw), size):
                    seen.extend(parser.feed(raw[offset:offset + size]))
                self.assertEqual(seen, expected[:1])
                self.assertEqual(parser.finish(self.result(raw)), expected)

    def test_partial_numeric_brackets_are_not_published_as_end_times(self):
        parser = MossStreamParser(5)
        self.assertEqual(parser.feed('[0][S01]数值[1]'), [])
        self.assertEqual(parser.feed('还有原文[2]\n[2][S'), [])
        self.assertEqual(parser.feed('02]')[0]['text'], '数值[1]还有原文')

    def test_incremental_time_checks_and_terminal_checks_still_apply(self):
        for raw in ('[0][S01]超时[9][1][S02]', '[2][S01]逆序[1][2][S02]',
                    '[0][S01] [1][1][S02]'):
            with self.subTest(raw=raw), self.assertRaises(MossError):
                MossStreamParser(3).feed(raw)
        parser = MossStreamParser(3)
        parser.feed('[1][S01]一[2][0][S02]二[1]')
        with self.assertRaises(MossError):
            parser.finish(self.result(parser.raw))
        parser = MossStreamParser(3)
        parser.feed('[0][S01]一[1][1][S02]二[2]')
        for result in [self.result(parser.raw, finish_reason='length'),
                       self.result(parser.raw + '未解析内容'),
                       self.result('[0][S01]被修改[1][1][S02]二[2]')]:
            with self.assertRaises(MossError):
                parser.finish(result)

    def test_missing_runtime_fails_without_loading_local_asr_or_downloading(self):
        client = MossTranscriber()
        try:
            with patch.dict(os.environ, {"MOSS_PYTHON": "", "MOSS_MODEL_PATH": ""}), self.assertRaises(MossUnavailable):
                client.transcribe("unused.wav", 1)
            self.assertIsNone(client._process)
        finally:
            client.close()

    def test_busy_request_cannot_stop_another_inference(self):
        client = MossTranscriber()
        client._lock.acquire()
        try:
            with patch.object(client, "_stop") as stop, self.assertRaises(MossBusy):
                client.transcribe("unused.wav", 1)
            stop.assert_not_called()
        finally:
            client._lock.release()
            client.close()


if __name__ == "__main__":
    unittest.main()
