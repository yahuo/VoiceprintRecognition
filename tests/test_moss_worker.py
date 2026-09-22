import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app.services.moss import MossCancelled, MossError, MossTimeout, MossTranscriber


class MossWorkerLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audio = Path(self.temp.name) / 'audio.wav'
        self.audio.write_bytes(b'synthetic')
        self.processes = []
        popen = subprocess.Popen

        def start(_command, **kwargs):
            process = popen([sys.executable, '-u', str(Path(__file__).parent / 'fixtures/moss_worker.py')], **kwargs)
            self.processes.append(process)
            return process

        self.patches = [
            patch.dict(os.environ, {'MOSS_PYTHON': sys.executable, 'MOSS_MODEL_PATH': self.temp.name,
                                   'MOSS_START_TIMEOUT_SECONDS': '5', 'MOSS_TIMEOUT_SECONDS': '0.5',
                                   'MOSS_CANCEL_TIMEOUT_SECONDS': '.3'}),
            patch('app.services.moss.subprocess.Popen', side_effect=start),
        ]
        for p in self.patches:
            p.start()
        self.client = MossTranscriber()

    def tearDown(self):
        self.client.close()
        for process in self.processes:
            self.assertIsNotNone(process.poll(), 'owned worker leaked')
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_worker_is_reused_and_source_not_deleted(self):
        for _ in range(2):
            rows = self.client.transcribe(str(self.audio), 2)
            self.assertEqual(rows[0]['text'], 'synthetic')
        self.assertEqual(len(self.processes), 1)
        self.assertTrue(self.audio.exists())

    def test_timeout_reaps_worker_and_allows_clean_retry(self):
        with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'timeout'}), self.assertRaises(MossTimeout):
            self.client.transcribe(str(self.audio), 2)
        self.assertIsNone(self.client._process)
        self.assertFalse(self.client._lock.locked())
        self.assertEqual(self.client.transcribe(str(self.audio), 2)[0]['text'], 'synthetic')
        self.assertEqual(len(self.processes), 2)

    def test_invalid_output_and_exit_cannot_be_success(self):
        for mode in ('exit', 'malformed', 'partial'):
            with self.subTest(mode=mode), patch.dict(os.environ, {'FAKE_MOSS_MODE': mode}), self.assertRaises(MossError):
                self.client.transcribe(str(self.audio), 2)
            self.assertIsNone(self.client._reader)
            self.assertIsNone(self.client._process)

    def test_segment_arrives_while_worker_is_still_generating_and_tail_is_not_duplicated(self):
        seen = []
        def on_segment(segment):
            seen.append(segment)
            self.assertIsNone(self.client._process.poll())
            self.audio.with_suffix('.continue').touch()
        with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'stream'}):
            rows = self.client.transcribe(str(self.audio), 2, on_segment=on_segment)
        self.assertEqual(rows, seen)
        self.assertEqual([r['text'] for r in seen], ['synthetic', 'tail'])

    def test_partial_segments_never_turn_truncation_or_changed_final_into_success(self):
        for mode in ('stream_truncated', 'stream_changed'):
            seen = []
            def on_segment(segment):
                seen.append(segment)
                self.audio.with_suffix('.continue').touch()
            with self.subTest(mode=mode), patch.dict(os.environ, {'FAKE_MOSS_MODE': mode}), self.assertRaises(MossError):
                self.client.transcribe(str(self.audio), 2, on_segment=on_segment)
            self.assertEqual([r['text'] for r in seen], ['synthetic'])
            self.assertIsNone(self.client._process)

    def test_deltas_do_not_reset_total_timeout(self):
        started = time.monotonic()
        with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'stream_timeout', 'MOSS_TIMEOUT_SECONDS': '.15'}), self.assertRaises(MossTimeout):
            self.client.transcribe(str(self.audio), 2)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIsNone(self.client._process)

    def test_cancellation_after_a_visible_segment_keeps_worker_for_next_request(self):
        cancelled = threading.Event()
        seen = []
        def on_segment(segment):
            seen.append(segment)
            cancelled.set()
        with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'stream'}), self.assertRaises(MossError):
            self.client.transcribe(str(self.audio), 2, cancelled=cancelled, on_segment=on_segment)
        self.assertEqual(len(seen), 1)
        process = self.client._process
        self.assertIsNone(process.poll())
        self.assertEqual(self.client.transcribe(str(self.audio), 2)[0]['text'], 'synthetic')
        self.assertIs(self.client._process, process)
        self.assertEqual(len(self.processes), 1)

    def test_cancellation_keeps_worker_and_source(self):
        cancelled = threading.Event()
        timer = threading.Timer(.15, cancelled.set)
        timer.start()
        try:
            with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'timeout'}), self.assertRaises(MossError):
                self.client.transcribe(str(self.audio), 2, cancelled=cancelled)
        finally:
            timer.cancel()
        process = self.client._process
        self.assertIsNone(process.poll())
        self.assertTrue(self.audio.exists())
        self.assertEqual(self.client.transcribe(str(self.audio), 2)[0]['text'], 'synthetic')
        self.assertIs(self.client._process, process)

    def test_cancel_during_startup_keeps_newly_loaded_worker(self):
        cancelled = threading.Event()
        timer = threading.Timer(.05, cancelled.set)
        timer.start()
        try:
            with patch.dict(os.environ, {'FAKE_MOSS_START_DELAY': '.15'}), self.assertRaises(MossCancelled):
                self.client.transcribe(str(self.audio), 2, cancelled=cancelled)
        finally:
            timer.cancel()
        process = self.client._process
        self.assertIsNone(process.poll())
        self.client.transcribe(str(self.audio), 2)
        self.assertIs(self.client._process, process)

    def test_cancel_from_eos_callback_does_not_leave_an_abort_reply_for_next_request(self):
        def on_segment(_segment):
            raise MossCancelled('cancelled while publishing final segment')
        with patch.object(self.client, '_cancel_request', wraps=self.client._cancel_request) as abort:
            with self.assertRaises(MossCancelled):
                self.client.transcribe(str(self.audio), 2, on_segment=on_segment)
            abort.assert_not_called()
        process = self.client._process
        self.assertEqual(self.client.transcribe(str(self.audio), 2)[0]['text'], 'synthetic')
        self.assertIs(self.client._process, process)

    def test_cancel_ack_timeout_still_reaps_unhealthy_worker(self):
        cancelled = threading.Event()
        timer = threading.Timer(.15, cancelled.set)
        timer.start()
        try:
            with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'cancel_hang'}), self.assertRaises(MossTimeout):
                self.client.transcribe(str(self.audio), 2, cancelled=cancelled)
        finally:
            timer.cancel()
        self.assertIsNone(self.client._process)

    def test_cancellation_raised_by_segment_callback_is_acknowledged(self):
        def on_segment(_segment):
            raise MossCancelled('cancelled by job')
        with patch.dict(os.environ, {'FAKE_MOSS_MODE': 'stream'}), self.assertRaises(MossCancelled):
            self.client.transcribe(str(self.audio), 2, on_segment=on_segment)
        process = self.client._process
        self.client.transcribe(str(self.audio), 2)
        self.assertIs(self.client._process, process)


if __name__ == '__main__':
    unittest.main()
