import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from app.core import ModelService
from app.services.meeting import process_meeting, _clean_speaker_window, export_markdown
from app.services.moss import AudioTooLong, InvalidAudio, MossBusy


class MeetingVoiceprintPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "test.wav")
        sf.write(self.path, np.ones(16000 * 5) * 0.1, 16000)
        self.service = ModelService()
        self.service.moss.transcribe = Mock(return_value=[
            {"start_ms": 0, "end_ms": 2000, "diarizationSpeaker": "S01", "text": "第一句。"},
            {"start_ms": 2000, "end_ms": 4000, "diarizationSpeaker": "S02", "text": "第二句。"},
        ])
        self.service.extract_embedding = Mock(return_value=np.ones(2))
        self.service.match_registered_speaker_guarded = Mock(return_value=(None, 0.22))

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def test_no_selection_skips_all_voiceprint_work_and_keeps_anonymous_labels(self):
        rows = process_meeting(self.service, self.path)
        self.service.extract_embedding.assert_not_called()
        self.service.match_registered_speaker_guarded.assert_not_called()
        self.assertEqual([x["speaker"] for x in rows], ["陌生人1", "陌生人2"])
        self.assertTrue(all(x["speakerId"] is None and x["confidence"] == 0 for x in rows))

    def test_matches_current_segment_only_without_identity_inheritance(self):
        self.service.registered_speakers = {"a": {"name": "医生"}}
        self.service.match_registered_speaker_guarded.side_effect = [("a", .8), (None, .22)]
        self.service.moss.transcribe.return_value[1]["diarizationSpeaker"] = "S01"
        rows = process_meeting(self.service, self.path, .4, ["a"], match_registered_speakers=True)
        self.assertEqual(rows[0]["speakerId"], "a")
        self.assertIsNone(rows[1]["speakerId"])
        self.assertEqual(rows[1]["confidence"], 0)
        self.assertEqual(self.service.match_registered_speaker_guarded.call_args.kwargs["allowed_speaker_ids"], ["a"])
        self.assertEqual(self.service.match_registered_speaker_guarded.call_args.kwargs["threshold"], .4)

    def test_full_input_is_sent_once_with_original_text_and_times(self):
        seen = []
        def transcribe(path, duration, **kwargs):
            audio, sr = sf.read(path)
            seen.append((len(audio), sr, duration))
            return [{"start_ms": 125, "end_ms": 4800, "diarizationSpeaker": "S01", "text": "118、26，不做常识纠正。"}]
        self.service.moss.transcribe.side_effect = transcribe
        rows = process_meeting(self.service, self.path)
        self.assertEqual(seen, [(80000, 16000, 5.0)])
        self.assertEqual(rows[0]["text"], "118、26，不做常识纠正。")
        self.assertEqual((rows[0]["start_ms"], rows[0]["end_ms"]), (125, 4800))

    def test_overlap_excluded_from_identity_sampling_but_kept_in_output(self):
        rows = self.service.moss.transcribe.return_value
        rows[1]["start_ms"] = 1000
        self.assertEqual(_clean_speaker_window(rows, 0), (0, 1000))
        self.assertEqual(_clean_speaker_window(rows, 1), (2000, 4000))
        result = process_meeting(self.service, self.path)
        self.assertEqual(result[1]["start_ms"], 1000)

    def test_streamed_text_is_published_before_model_returns(self):
        rows = self.service.moss.transcribe.return_value
        emitted = []
        def infer(_path, _duration, *, cancelled, on_segment):
            on_segment(rows[0])
            self.assertEqual([e['text'] for e in emitted if e['type'] == 'segment'], ['第一句。'])
            self.assertFalse(any(e['type'] == 'info' for e in emitted))
            on_segment(rows[1])
            return rows
        self.service.moss.transcribe.side_effect = infer
        result = process_meeting(self.service, self.path, progress=emitted.append)
        self.assertEqual(len(result), 2)
        self.assertEqual([e['index'] for e in emitted if e['type'] == 'segment'], [0, 1])
        self.assertEqual(emitted[-1]['total_segments'], 2)

    def test_streaming_identity_waits_for_future_overlap_watermark(self):
        rows = [
            {'start_ms': 0, 'end_ms': 4000, 'diarizationSpeaker': 'S01', 'text': '一'},
            {'start_ms': 1500, 'end_ms': 3500, 'diarizationSpeaker': 'S02', 'text': '二'},
            {'start_ms': 4000, 'end_ms': 5000, 'diarizationSpeaker': 'S01', 'text': '三'},
        ]
        emitted = []
        def infer(_path, _duration, *, cancelled, on_segment):
            on_segment(rows[0])
            on_segment(rows[1])
            self.service.extract_embedding.assert_not_called()
            self.assertFalse(any(e['type'] == 'segment' for e in emitted))
            on_segment(rows[2])
            self.assertEqual(self.service.extract_embedding.call_count, 1)
            self.assertEqual(len(self.service.extract_embedding.call_args.args[0]), 24000)
            return rows
        self.service.moss.transcribe.side_effect = infer
        result = process_meeting(self.service, self.path, allowed_speaker_ids=['a'],
                                 match_registered_speakers=True, progress=emitted.append)
        self.assertEqual(len(result), 3)
        self.assertEqual(self.service.extract_embedding.call_count, 2)
        self.assertEqual([e['index'] for e in emitted if e['type'] == 'segment'], [0, 1, 2])
        self.assertEqual(result[0]['end_ms'], 4000)

    def test_overlong_input_is_rejected_without_inference(self):
        with patch.dict(os.environ, {"MOSS_MAX_AUDIO_SECONDS": "1"}):
            with self.assertRaises(AudioTooLong):
                process_meeting(self.service, self.path)
        self.service.moss.transcribe.assert_not_called()
        self.assertFalse(self.service._offline_lock.locked())

    def test_identity_failure_keeps_text_as_unknown(self):
        self.service.extract_embedding.side_effect = RuntimeError("unavailable")
        rows = process_meeting(self.service, self.path, allowed_speaker_ids=["a"], match_registered_speakers=True)
        self.assertEqual([r["text"] for r in rows], ["第一句。", "第二句。"])
        self.assertTrue(all(r["speakerId"] is None and r["confidence"] == 0 for r in rows))

    def test_decoder_rejects_playlists_before_model_inference(self):
        playlist = os.path.join(self.temp.name, "list.ffconcat")
        with open(playlist, "w") as out:
            out.write(f"ffconcat version 1.0\nfile '{self.path}'\n")
        with self.assertRaises(InvalidAudio):
            process_meeting(self.service, playlist)
        self.service.moss.transcribe.assert_not_called()

    def test_export_refuses_to_overwrite_existing_or_reviewed_files(self):
        target = os.path.join(self.temp.name, "review.md")
        with open(target, "w") as out:
            out.write("人工已核对")
        with self.assertRaises(FileExistsError):
            export_markdown([], target, self.path)
        with open(target) as saved:
            self.assertEqual(saved.read(), "人工已核对")

    def test_busy_rejected_before_decode(self):
        self.service._offline_lock.acquire()
        try:
            with self.assertRaises(MossBusy):
                process_meeting(self.service, self.path)
            self.service.moss.transcribe.assert_not_called()
        finally:
            self.service._offline_lock.release()


if __name__ == "__main__":
    unittest.main()
