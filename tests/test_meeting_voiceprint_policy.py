import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from app.core import ModelService
from app.services.meeting import (process_meeting, _clean_speaker_window, export_markdown,
                                  _speaker_sample_windows, _verify_speaker_windows)
from app.services.moss import AudioTooLong, InvalidAudio, MossBusy, MossCancelled


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
        self.assertTrue(all(x["voiceprintScore"] is None and x["identityStatus"] == "not_requested" for x in rows))

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

    def test_long_segment_samples_middle_and_tail_without_changing_transcript(self):
        sf.write(self.path, np.linspace(-.1, .1, 16000 * 20), 16000)
        self.service.moss.transcribe.return_value = [
            {"start_ms": 1000, "end_ms": 19540, "diarizationSpeaker": "S01", "text": "连续发言。"}]
        self.service.match_registered_speaker_guarded.side_effect = [(None, .25), (None, .29), ("a", .324)]
        emitted = []
        result = process_meeting(self.service, self.path, allowed_speaker_ids=["a", "b"],
                                 match_registered_speakers=True, progress=emitted.append)
        self.assertEqual(result[0]["speakerId"], "a")
        self.assertEqual(result[0]["confidence"], .32)
        self.assertEqual(result[0]["voiceprintScore"], .324)
        self.assertEqual(result[0]["identityStatus"], "matched")
        self.assertEqual(result[0]["end_ms"], 19540)
        calls = self.service.extract_embedding.call_args_list
        self.assertEqual([len(c.args[0]) for c in calls], [192000, 96000, 96000])
        self.assertLess(calls[0].args[0][0], calls[1].args[0][0])
        self.assertLess(calls[1].args[0][0], calls[2].args[0][0])
        event = next(e for e in emitted if e['type'] == 'segment')
        self.assertEqual({k: event[k] for k in result[0]}, result[0])

    def test_sampling_is_bounded_and_keeps_short_segments_unchanged(self):
        self.assertEqual(_speaker_sample_windows(500, 12500), [(500, 12500)])
        self.assertEqual(_speaker_sample_windows(34510, 53050),
                         [(34510, 46510), (40780, 46780), (47050, 53050)])
        for start, end in [(0, 12001), (2000, 1000000)]:
            windows = _speaker_sample_windows(start, end)
            self.assertEqual(len(windows), 3)
            self.assertTrue(all(start <= a < b <= end and b-a <= 12000 for a,b in windows))

    def verify_windows(self):
        return _verify_speaker_windows(self.service, np.ones(20 * 16000), 16000, 0, 20000,
                                       .3, None, ["a", "b"], lambda: None)

    def test_conflicting_passed_windows_do_not_pick_highest_score(self):
        self.service.match_registered_speaker_guarded.side_effect = [("a", .8), (None, .2), ("b", .9)]
        self.assertEqual(self.verify_windows(), (None, 0, .9, "conflicting_windows"))

    def test_unconfirmed_score_is_preserved_without_fabricating_confidence(self):
        self.service.match_registered_speaker_guarded.side_effect = [(None, .25), (None, .27), (None, .29)]
        self.assertEqual(self.verify_windows(), (None, 0, .29, "unconfirmed"))
        self.service.match_registered_speaker_guarded.side_effect = None
        result = process_meeting(self.service, self.path, allowed_speaker_ids=["a"], match_registered_speakers=True)
        self.assertEqual(result[0]["confidence"], 0)
        self.assertEqual(result[0]["voiceprintScore"], .22)
        self.assertEqual(result[0]["identityStatus"], "unconfirmed")

    def test_matched_score_is_mean_of_passed_windows_not_maximum(self):
        self.service.match_registered_speaker_guarded.side_effect = [("a", .8), (None, .2), ("a", .4)]
        speaker, confidence, score, status = self.verify_windows()
        self.assertEqual((speaker, status), ("a", "matched"))
        self.assertAlmostEqual(confidence, .6)
        self.assertEqual(confidence, score)

    def test_failed_embedding_does_not_silently_skip_possible_conflict(self):
        self.service.match_registered_speaker_guarded.return_value = ("a", .8)
        self.service.extract_embedding.side_effect = [np.ones(2), None]
        self.assertEqual(self.verify_windows(), (None, 0, None, "unavailable"))

    def test_cancellation_between_windows_is_not_swallowed(self):
        check = Mock(side_effect=[None, MossCancelled("cancelled")])
        with self.assertRaises(MossCancelled):
            _verify_speaker_windows(self.service, np.ones(20 * 16000), 16000, 0, 20000,
                                    .3, None, ["a", "b"], check)
        self.assertEqual(self.service.extract_embedding.call_count, 1)

    def test_multi_window_preserves_real_outside_winner_and_single_candidate_guards(self):
        self.service._emb_names = ["a", "b", "outside"]
        self.service._emb_name_to_idx = {name: i for i, name in enumerate(self.service._emb_names)}
        self.service._emb_matrix = np.eye(3, dtype=np.float32)
        self.service.match_registered_speaker_guarded = ModelService.match_registered_speaker_guarded.__get__(self.service)
        for ids in [["a"], ["a", "b"]]:
            self.service.extract_embedding.side_effect = [
                np.array([.6, 0, .8], dtype=np.float32),
                np.array([1, 0, 0], dtype=np.float32),
                np.array([1, 0, 0], dtype=np.float32),
            ]
            scope = self.service.build_matching_scope(ids)
            result = _verify_speaker_windows(self.service, np.ones(20 * 16000), 16000, 0, 20000,
                                             .3, scope, ids, lambda: None)
            self.assertIsNone(result[0])
            self.assertEqual(result[1], 0)
            self.assertEqual(result[3], "outside_winner")

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
