import unittest

import numpy as np

from app.core import ModelService


class SpeakerScopeTest(unittest.TestCase):
    def setUp(self):
        self.service = ModelService()
        self.service._emb_names = ["spk-a", "spk-b", "spk-c"]
        self.service._emb_name_to_idx = {
            speaker_id: idx for idx, speaker_id in enumerate(self.service._emb_names)
        }
        base_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.service._emb_matrix = base_matrix
        self.service.registered_embeddings = {
            speaker_id: base_matrix[idx] for idx, speaker_id in enumerate(self.service._emb_names)
        }
        self.service.registered_speakers = {
            "spk-a": {"id": "spk-a", "name": "张三", "file": "unused-a.npy"},
            "spk-b": {"id": "spk-b", "name": "李四", "file": "unused-b.npy"},
            "spk-c": {"id": "spk-c", "name": "王五", "file": "unused-c.npy"},
        }

    def test_single_selected_participant_does_not_absorb_other_registered_speakers(self):
        query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"spk-b"})

        all_id, all_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
        )
        scoped_id, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
            match_scope=matching_scope,
            allowed_speaker_ids={"spk-b"},
        )

        self.assertEqual(all_id, "spk-a")
        self.assertAlmostEqual(all_score, 0.8, places=5)
        self.assertIsNone(scoped_id)
        self.assertAlmostEqual(scoped_score, 0.6, places=5)

    def test_single_selected_participant_can_still_match_self(self):
        query = np.array([0.2, 0.95, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"spk-b"})

        scoped_id, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=2500,
            match_scope=matching_scope,
            allowed_speaker_ids={"spk-b"},
        )

        self.assertEqual(scoped_id, "spk-b")
        self.assertGreater(scoped_score, 0.95)

    def test_empty_scope_has_no_candidates(self):
        query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope(set())

        speaker_id, score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
            match_scope=matching_scope,
            allowed_speaker_ids=set(),
        )

        self.assertIsNone(speaker_id)
        self.assertEqual(score, 0.0)

    def test_consensus_respects_selected_participants(self):
        matching_scope = self.service.build_matching_scope({"spk-b"})
        candidates = [
            (np.array([0.25, 0.96, 0.0], dtype=np.float32), 4500),
            (np.array([0.30, 0.92, 0.0], dtype=np.float32), 4300),
        ]

        speaker_id, score = self.service.match_registered_speaker_consensus(
            candidates,
            threshold=0.3,
            match_scope=matching_scope,
            allowed_speaker_ids={"spk-b"},
        )

        self.assertEqual(speaker_id, "spk-b")
        self.assertGreater(score, 0.45)

    def test_scoped_guarded_match_relaxes_long_audio_floor(self):
        query = np.array([0.32, 0.31, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"spk-a", "spk-b"})

        all_id, _ = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=1200,
        )
        scoped_id, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=1200,
            match_scope=matching_scope,
            allowed_speaker_ids={"spk-a", "spk-b"},
        )

        self.assertIsNone(all_id)
        self.assertEqual(scoped_id, "spk-a")
        self.assertGreater(scoped_score, 0.7)

    def test_single_selected_participant_disables_short_window_direct_match(self):
        query = np.array([0.2, 0.95, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"spk-b"})

        scoped_id, scoped_score = self.service.match_registered_speaker_short_window(
            query,
            threshold=0.3,
            duration_ms=600,
            match_scope=matching_scope,
            allowed_speaker_ids={"spk-b"},
        )

        self.assertIsNone(scoped_id)
        self.assertGreater(scoped_score, 0.9)


if __name__ == "__main__":
    unittest.main()
