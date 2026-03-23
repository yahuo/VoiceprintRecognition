import unittest

import numpy as np

from app.core import ModelService


class SpeakerScopeTest(unittest.TestCase):
    def setUp(self):
        self.service = ModelService()
        self.service._emb_names = ["张三", "李四", "王五"]
        self.service._emb_name_to_idx = {
            name: idx for idx, name in enumerate(self.service._emb_names)
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
            name: base_matrix[idx] for idx, name in enumerate(self.service._emb_names)
        }

    def test_single_selected_participant_does_not_absorb_other_registered_speakers(self):
        query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"李四"})

        all_name, all_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
        )
        scoped_name, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
            match_scope=matching_scope,
            allowed_speakers={"李四"},
        )

        self.assertEqual(all_name, "张三")
        self.assertAlmostEqual(all_score, 0.8, places=5)
        self.assertEqual(scoped_name, "未知")
        self.assertAlmostEqual(scoped_score, 0.6, places=5)

    def test_single_selected_participant_can_still_match_self(self):
        query = np.array([0.2, 0.95, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"李四"})

        scoped_name, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=2500,
            match_scope=matching_scope,
            allowed_speakers={"李四"},
        )

        self.assertEqual(scoped_name, "李四")
        self.assertGreater(scoped_score, 0.95)

    def test_empty_scope_has_no_candidates(self):
        query = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope(set())

        name, score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=4000,
            match_scope=matching_scope,
            allowed_speakers=set(),
        )

        self.assertEqual(name, "未知")
        self.assertEqual(score, 0.0)

    def test_consensus_respects_selected_participants(self):
        matching_scope = self.service.build_matching_scope({"李四"})
        candidates = [
            (np.array([0.25, 0.96, 0.0], dtype=np.float32), 4500),
            (np.array([0.30, 0.92, 0.0], dtype=np.float32), 4300),
        ]

        name, score = self.service.match_registered_speaker_consensus(
            candidates,
            threshold=0.3,
            match_scope=matching_scope,
            allowed_speakers={"李四"},
        )

        self.assertEqual(name, "李四")
        self.assertGreater(score, 0.45)

    def test_scoped_guarded_match_relaxes_long_audio_floor(self):
        query = np.array([0.32, 0.31, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"张三", "李四"})

        all_name, _ = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=1200,
        )
        scoped_name, scoped_score = self.service.match_registered_speaker_guarded(
            query,
            threshold=0.3,
            duration_ms=1200,
            match_scope=matching_scope,
            allowed_speakers={"张三", "李四"},
        )

        self.assertEqual(all_name, "未知")
        self.assertEqual(scoped_name, "张三")
        self.assertGreater(scoped_score, 0.7)

    def test_single_selected_participant_disables_short_window_direct_match(self):
        query = np.array([0.2, 0.95, 0.0], dtype=np.float32)
        matching_scope = self.service.build_matching_scope({"李四"})

        scoped_name, scoped_score = self.service.match_registered_speaker_short_window(
            query,
            threshold=0.3,
            duration_ms=600,
            match_scope=matching_scope,
            allowed_speakers={"李四"},
        )

        self.assertEqual(scoped_name, "未知")
        self.assertGreater(scoped_score, 0.9)


if __name__ == "__main__":
    unittest.main()
