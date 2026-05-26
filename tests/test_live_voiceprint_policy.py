import unittest

from app.server import _selected_speaker_matching_enabled


class SelectedSpeakerMatchingPolicyTest(unittest.TestCase):
    def test_speaker_matching_requires_selected_participants(self):
        self.assertFalse(_selected_speaker_matching_enabled(None))
        self.assertFalse(_selected_speaker_matching_enabled([]))
        self.assertTrue(_selected_speaker_matching_enabled(["张三"]))


if __name__ == "__main__":
    unittest.main()
