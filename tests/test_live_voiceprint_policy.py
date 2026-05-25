import unittest

from app.server import _live_voiceprint_enabled


class LiveVoiceprintPolicyTest(unittest.TestCase):
    def test_live_voiceprint_requires_selected_participants(self):
        self.assertFalse(_live_voiceprint_enabled(None))
        self.assertFalse(_live_voiceprint_enabled([]))
        self.assertTrue(_live_voiceprint_enabled(["张三"]))


if __name__ == "__main__":
    unittest.main()
