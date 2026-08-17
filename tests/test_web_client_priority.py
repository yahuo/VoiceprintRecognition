import unittest
from html.parser import HTMLParser
from pathlib import Path


CLIENT_PATH = Path(__file__).resolve().parents[1] / "static" / "web_client.html"


class _InputCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}

    def handle_starttag(self, tag, attrs):
        if tag != "input":
            return
        attributes = dict(attrs)
        input_id = attributes.get("id")
        if input_id:
            self.inputs[input_id] = attributes


class WebClientPriorityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = CLIENT_PATH.read_text(encoding="utf-8")
        parser = _InputCollector()
        parser.feed(cls.html)
        cls.inputs = parser.inputs

    def test_live_and_upload_priority_controls_default_to_speed(self):
        for scope in ("live", "upload"):
            with self.subTest(scope=scope):
                speed = self.inputs[f"{scope}PrioritySpeed"]
                accuracy = self.inputs[f"{scope}PriorityAccuracy"]

                self.assertEqual(speed["type"], "radio")
                self.assertEqual(speed["name"], f"{scope}Priority")
                self.assertEqual(speed["value"], "speed")
                self.assertIn("checked", speed)
                self.assertEqual(accuracy["type"], "radio")
                self.assertEqual(accuracy["name"], f"{scope}Priority")
                self.assertEqual(accuracy["value"], "accuracy")
                self.assertNotIn("checked", accuracy)

    def test_all_transcription_requests_forward_the_selected_priority(self):
        self.assertIn(
            "getWsUrl(selectedParticipants, getTranscriptionPriority('live'))",
            self.html,
        )
        self.assertIn("url.searchParams.set('priority', priority);", self.html)
        self.assertIn(
            "formData.append('priority', getTranscriptionPriority('upload'));",
            self.html,
        )
        self.assertIn(
            "const priority = getTranscriptionPriority('live');",
            self.html,
        )
        self.assertIn(
            "params.set('priority', priority);",
            self.html,
        )

    def test_priority_controls_are_locked_during_active_requests(self):
        for scope in ("live", "upload"):
            with self.subTest(scope=scope):
                self.assertIn(
                    f"setPriorityControlDisabled('{scope}', true);",
                    self.html,
                )
                self.assertIn(
                    f"setPriorityControlDisabled('{scope}', false);",
                    self.html,
                )


if __name__ == "__main__":
    unittest.main()
