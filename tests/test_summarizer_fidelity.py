import copy
import unittest
from unittest.mock import Mock, patch

from app.services.summarizer import summarize_meeting


class SummarizerFidelityTest(unittest.TestCase):
    def test_original_numbers_and_unknown_identity_are_preserved_without_mutating_input(self):
        transcript = [{'speaker':'未知', 'time':'00:01', 'text':'血压118、26，无发热。'}]
        original = copy.deepcopy(transcript)
        response = Mock(status_code=200)
        response.json.return_value = {'choices':[{'message':{'content':'待核对：血压118、26。'}}]}
        with patch('app.services.summarizer.requests.post', return_value=response) as post:
            result = summarize_meeting(transcript, base_url='https://test.invalid/v1', api_key='synthetic-key')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(transcript, original)
        messages = post.call_args.kwargs['json']['messages']
        self.assertIn('血压118、26，无发热。', messages[1]['content'])
        self.assertIn('未知', messages[1]['content'])
        self.assertIn('不得用医学常识', messages[0]['content'])
        self.assertIn('待核对', messages[0]['content'])


if __name__ == '__main__':
    unittest.main()
