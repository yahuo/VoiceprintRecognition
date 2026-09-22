"""只模拟固定 vLLM 的 renderer/step API，断言不经过强制 FINAL_ONLY 的便捷接口。"""
import importlib
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.services import moss

with patch.dict(sys.modules, {'moss': moss}):
    stream_generate = importlib.import_module('app.services.moss_worker').stream_generate


class MossEngineStreamingTest(unittest.TestCase):
    def test_complete_input_enqueued_once_and_deltas_emitted_before_finish(self):
        chunks = [
            SimpleNamespace(finished=False, outputs=[SimpleNamespace(text='[0][S01]前段[1]', token_ids=[10], finish_reason=None)]),
            SimpleNamespace(finished=False, outputs=[SimpleNamespace(text='[1][S02]后段[2]', token_ids=[20], finish_reason=None)]),
            SimpleNamespace(finished=True, outputs=[SimpleNamespace(text='', token_ids=[99], finish_reason='stop')]),
        ]
        engine = Mock()
        pending = []
        engine.has_unfinished_requests.side_effect = lambda: bool(pending)
        engine.add_request.side_effect = lambda *_: pending.extend(chunks)
        engine.step.side_effect = lambda: [pending.pop(0)]
        whole_audio = object()
        prompt = {'prompt': 'test', 'multi_modal_data': {'audio': (whole_audio, 16000)}}
        llm = SimpleNamespace(llm_engine=engine, renderer=SimpleNamespace(render_cmpl=Mock(return_value=[prompt])))
        events = []
        def send(payload):
            events.append((payload, bool(pending)))
        params = object()
        stream_generate(llm, prompt, params, [99], send, 'one')
        llm.renderer.render_cmpl.assert_called_once_with([prompt])
        engine.add_request.assert_called_once_with('one', prompt, params)
        self.assertEqual([e['type'] for e, _ in events], ['delta', 'delta', 'result'])
        self.assertEqual([active for _, active in events], [True, True, False])
        self.assertTrue(events[-1][0]['eos_reached'])
        self.assertEqual(events[-1][0]['raw'], '[0][S01]前段[1][1][S02]后段[2]')

    def test_cancel_aborts_internal_request_without_rebuilding_engine(self):
        engine = Mock()
        pending = []
        engine.has_unfinished_requests.side_effect = lambda: bool(pending)
        def add(*_):
            pending.append(True)
            return 'internal-request-id'
        engine.add_request.side_effect = add
        engine.abort_request.side_effect = lambda *_args, **_kwargs: pending.clear()
        engine.step.return_value = [SimpleNamespace(finished=False, outputs=[
            SimpleNamespace(text='prefix', token_ids=[10], finish_reason=None)])]
        llm = SimpleNamespace(llm_engine=engine, renderer=SimpleNamespace(render_cmpl=Mock(return_value=['whole'])))
        cancelled = threading.Event()
        events = []
        def send(event):
            events.append(event)
            if event['type'] == 'delta':
                cancelled.set()
        stream_generate(llm, {}, object(), [99], send, 'external-id', cancelled)
        engine.abort_request.assert_called_once_with(['internal-request-id'], internal=True)
        self.assertEqual([e['type'] for e in events], ['delta', 'cancelled'])
        self.assertFalse(engine.has_unfinished_requests())

    def test_engine_without_terminal_output_cannot_send_a_success_result(self):
        engine = Mock()
        engine.has_unfinished_requests.return_value = False
        llm = SimpleNamespace(llm_engine=engine, renderer=SimpleNamespace(render_cmpl=Mock(return_value=['whole'])))
        send = Mock()
        with self.assertRaises(RuntimeError):
            stream_generate(llm, {}, object(), [99], send, 'one')
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
