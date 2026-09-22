"""通过真实 FastAPI ASGI 路由验证 HTTP 契约，不需要下载模型或额外测试客户端依赖。"""
import asyncio
import io
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import wave

from app import server
from app.core import ModelService
from app.services.meeting_jobs import MeetingJobStore
from app.services.moss import MossCancelled, MossError
from app.services.recording_store import RecordingStore


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\x01\x00' * 32000)
    return output.getvalue()


def multipart(priority='speed'):
    boundary = 'test-voiceprint-boundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.wav"\r\n'
            'Content-Type: audio/wav\r\n\r\n').encode() + wav_bytes()
    body += (f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="priority"\r\n\r\n{priority}'
             f'\r\n--{boundary}--\r\n').encode()
    return body, f'multipart/form-data; boundary={boundary}'


async def request(path, body=b'', content_type='application/json', method='POST', disconnect_after=None, on_body=None):
    path, _, query = path.partition('?')
    completed = asyncio.Event()
    received = False
    messages = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        if disconnect_after is not None:
            await asyncio.to_thread(disconnect_after.wait, 3)
        else:
            await completed.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.body' and on_body is not None:
            on_body(message.get('body', b''))
        if message['type'] == 'http.response.body' and not message.get('more_body'):
            completed.set()

    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
             'http_version': '1.1', 'scheme': 'http', 'method': method,
             'path': path, 'raw_path': path.encode(), 'query_string': query.encode(),
             'root_path': '', 'server': ('test', 80), 'client': ('127.0.0.1', 1),
             'headers': [(b'content-type', content_type.encode()), (b'content-length', str(len(body)).encode())]}
    await server.app(scope, receive, send)
    status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
    data = b''.join(m.get('body', b'') for m in messages)
    return status, data


class MeetingHttpContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = ModelService()
        self.service.moss.transcribe = Mock(return_value=[{
            'start_ms': 0, 'end_ms': 1000, 'diarizationSpeaker': 'S01', 'text': '测试。',
        }])
        self.store = RecordingStore(self.temp.name)
        self.jobs = MeetingJobStore()
        self.patches = [patch.object(server, 'service', self.service), patch.object(server, 'recording_store', self.store),
                        patch.object(server, 'meeting_jobs', self.jobs)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        self.service.close()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_four_http_endpoints_return_identical_segments(self):
        file_id = self.store.save_pcm_wav(b'\x01\x00' * 32000)
        body, content_type = multipart('accuracy')
        results = []
        for path, data, mime, sse in [
            ('/v1/meeting/transcribe', body, content_type, False),
            ('/v1/meeting/transcribe/stream', body, content_type, True),
            (f'/v1/meeting/recordings/{file_id}/transcribe?priority=accuracy', b'', 'application/json', False),
            (f'/v1/meeting/recordings/{file_id}/transcribe/stream?priority=accuracy', b'', 'application/json', True),
        ]:
            with self.subTest(path=path):
                status, response = asyncio.run(request(path, data, mime))
                self.assertEqual(status, 200)
                if sse:
                    events = [json.loads(e[6:]) for e in response.decode().split('\n\n') if e.startswith('data: ')]
                    self.assertEqual(events[-1]['type'], 'done')
                    rows = [{k:v for k,v in e.items() if k not in {'index', 'type', 'eventId'}} for e in events if e['type'] == 'segment']
                else:
                    payload = json.loads(response)
                    self.assertEqual(payload['effectivePriority'], 'accuracy')
                    self.assertEqual(payload['processingMode'], 'unified')
                    rows = payload['transcript']
                results.append(rows)
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(self.service.moss.transcribe.call_count, 4)
        self.assertTrue(os.path.isfile(self.store.resolve(file_id).path))

    def test_asgi_refresh_keeps_inference_and_input_then_replays_without_restarting(self):
        entered, finish = threading.Event(), threading.Event()
        observed = []
        rows = self.service.moss.transcribe.return_value
        def infer(path, _duration, *, cancelled, on_segment):
            observed.append(path)
            on_segment(rows[0])
            entered.set()
            self.assertTrue(finish.wait(3))
            self.assertFalse(cancelled.is_set(), 'subscriber disconnect cancelled the job')
            self.assertTrue(os.path.exists(path))
            return rows
        self.service.moss.transcribe.side_effect = infer
        body, mime = multipart()
        async def run():
            status, _ = await request('/v1/meeting/transcribe/stream', body, mime, disconnect_after=entered)
            self.assertEqual(status, 200)
            job = next(iter(self.jobs.jobs.values()))
            self.assertFalse(job.finished)
            self.assertTrue(os.path.exists(observed[0]))
            finish.set()
            status, raw = await request(f'/v1/meeting/jobs/{job.id}/events', method='GET')
            events = [json.loads(x[6:]) for x in raw.decode().split('\n\n') if x.startswith('data: ')]
            self.assertEqual(status, 200)
            self.assertEqual(events[-1]['type'], 'done')
            self.assertEqual(sum(x['type'] == 'segment' for x in events), 1)
            self.assertEqual([e['eventId'] for e in events], list(range(1, len(events) + 1)))
            self.assertFalse(os.path.exists(observed[0]))
            self.assertEqual(self.service.moss.transcribe.call_count, 1)
            _, remaining = await request(f'/v1/meeting/jobs/{job.id}/events?after={len(events)-1}', method='GET')
            self.assertEqual(json.loads(remaining.decode()[6:])['type'], 'done')
        with self.assertNoLogs('asyncio', level='ERROR'):
            asyncio.run(run())

    def test_explicit_cancel_and_shutdown_wait_for_input_cleanup(self):
        for stopping, saved_source in ((False, False), (True, False), (False, True)):
            entered = threading.Event()
            observed = []
            self.jobs = MeetingJobStore()
            def infer(path, _duration, *, cancelled, on_segment):
                entered.set()
                self.assertTrue(cancelled.wait(3))
                observed.append((path, os.path.exists(path)))
                raise MossCancelled('cancelled')
            self.service.moss.transcribe.side_effect = infer
            body, mime = multipart()
            file_id = self.store.save_pcm_wav(b'\x01\x00' * 32000) if saved_source else None
            async def run():
                if file_id:
                    await request(f'/v1/meeting/recordings/{file_id}/transcribe/stream', disconnect_after=entered)
                else:
                    await request('/v1/meeting/transcribe/stream', body, mime, disconnect_after=entered)
                job = next(iter(self.jobs.jobs.values()))
                if stopping:
                    await self.jobs.close()
                else:
                    status, _ = await request(f'/v1/meeting/jobs/{job.id}/cancel')
                    self.assertEqual(status, 200)
                    await job.task
                self.assertEqual(job.state, 'cancelled')
                self.assertTrue(observed[0][1])
                self.assertFalse(os.path.exists(observed[0][0]))
                if file_id:
                    self.assertTrue(os.path.exists(self.store.resolve(file_id).path))
                events = [json.loads(x[6:]) for x in job.events]
                self.assertEqual(events[-1]['type'], 'error')
                self.assertNotIn('done', [e['type'] for e in events])
            with self.subTest(stopping=stopping, saved_source=saved_source), patch.object(server, 'meeting_jobs', self.jobs):
                asyncio.run(run())

    def test_real_sse_segment_is_sent_before_model_is_allowed_to_finish(self):
        delivered = threading.Event()
        rows = self.service.moss.transcribe.return_value
        def infer(_path, _duration, *, cancelled, on_segment):
            on_segment(rows[0])
            self.assertTrue(delivered.wait(3), 'only status/heartbeat arrived, no transcript')
            return rows
        self.service.moss.transcribe.side_effect = infer
        def on_body(body):
            for block in body.decode().split('\n\n'):
                if block.startswith('data: ') and json.loads(block[6:])['type'] == 'segment':
                    delivered.set()
        body, mime = multipart()
        status, raw = asyncio.run(request('/v1/meeting/transcribe/stream', body, mime, on_body=on_body))
        events = [json.loads(e[6:]) for e in raw.decode().split('\n\n') if e.startswith('data: ')]
        self.assertEqual(status, 200)
        self.assertTrue(delivered.is_set())
        self.assertEqual([e['type'] for e in events][-3:], ['segment', 'info', 'done'])

    def test_failure_after_streamed_text_emits_error_without_done(self):
        rows = self.service.moss.transcribe.return_value
        def infer(_path, _duration, *, cancelled, on_segment):
            on_segment(rows[0])
            raise MossError('未正常EOS')
        self.service.moss.transcribe.side_effect = infer
        body, mime = multipart()
        status, raw = asyncio.run(request('/v1/meeting/transcribe/stream', body, mime))
        events = [json.loads(e[6:]) for e in raw.decode().split('\n\n') if e.startswith('data: ')]
        self.assertEqual(status, 200)
        self.assertTrue(any(e['type'] == 'segment' for e in events))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertFalse(any(e['type'] == 'done' for e in events))

    def test_invalid_priority_is_400_before_inference(self):
        body, mime = multipart('other')
        status, _ = asyncio.run(request('/v1/meeting/transcribe/stream', body, mime))
        self.assertEqual(status, 400)
        self.service.moss.transcribe.assert_not_called()

    def test_upload_limit_is_413_before_inference(self):
        body, mime = multipart()
        with patch.object(server, 'MAX_UPLOAD_BYTES', 1024):
            status, _ = asyncio.run(request('/v1/meeting/transcribe', body, mime))
        self.assertEqual(status, 413)
        self.service.moss.transcribe.assert_not_called()

    def test_truncated_model_output_fails_instead_of_success(self):
        self.service.moss.transcribe.side_effect = MossError('输出不完整')
        body, mime = multipart()
        status, response = asyncio.run(request('/v1/meeting/transcribe', body, mime))
        self.assertEqual(status, 502)
        self.assertIn('输出不完整', json.loads(response)['detail'])

    def test_unknown_candidate_and_invalid_threshold_are_rejected(self):
        file_id = self.store.save_pcm_wav(b'\x01\x00' * 10)
        for suffix in ['allowed_speaker_ids=not-enrolled', 'threshold=nan', 'threshold=1.1']:
            status, _ = asyncio.run(request(f'/v1/meeting/recordings/{file_id}/transcribe?{suffix}'))
            self.assertEqual(status, 400)
        self.service.moss.transcribe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
