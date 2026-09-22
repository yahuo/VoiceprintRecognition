import asyncio
import json
import time
import unittest
from unittest.mock import Mock

from fastapi import HTTPException
from app.services.meeting_jobs import MeetingJob, MeetingJobStore, normalize_job_id


class MeetingJobLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MeetingJobStore(retention_seconds=60, max_jobs=2)

    async def asyncTearDown(self):
        await self.store.close()

    async def test_multiple_subscribers_replay_one_background_task(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        calls = []
        cleanup = Mock()
        async def run(publish, cancelled):
            calls.append(1)
            publish({'type': 'segment', 'index': 0, 'text': '原文118、26'})
            entered.set()
            await finish.wait()
            self.assertFalse(cancelled.is_set())
            return {'segments': 1}
        job = self.store.start(None, 'sample.wav', run, cleanup)
        await entered.wait()
        first = job.stream()
        await anext(first)
        await first.aclose()
        self.assertFalse(job.cancelled.is_set())
        cleanup.assert_not_called()
        finish.set()
        a, b = await asyncio.gather(self.collect(job), self.collect(job))
        self.assertEqual(a, b)
        self.assertEqual(calls, [1])
        self.assertEqual(a[-1]['type'], 'done')
        self.assertEqual(a[1]['text'], '原文118、26')
        cleanup.assert_called_once()

    async def collect(self, job, after=0):
        return [json.loads(wire[6:]) async for wire in job.stream(after) if wire.startswith('data: ')]

    async def test_duplicate_id_and_other_job_are_rejected_while_running(self):
        finish = asyncio.Event()
        async def run(*_):
            await finish.wait()
            return {'segments': 0}
        job = self.store.start(None, 'sample', run, lambda: None)
        for job_id, status in ((job.id, 409), (None, 503)):
            with self.assertRaises(HTTPException) as caught:
                self.store.start(job_id, 'sample', run, lambda: None)
            self.assertEqual(caught.exception.status_code, status)
        finish.set()
        await job.task

    async def test_retention_and_max_history_do_not_remove_active_jobs(self):
        async def run(*_):
            return {'segments': 0}
        first = self.store.start(None, 'one', run, lambda: None)
        await first.task
        second = self.store.start(None, 'two', run, lambda: None)
        await second.task
        third = self.store.start(None, 'three', run, lambda: None)
        await third.task
        with self.assertRaises(HTTPException):
            self.store.get(first.id)
        second.finished_at = time.time() - 61
        with self.assertRaises(HTTPException):
            self.store.get(second.id)
        self.assertIs(self.store.get(third.id), third)
        self.assertEqual(len(self.store.jobs), 1)

    async def test_event_budget_fails_closed_and_requests_cancellation(self):
        async def run(publish, cancelled):
            publish({'type': 'segment', 'index': 0, 'text': 'x' * 2000})
            self.assertTrue(cancelled.is_set())
            return {'segments': 1}
        job = self.store.start(None, 'sample', run, lambda: None)
        job.MAX_EVENT_BYTES = 1024
        events = await self.collect(job)
        self.assertEqual(job.state, 'error')
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('done', [e['type'] for e in events])

    async def test_subscribers_are_bounded_and_do_not_own_task(self):
        finish = asyncio.Event()
        async def run(*_):
            await finish.wait()
            return {'segments': 0}
        job = self.store.start(None, 'sample', run, lambda: None)
        readers = [job.stream() for _ in range(job.MAX_SUBSCRIBERS + 1)]
        for reader in readers[:-1]:
            await anext(reader)
        denied = json.loads((await anext(readers[-1]))[6:])
        self.assertEqual(denied['type'], 'error')
        self.assertEqual(job.subscribers, job.MAX_SUBSCRIBERS)
        for reader in readers:
            await reader.aclose()
        self.assertEqual(job.subscribers, 0)
        self.assertFalse(job.cancelled.is_set())
        finish.set()
        await job.task

    def test_only_opaque_uuid4_ids_are_accepted(self):
        value = normalize_job_id()
        self.assertEqual(normalize_job_id(value), value)
        for bad in ('one', '../file', '0' * 36, '00000000-0000-0000-0000-000000000000'):
            with self.assertRaises(HTTPException):
                normalize_job_id(bad)


if __name__ == '__main__':
    unittest.main()
