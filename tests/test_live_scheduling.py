import asyncio
import threading
import time
import unittest

import torch

from app import server
from app.core import CONFIG
from app.services import live_session
from app.services.live_session import InferenceStage, SegmentWorkQueue


def work(revision, *, final=False, segment="segment-1"):
    return {"segmentId": segment, "revision": revision, "isFinal": final, "audio": bytes(revision)}


class SegmentWorkQueueTest(unittest.TestCase):
    def drain(self, queue, count):
        async def run():
            return [await queue.get() for _ in range(count)]
        return asyncio.run(run())

    def test_pending_interims_coalesce_to_latest_snapshot(self):
        async def run():
            queue = SegmentWorkQueue(maxsize=8)
            for revision in (1, 2, 3):
                await queue.put(work(revision))
            self.assertEqual(queue._queue.qsize(), 1)
            item = await queue.get()
            self.assertEqual((item["revision"], item["audio"]), (3, bytes(3)))
        asyncio.run(run())

    def test_final_supersedes_pending_interim_and_is_kept(self):
        async def run():
            queue = SegmentWorkQueue(maxsize=8)
            await queue.put(work(1))
            await queue.put(work(2, final=True))
            await queue.put(work(1, segment="segment-2"))
            first, final, nxt = [await queue.get() for _ in range(3)]
            self.assertTrue(first.get("superseded"))
            self.assertTrue(final["isFinal"])
            self.assertFalse(final.get("superseded"))
            self.assertEqual(nxt["segmentId"], "segment-2")
            self.assertFalse(nxt.get("superseded"))
        asyncio.run(run())

    def test_dequeued_interim_is_not_updated_in_place(self):
        async def run():
            queue = SegmentWorkQueue(maxsize=8)
            await queue.put(work(1))
            taken = await queue.get()
            await queue.put(work(2))
            self.assertEqual(taken["revision"], 1)
            self.assertEqual((await queue.get())["revision"], 2)
            await queue.put(None)
            self.assertIsNone(await queue.get())
        asyncio.run(run())


class InferenceStageTest(unittest.TestCase):
    def test_calls_run_in_submission_order_with_metrics(self):
        stage = InferenceStage("test-fifo", 1)
        stage.start()
        order = []

        def call(i):
            time.sleep(0.005)
            order.append((i, threading.current_thread().name))
            return i

        async def run():
            return await asyncio.gather(*(stage.run(call, i) for i in range(5)))

        self.assertEqual(asyncio.run(run()), list(range(5)))
        self.assertEqual([i for i, _ in order], list(range(5)))
        self.assertEqual(len({name for _, name in order}), 1)
        snapshot = stage.snapshot()
        self.assertEqual((snapshot["pending"], snapshot["calls"]), (0, 5))
        self.assertEqual(snapshot["recent"]["samples"], 5)
        self.assertGreater(snapshot["recent"]["wait_ms"]["max"], 0)

    def test_exception_propagates_and_releases_pending(self):
        stage = InferenceStage("test-error", 1)

        def fail():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(stage.run(fail))
        self.assertEqual((stage.snapshot()["pending"], stage.snapshot()["calls"]), (0, 1))

    def test_stage_thread_count_does_not_leak_into_process_default(self):
        def fresh_thread_default():
            value = []
            thread = threading.Thread(target=lambda: value.append(torch.get_num_threads()))
            thread.start()
            thread.join()
            return value[0]

        async def threads(stage):
            return await stage.run(torch.get_num_threads)

        default = fresh_thread_default()
        live_session._start_stages()
        single = InferenceStage("test-threads", 1)
        self.assertEqual(asyncio.run(threads(single)), 1)
        self.assertEqual(asyncio.run(threads(live_session.VAD_STAGE)), CONFIG["live_vad_cpu_threads"])
        self.assertEqual(asyncio.run(threads(live_session.NANO_STAGE)), default)
        self.assertEqual(fresh_thread_default(), default)


class LiveMetricsTest(unittest.TestCase):
    def test_metrics_endpoint_reports_stages_without_content(self):
        response = asyncio.run(server.get_live_metrics())
        self.assertEqual(set(response["stages"]), {"vad", "paraformer", "nano", "campp"})
        self.assertIsInstance(response["active_sessions"], int)
        for stage in response["stages"].values():
            self.assertEqual(
                set(stage), {"pending", "calls", "wait_seconds_total", "run_seconds_total", "recent"},
            )


if __name__ == "__main__":
    unittest.main()
