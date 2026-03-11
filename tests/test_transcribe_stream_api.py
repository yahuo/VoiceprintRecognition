import io
import json
import mimetypes
import os
import time
import unittest
import urllib.request
import uuid
from collections import Counter


def build_multipart_form(file_path: str, threshold: str | None = None) -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()
    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    if threshold is not None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="threshold"\r\n\r\n')
        body.write(str(threshold).encode())
        body.write(b"\r\n")

    body.write(f"--{boundary}\r\n".encode())
    body.write(
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    with open(file_path, "rb") as audio_file:
        body.write(audio_file.read())
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    return body.getvalue(), boundary


class TranscribeStreamApiTest(unittest.TestCase):
    def test_stream_emits_first_event_and_done(self):
        audio_path = os.environ.get("STREAM_TEST_AUDIO")
        if not audio_path:
            self.skipTest("Set STREAM_TEST_AUDIO to a local audio file path before running this test.")
        if not os.path.exists(audio_path):
            self.fail(f"STREAM_TEST_AUDIO does not exist: {audio_path}")

        base_url = os.environ.get("STREAM_TEST_BASE_URL", "http://127.0.0.1:8000")
        endpoint = f"{base_url.rstrip('/')}/v1/meeting/transcribe/stream"
        threshold = os.environ.get("STREAM_TEST_THRESHOLD")

        payload, boundary = build_multipart_form(audio_path, threshold)
        request = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        started_at = time.perf_counter()
        first_event_at = None
        first_segment_at = None
        segment_count = 0
        saw_done = False
        event_types: list[str] = []
        buffer = ""

        with urllib.request.urlopen(request, timeout=600) as response:
            while True:
                chunk = response.read1(4096) if hasattr(response, "read1") else response.read(4096)
                if not chunk:
                    break

                buffer += chunk.decode("utf-8", errors="ignore")
                parts = buffer.split("\n\n")
                buffer = parts.pop()

                for part in parts:
                    for line in part.splitlines():
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line[6:])
                        if first_event_at is None:
                            first_event_at = time.perf_counter()
                        event_type = payload.get("type", "unknown")
                        event_types.append(event_type)
                        if event_type == "segment":
                            if first_segment_at is None:
                                first_segment_at = time.perf_counter()
                            segment_count += 1
                        if event_type == "done":
                            saw_done = True

        finished_at = time.perf_counter()

        self.assertIsNotNone(first_event_at, "No SSE event was received from the stream endpoint.")
        self.assertIsNotNone(first_segment_at, "No segment event was received from the stream endpoint.")
        self.assertTrue(saw_done, "Stream completed without emitting a done event.")

        first_event_seconds = first_event_at - started_at
        first_segment_seconds = first_segment_at - started_at
        total_seconds = finished_at - started_at

        event_summary = Counter(event_types)
        avg_segment_seconds = total_seconds / segment_count if segment_count else 0.0

        print()
        print("=" * 64)
        print("SSE Upload Benchmark")
        print("=" * 64)
        print(f"Audio File     : {audio_path}")
        print(f"Endpoint       : {endpoint}")
        print("-" * 64)
        print(f"First Event    : {first_event_seconds:8.3f} s")
        print(f"First Segment  : {first_segment_seconds:8.3f} s")
        print(f"Total Time     : {total_seconds:8.3f} s")
        print(f"Segments       : {segment_count:8d}")
        print(f"Avg/Segment    : {avg_segment_seconds:8.3f} s")
        print("-" * 64)
        print(
            "Events         : "
            f"status={event_summary.get('status', 0)}, "
            f"info={event_summary.get('info', 0)}, "
            f"segment={event_summary.get('segment', 0)}, "
            f"done={event_summary.get('done', 0)}, "
            f"error={event_summary.get('error', 0)}"
        )
        print("=" * 64)

        max_first_event = os.environ.get("STREAM_TEST_MAX_FIRST_EVENT_SECONDS")
        if max_first_event:
            self.assertLessEqual(
                first_event_seconds,
                float(max_first_event),
                f"First SSE event exceeded threshold: {first_event_seconds:.3f}s > {max_first_event}s",
            )

        max_total = os.environ.get("STREAM_TEST_MAX_TOTAL_SECONDS")
        if max_total:
            self.assertLessEqual(
                total_seconds,
                float(max_total),
                f"Total stream time exceeded threshold: {total_seconds:.3f}s > {max_total}s",
            )


if __name__ == "__main__":
    unittest.main()
