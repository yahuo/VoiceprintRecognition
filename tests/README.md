# Tests

## `transcribe/stream` API benchmark

This test exercises the upload SSE endpoint and prints:

- time to first SSE event
- total request time
- number of `segment` events

Set an audio path before running:

```bash
export STREAM_TEST_AUDIO=/absolute/path/to/sample.wav
```

Run against a locally started server:

```bash
python3 -m unittest -v tests.test_transcribe_stream_api
```

Run against a Docker-published port from the host:

```bash
STREAM_TEST_BASE_URL=http://127.0.0.1:18008 python3 -m unittest -v tests.test_transcribe_stream_api
```

Run inside the container:

```bash
docker exec -e STREAM_TEST_AUDIO=/app/tests/sample.wav -it voiceprint-server python -m unittest -v tests.test_transcribe_stream_api
```

Optional thresholds:

```bash
export STREAM_TEST_MAX_FIRST_EVENT_SECONDS=25
export STREAM_TEST_MAX_TOTAL_SECONDS=60
```
