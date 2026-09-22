#!/usr/bin/env python3
"""麦克风 CLI，与 WebSocket 使用同一个两遍流式会话实现。"""

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import signal

from app.core import ModelService
from app.services.live_session import run_live_session
from app.services.recording_store import recording_store


class MicrophoneSession:
    def __init__(self, stream, output_file):
        self.stream = stream
        self.output_file = output_file
        self.stopping = False
        self.file_id = None

    async def receive(self):
        if self.stopping:
            return {"type": "websocket.receive", "text": json.dumps({"type": "stop_recording"})}
        data = await asyncio.to_thread(self.stream.read, 1024, exception_on_overflow=False)
        return {"type": "websocket.receive", "bytes": data}

    async def send_json(self, payload):
        kind = payload.get("type")
        if kind == "transcript":
            print(f"\r[{payload['time']}] {payload['speaker']}: {payload['text']}", end="\n" if payload["isFinal"] else "", flush=True)
            if payload["isFinal"]:
                with open(self.output_file, "a", encoding="utf-8") as output:
                    note = "（精修失败，保留首遍稿）" if payload.get("degraded") else ""
                    output.write(f"**[{payload['time']}] {payload['speaker']}**{note}:\n> {payload['text']}\n\n")
        elif kind == "recording_saved":
            self.file_id = payload["fileId"]
            print(f"\n原始录音 fileId: {self.file_id}")
        elif kind == "error":
            print(f"\n警告: {payload['message']}")


async def record(args):
    import pyaudio
    service = ModelService()
    audio = stream = None
    try:
        await asyncio.to_thread(service.load_models, device=args.device)
        if set(args.speaker_id) - service.registered_embeddings.keys():
            raise ValueError("存在未注册的 speaker-id")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                            input=True, frames_per_buffer=1024)
        session = MicrophoneSession(stream, args.output)
        with open(args.output, "x", encoding="utf-8") as output:
            output.write(f"# 实时会议稿\n\n日期: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: setattr(session, "stopping", True))
        try:
            print("开始录音，Ctrl+C 停止并排空尾段。")
            await run_live_session(session, service, recording_store,
                                   priority="speed", allowed_speaker_ids=args.speaker_id)
        finally:
            loop.remove_signal_handler(signal.SIGINT)
        if session.file_id:
            recording = recording_store.resolve(session.file_id)
            print(f"实时稿: {args.output}\n原始录音: {recording.path}")
            print("会后复核请使用 python -m app.services.meeting --audio <原始录音路径> --output <新文件>，不要覆盖实时稿。")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if audio is not None:
            audio.terminate()
        service.close()


def main():
    parser = argparse.ArgumentParser(description="两遍实时会议记录（使用与 WebSocket 相同的流水线）")
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--output", "-o", default="live_meeting.md")
    parser.add_argument("--speaker-id", action="append", default=[], help="只匹配选定的声纹 id，可重复；默认不识别人名")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("输出已存在，请选择新的实时稿文件")
    asyncio.run(record(args))


if __name__ == "__main__":
    main()
