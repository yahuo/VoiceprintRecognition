#!/usr/bin/env python3
"""
Meeting recording storage.

The service API depends on this small store boundary so local files can be
replaced by an object store implementation later without changing WebSocket
or REST contracts.
"""

from __future__ import annotations

import os
import uuid
import wave
from dataclasses import dataclass

from app.core import PROJECT_ROOT


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
DEFAULT_RECORDINGS_DIR = os.environ.get(
    "RECORDINGS_DIR",
    os.path.join(PROJECT_ROOT, "recordings"),
)


@dataclass(frozen=True)
class RecordingFile:
    file_id: str
    path: str
    filename: str


class RecordingWriter:
    def __init__(self, file_id: str, temp_path: str, final_path: str, wav_file: wave.Wave_write):
        self.file_id = file_id
        self.temp_path = temp_path
        self.final_path = final_path
        self._wav_file = wav_file
        self._closed = False
        self.frames_written = 0

    def write_pcm(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        if len(pcm_bytes) % SAMPLE_WIDTH_BYTES != 0:
            pcm_bytes = pcm_bytes[:-(len(pcm_bytes) % SAMPLE_WIDTH_BYTES)]
            if not pcm_bytes:
                return
        self._wav_file.writeframes(pcm_bytes)
        self.frames_written += len(pcm_bytes) // SAMPLE_WIDTH_BYTES

    def commit(self) -> str:
        self._close()
        if self.frames_written == 0:
            self.abort()
            raise ValueError("录音内容为空")
        os.replace(self.temp_path, self.final_path)
        return self.file_id

    def abort(self):
        self._close()
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def _close(self):
        if not self._closed:
            self._wav_file.close()
            self._closed = True


class RecordingStore:
    def __init__(self, root_dir: str = DEFAULT_RECORDINGS_DIR):
        self.root_dir = root_dir

    def normalize_file_id(self, file_id: str) -> str:
        try:
            return str(uuid.UUID(str(file_id)))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("非法 fileId")

    def _path_for(self, file_id: str) -> str:
        normalized = self.normalize_file_id(file_id)
        return os.path.join(self.root_dir, f"{normalized}.wav")

    def _temp_path_for(self, file_id: str) -> str:
        normalized = self.normalize_file_id(file_id)
        return os.path.join(self.root_dir, f"{normalized}.wav.tmp")

    def _open_wav_writer(self, path: str) -> wave.Wave_write:
        wav_file = wave.open(path, "wb")
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        return wav_file

    def begin_pcm_wav(self) -> RecordingWriter:
        file_id = str(uuid.uuid4())
        os.makedirs(self.root_dir, exist_ok=True)
        temp_path = self._temp_path_for(file_id)
        final_path = self._path_for(file_id)
        return RecordingWriter(
            file_id=file_id,
            temp_path=temp_path,
            final_path=final_path,
            wav_file=self._open_wav_writer(temp_path),
        )

    def cleanup_temp_files(self) -> int:
        if not os.path.isdir(self.root_dir):
            return 0

        removed = 0
        for name in os.listdir(self.root_dir):
            if not name.endswith(".wav.tmp"):
                continue
            path = os.path.join(self.root_dir, name)
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed

    def save_pcm_wav(self, pcm_bytes: bytes) -> str:
        writer = self.begin_pcm_wav()
        try:
            writer.write_pcm(pcm_bytes)
            return writer.commit()
        except Exception:
            writer.abort()
            raise

    def resolve(self, file_id: str) -> RecordingFile:
        normalized = self.normalize_file_id(file_id)
        path = os.path.join(self.root_dir, f"{normalized}.wav")
        if not os.path.exists(path):
            raise FileNotFoundError(normalized)
        return RecordingFile(
            file_id=normalized,
            path=path,
            filename=f"{normalized}.wav",
        )

    def delete_many(self, file_ids: list[str]) -> dict[str, list[str]]:
        result = {
            "deleted": [],
            "missing": [],
            "failed": [],
        }

        for raw_file_id in file_ids:
            normalized = self.normalize_file_id(raw_file_id)
            path = self._path_for(normalized)
            if not os.path.exists(path):
                result["missing"].append(normalized)
                continue
            try:
                os.remove(path)
                result["deleted"].append(normalized)
            except OSError:
                result["failed"].append(normalized)

        return result


recording_store = RecordingStore()
