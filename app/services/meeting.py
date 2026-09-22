#!/usr/bin/env python3
"""离线唯一链路：完整音频 MOSS 转写/匿名分人 → CAM++ 保守身份验证。

JSON、SSE、已保存录音及 CLI 共用 process_meeting；不再运行旧离线 ASR/对齐分支。
"""

import argparse
from collections.abc import Collection
from datetime import datetime
import os
import subprocess
import tempfile
import time
import wave

import numpy as np

from app.core import CONFIG, ModelService, format_time
from app.services.moss import AudioTooLong, InvalidAudio, MossBusy, MossCancelled, MossError, MossTimeout, MossUnavailable, max_audio_seconds


def _decode_audio(audio_path, wav_path, cancelled=None):
    """与验收基准相同的 FFmpeg PCM16 解码；先限长/下混，避免展开原采样率多通道大数组。"""
    limit = max_audio_seconds()
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-xerror", "-threads", "2",
        "-protocol_whitelist", "file,pipe",
        "-format_whitelist", "wav,mp3,mov,mp4,m4a,3gp,3g2,mj2,aac,flac,ogg,aiff,asf,matroska,webm,amr",
        "-i", os.path.abspath(audio_path), "-t", str(limit + 0.1),
        "-vn", "-sn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-map_metadata", "-1", "-f", "wav", wav_path,
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise MossUnavailable("本机缺少 FFmpeg，无法解码录音") from exc
    try:
        deadline = time.monotonic() + 120
        while process.poll() is None:
            if cancelled is not None and cancelled.is_set():
                raise MossCancelled("离线转写已取消")
            if time.monotonic() >= deadline:
                raise MossTimeout("音频解码超时，未启动模型推理")
            time.sleep(0.05)
        if process.returncode:
            raise InvalidAudio("无法解码录音，请检查格式；不接受网络音频引用或播放列表")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    try:
        with wave.open(wav_path, "rb") as decoded:
            if (decoded.getframerate(), decoded.getnchannels(), decoded.getsampwidth()) != (16000, 1, 2):
                raise InvalidAudio("解码结果不是 16kHz 单声道 PCM16")
            # 多解码 0.1s 只用于检测超长；超过上限绝不送入 MOSS。
            if decoded.getnframes() > int(limit * 16000):
                raise AudioTooLong(f"录音超过当前 MOSS 完整上下文上限 {limit:g} 秒，未截断或分块识别")
            pcm = decoded.readframes(decoded.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InvalidAudio("无法读取解码后的录音") from exc
    if not pcm:
        raise InvalidAudio("录音为空")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, 16000


def process_meeting(
    service: ModelService,
    audio_path: str,
    threshold: float = None,
    allowed_speaker_ids: Collection[str] | None = None,
    matching_scope=None,
    match_registered_speakers: bool = False,
    *,
    progress=None,
    cancelled=None,
) -> list:
    """保留源文本与重叠时间；匿名编号绝不当作注册身份或匹配置信度。"""
    if not service._offline_lock.acquire(blocking=False):
        raise MossBusy("已有离线录音正在处理，请稍后重试")

    def check_cancelled():
        if cancelled is not None and cancelled.is_set():
            raise MossCancelled("离线转写已取消")

    def emit(event):
        check_cancelled()
        if progress is not None:
            progress(event)

    try:
        threshold = CONFIG["speaker_threshold"] if threshold is None else threshold
        if match_registered_speakers and matching_scope is None:
            matching_scope = service.build_matching_scope(allowed_speaker_ids)
        emit({"type": "status", "phase": "loading", "message": "正在解码完整录音..."})
        segments, transcript = [], []
        anonymous_names = {}

        def publish_available(*, final=False):
            while len(transcript) < len(segments):
                check_cancelled()
                index = len(transcript)
                segment = segments[index]
                # 后续起点单调递增。等水位越过当前段末，才排除尚未生成的跨人重叠；
                # 不为提前显示姓名而放松声纹取样保护。匿名模式不需等待这个水位。
                if (match_registered_speakers and not final
                        and segments[-1]["start_ms"] < segment["end_ms"]):
                    break
                label = segment["diarizationSpeaker"]
                anonymous_names.setdefault(label, f"陌生人{len(anonymous_names) + 1}")
                speaker_id, confidence = None, 0.0
                if match_registered_speakers:
                    start, end = _clean_speaker_window(segments, index)
                    if end - start >= CONFIG["offline_scoped_match_min_duration_ms"]:
                        end = min(end, start + 12000)
                        try:
                            embedding = service.extract_embedding(audio[start * sr // 1000:end * sr // 1000])
                            if embedding is not None:
                                speaker_id, confidence = service.match_registered_speaker_guarded(
                                    embedding, threshold=threshold, duration_ms=end - start,
                                    match_scope=matching_scope, allowed_speaker_ids=allowed_speaker_ids,
                                )
                        except Exception:
                            speaker_id, confidence = None, 0.0
                            emit({"type": "status", "phase": "identity_warning", "message": "声纹验证失败，保留匿名说话人与原文"})
                    # 不从同一匿名簇、上一句或文件名强行继承真实身份。
                    if speaker_id is None:
                        confidence = 0.0
                item = {
                    **segment, "time": format_time(segment["start_ms"]),
                    "speakerId": speaker_id,
                    "speaker": service.get_speaker_name(speaker_id) if speaker_id is not None else anonymous_names[label],
                    "confidence": round(float(confidence), 2),
                }
                transcript.append(item)
                emit({"type": "segment", "index": index, **item})

        def on_segment(segment):
            check_cancelled()
            segments.append(segment)
            publish_available()

        # worker 只读取请求专属的本机 WAV；等待推理结束后才删除，不切分音频输入。
        with tempfile.TemporaryDirectory(prefix="voiceprint-moss-") as work_dir:
            wav_path = os.path.join(work_dir, "audio.wav")
            audio, sr = _decode_audio(audio_path, wav_path, cancelled)
            emit({"type": "status", "phase": "transcribing", "message": "正在转写..."})
            result = service.moss.transcribe(
                wav_path, len(audio) / sr, cancelled=cancelled, on_segment=on_segment,
            )
        if segments != result[:len(segments)]:
            raise MossError("MOSS 流式片段与最终结果不一致")
        for segment in result[len(segments):]:
            on_segment(segment)
        # 总数只能在正常 EOS 后确定；info 保持整数总数，允许它晚于流式 segment。
        emit({"type": "info", "method": "moss", "total_segments": len(result)})
        publish_available(final=True)
        return transcript
    finally:
        service._offline_lock.release()


def _clean_speaker_window(segments, index):
    """从原片段扣除其他匿名说话人的重叠区，选最长连续区做身份验证。"""
    current = segments[index]
    windows = [(current["start_ms"], current["end_ms"])]
    for other in segments:
        if other["start_ms"] >= current["end_ms"]:
            break
        if other["diarizationSpeaker"] == current["diarizationSpeaker"] or other["end_ms"] <= current["start_ms"]:
            continue
        remaining = []
        for start, end in windows:
            if other["end_ms"] <= start or other["start_ms"] >= end:
                remaining.append((start, end))
            else:
                if start < other["start_ms"]:
                    remaining.append((start, other["start_ms"]))
                if other["end_ms"] < end:
                    remaining.append((other["end_ms"], end))
        windows = remaining
    return max(windows, key=lambda span: span[1] - span[0], default=(0, 0))


def print_transcript(transcript: list):
    for item in transcript:
        print(f"[{item['time']}] {item['speaker']}: {item['text']}")


def export_markdown(transcript: list, output_path: str, audio_path: str):
    with open(output_path, "x", encoding="utf-8") as output:
        output.write(f"# 会议复核稿\n\n日期: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        output.write(f"音频文件: {os.path.basename(audio_path)}\n\n")
        output.write("> MOSS 自动转写；医学数字、术语与身份归属需人工核对。\n\n")
        for item in transcript:
            output.write(f"**[{item['time']}] {item['speaker']}**:\n\n> {item['text']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="MOSS + CAM++ 会议复核稿")
    parser.add_argument("--audio", "-a", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--threshold", "-t", type=float, default=None)
    parser.add_argument("--speaker-id", action="append", default=[], help="仅匹配这些已注册声纹 id；可重复")
    parser.add_argument("--device", "-d", default="cpu", help="CAM++ 设备；MOSS 使用独立 CUDA worker")
    args = parser.parse_args()
    if args.output and os.path.exists(args.output):
        parser.error("输出已存在，请为复核稿选择新文件，避免覆盖实时稿或人工修改")
    service = ModelService()
    try:
        service.load_models(device=args.device, load_live=False)
        missing = set(args.speaker_id) - service.registered_embeddings.keys()
        if missing:
            parser.error("存在未注册的 speaker-id")
        transcript = process_meeting(
            service, args.audio, args.threshold, args.speaker_id,
            match_registered_speakers=bool(args.speaker_id),
        )
        print_transcript(transcript)
        if args.output:
            export_markdown(transcript, args.output, args.audio)
    finally:
        service.close()


if __name__ == "__main__":
    main()
