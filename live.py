#!/usr/bin/env python3
"""
实时会议记录工具 (Live Meeting Transcription)
实时采集麦克风音频 → 自动切分 → 识别并输出带姓名的记录

基于 Fun-ASR-Nano (LLM) 和 CAM++ (声纹)
"""

import argparse
import os
import threading
import queue
import numpy as np
import pyaudio
import wave
import tempfile
from datetime import datetime

# 导入核心模块
from core import (
    CONFIG,
    ModelService,
    SpeakerTracker,
    match_speaker,
)


# 录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


class AudioProcessor:
    def __init__(self, device: str = "cpu", output_file: str = "live_meeting.md"):
        print("正在加载模型 (可能需要一些时间)...")
        
        # 使用核心模块的 ModelService
        self.service = ModelService()
        self.service.load_models(device=device, load_vad=False)  # 实时场景不需要 VAD
        
        # 使用 SpeakerTracker 进行说话人追踪
        self.tracker = SpeakerTracker()
        
        self.queue = queue.Queue()
        self.output_file = output_file
        self.running = True
        
        # 初始化输出文件
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"# 实时会议记录\n\n")
            f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")

    def process_segment(self, audio_data: bytes):
        """处理单个音频片段"""
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(audio_data)
                tmp_path = tmp.name
            
            # 1. ASR 识别 (使用核心模块的方法)
            text = self.service.transcribe_segment(tmp_path)
            
            if not text:
                os.remove(tmp_path)
                return
            
            # 2. 声纹识别
            emb = self.service.extract_embedding(tmp_path)
            
            speaker = "未知"
            score = 0.0
            
            if emb is not None:
                speaker, score = match_speaker(emb, self.service.registered_embeddings)
                
                # 使用 SpeakerTracker 处理继承逻辑
                speaker, score = self.tracker.update(speaker, score, self.service.registered_embeddings)
            
            # 过滤低置信度结果
            if score < CONFIG["min_confidence"]:
                os.remove(tmp_path)
                return
            
            # 输出结果
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\r[{timestamp}] {speaker} (conf:{score:.2f}): {text}")
            print("🎙️  正在聆听...", end="", flush=True)
            
            # 写入文件
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"**[{timestamp}] {speaker}** (conf:{score:.2f}):\n> {text}\n\n")
                
            os.remove(tmp_path)
            
        except Exception as e:
            print(f"\n处理出错: {e}")

    def worker(self):
        """后台工作线程"""
        while self.running:
            try:
                audio_data = self.queue.get(timeout=1)
                self.process_segment(audio_data)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker Error: {e}")

    def start(self):
        """开始录音和处理"""
        # 启动处理线程
        t = threading.Thread(target=self.worker)
        t.daemon = True
        t.start()
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("\n🎙️  开始录音... (按 Ctrl+C 停止)")
        print("🎙️  正在聆听...", end="", flush=True)
        
        frames = []
        silent_chunks = 0
        is_speaking = False
        chunks_per_second = RATE / CHUNK
        
        try:
            while self.running:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # 简单能量检测
                energy = np.abs(audio_np).mean()
                
                if energy > CONFIG["silence_energy"]:
                    is_speaking = True
                    silent_chunks = 0
                else:
                    if is_speaking:
                        silent_chunks += 1
                
                if is_speaking:
                    frames.append(data)
                
                # 判断一句话结束 (使用核心配置)
                if is_speaking and silent_chunks > CONFIG["silence_duration"] * chunks_per_second:
                    # 只有当录音长度足够长时才处理 (比如至少 0.5秒)
                    if len(frames) > 0.5 * chunks_per_second:
                        print("\n⏳ 正在处理...", end="", flush=True)
                        audio_content = b''.join(frames)
                        self.queue.put(audio_content)
                    
                    # 重置
                    frames = []
                    is_speaking = False
                    silent_chunks = 0
                    
        except KeyboardInterrupt:
            print("\n\n🛑 停止录音")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            print(f"✅ 会议记录已保存至: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="实时会议记录工具")
    parser.add_argument("--device", "-d", default="cpu", help="运行设备 (默认: cpu)")
    parser.add_argument("--output", "-o", default="live_meeting.md", help="输出文件路径")
    parser.add_argument("--threshold", "-t", type=int, default=None, help="静音能量阈值")
    args = parser.parse_args()
    
    if args.threshold:
        CONFIG["silence_energy"] = args.threshold
    
    processor = AudioProcessor(device=args.device, output_file=args.output)
    processor.start()


if __name__ == "__main__":
    main()
