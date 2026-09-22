"""在独立 CUDA/vLLM 环境运行；只接收本机 canonical WAV 路径，不开放网络端口。"""

import json
import os
from pathlib import Path
import queue
import sys
import threading
import wave

from moss import DEFAULT_PROMPT, max_audio_seconds


class RequestCommands:
    """私有 stdin 控制通道；取消只置位，模型操作始终在推理线程串行执行。"""

    def __init__(self, source):
        self.requests = queue.Queue()
        self.pending = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._read, args=(source,), daemon=True).start()

    def _read(self, source):
        try:
            for line in source:
                command = json.loads(line)
                request_id = command["request_id"]
                with self.lock:
                    if command.get("type") == "cancel":
                        if request_id in self.pending:
                            self.pending[request_id].set()
                    else:
                        if request_id in self.pending:
                            raise ValueError("Duplicate MOSS request")
                        cancelled = threading.Event()
                        self.pending[request_id] = cancelled
                        self.requests.put((command, cancelled))
        except Exception as exc:
            self.requests.put(exc)
        finally:
            with self.lock:
                for cancelled in self.pending.values():
                    cancelled.set()
            self.requests.put(None)

    def finish(self, request_id):
        with self.lock:
            self.pending.pop(request_id, None)


def stream_generate(llm, prompt, sampling_params, eos, send, request_id, cancelled=None):
    """vLLM 0.27.1 的 renderer/engine 增量路径；一次提交完整音频。

    LLM.generate/enqueue 会强制 FINAL_ONLY，不能用它们伪装流式输出。
    """
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError("MOSS engine still has an unfinished request")
    if cancelled is not None and cancelled.is_set():
        send({"type": "cancelled"})
        return
    (engine_input,) = llm.renderer.render_cmpl([prompt])
    internal_id = engine.add_request(request_id, engine_input, sampling_params)
    raw, last_token, final = "", None, None
    while engine.has_unfinished_requests():
        if cancelled is not None and cancelled.is_set():
            # vLLM 0.27.1：add_request 返回内部 id；同时移除 detokenizer 与 engine 请求。
            engine.abort_request([internal_id], internal=True)
            send({"type": "cancelled"})
            return
        for output in engine.step():
            if final is not None or len(output.outputs) != 1:
                raise RuntimeError("Unexpected MOSS engine output")
            result = output.outputs[0]
            if result.token_ids:
                last_token = result.token_ids[-1]
            if result.text:
                raw += result.text
                send({"type": "delta", "text": result.text})
            if output.finished:
                final = {
                    "type": "result", "raw": raw,
                    "finish_reason": result.finish_reason,
                    "eos_reached": last_token in eos,
                }
    if final is None:
        raise RuntimeError("MOSS engine ended without a final result")
    send(final)


def main():
    # 协议使用独立 FD；第三方初始化日志仍写标准错误，不污染结果也不保存录音/文本。
    protocol = os.fdopen(int(os.environ["MOSS_RESULT_FD"]), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(payload):
        protocol.write(json.dumps(payload, ensure_ascii=False) + "\n")

    try:
        import numpy as np
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if not torch.cuda.is_available():
            raise RuntimeError("MOSS worker 需要 CUDA；不在 CPU/MPS 上尝试完整长录音")
        torch.set_num_threads(int(os.environ.get("MOSS_CPU_THREADS", "4")))
        model = Path(os.environ["MOSS_MODEL_PATH"]).resolve()
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        generation = json.loads((model / "generation_config.json").read_text())
        eos = generation["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        # 与已验证的 GB10 whole-input 运行栈一致；chunked prefill 不是分块转写。
        llm = LLM(
            model=str(model), tokenizer=str(model), trust_remote_code=True,
            dtype="bfloat16", max_model_len=100000, max_num_seqs=1,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=float(os.environ.get("MOSS_GPU_MEMORY_UTILIZATION", "0.25")),
            enable_chunked_prefill=True, enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"audio": {"count": 1, "length": int(max_audio_seconds() * 16000)}},
            attention_backend="TRITON_ATTN", mm_encoder_attn_backend="TORCH_SDPA",
            async_scheduling=True, compilation_config={"cudagraph_capture_sizes": [1]}, seed=0,
        )
        commands = RequestCommands(sys.stdin)
        send({"ready": True})
        while True:
            item = commands.requests.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            request, cancelled = item
            request_id = request["request_id"]
            if cancelled.is_set():
                send({"type": "cancelled"})
                commands.finish(request_id)
                continue
            audio_path = request["audio_path"]
            with wave.open(audio_path, "rb") as audio_file:
                if (audio_file.getnchannels(), audio_file.getsampwidth(), audio_file.getframerate()) != (1, 2, 16000):
                    raise ValueError("MOSS 输入必须是 16kHz 单声道 PCM16 WAV")
                if audio_file.getnframes() > max_audio_seconds() * 16000:
                    raise ValueError("MOSS 音频超过完整上下文上限")
                pcm = audio_file.readframes(audio_file.getnframes())
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            prompt = processor.apply_chat_template([{
                "role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }], tokenize=False, add_generation_prompt=True)
            stream_generate(llm, {
                "prompt": prompt, "multi_modal_data": {"audio": (audio, 16000)},
            }, SamplingParams(
                temperature=0, max_tokens=65536, seed=0,
                repetition_penalty=1.0, stop_token_ids=eos,
                output_kind=RequestOutputKind.DELTA,
            ), eos, send, request_id, cancelled)
            commands.finish(request_id)
    except Exception:
        # 不把模型输入、临床文本或私有路径回传至 HTTP 错误响应。
        import traceback
        traceback.print_exc(file=sys.stderr)
        send({"error": "MOSS worker 推理失败，请检查本机运行日志；未回退旧模型"})
        sys.exit(1)


if __name__ == "__main__":
    main()
