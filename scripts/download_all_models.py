#!/usr/bin/env python3
"""
下载业务运行模型到 models/ 目录：
- ASR: FunAudioLLM/Fun-ASR-Nano-2512 (Hugging Face)
- VAD: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch (ModelScope)
- SPK: iic/speech_campplus_sv_zh-cn_16k-common (ModelScope)

说明：
- 默认下载 Nano / 流式 Paraformer / VAD / CAM++；--include-moss 下载固定版本的 MOSS。
- 不再下载或使用 Pyannote、离线 Paraformer 和独立标点模型。
- 已存在的目标目录会自动跳过，避免重复下载。

用法:
    python scripts/download_all_models.py
    python scripts/download_all_models.py --include-moss
    python scripts/download_all_models.py --output-dir /data/voiceprint/models --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str  # "hf" | "modelscope"
    repo_id: str
    subdir: str
    marker_file: str | None = None
    revision: str | None = None


MODEL_SPECS = [
    ModelSpec(
        name="ASR",
        source="hf",
        repo_id="FunAudioLLM/Fun-ASR-Nano-2512",
        subdir="asr/Fun-ASR-Nano-2512",
    ),
    ModelSpec(
        name="VAD",
        source="modelscope",
        repo_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        subdir="vad/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
    ModelSpec(
        name="SPK",
        source="modelscope",
        repo_id="iic/speech_campplus_sv_zh-cn_16k-common",
        subdir="spk/speech_campplus_sv_zh-cn_16k-common",
    ),
]

MODEL_SPECS.append(ModelSpec(
    name="STREAMING_ASR", source="modelscope",
    repo_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    subdir="asr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
))

MOSS_SPEC = ModelSpec(
    name="MOSS", source="hf", repo_id="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    subdir="moss/MOSS-Transcribe-Diarize",
    revision="704aa4a9c304e8520be88901e0d1960158ef5b15",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载离线模型到 models/ 目录")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="模型输出目录（默认: <project_root>/models）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（会删除目标模型目录后重下）",
    )
    parser.add_argument(
        "--include-moss", "--include-upload-asr",
        action="store_true",
        help="额外下载固定版本的 MOSS（旧 --include-upload-asr 仅为弃用别名）",
    )
    return parser.parse_args()


def has_any_file(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() and p.suffix in {".bin", ".safetensors", ".pt", ".pth", ".pb", ".onnx"}
               for p in path.rglob("*"))


def is_model_ready(path: Path, marker_file: str | None) -> bool:
    if marker_file:
        return (path / marker_file).exists()
    return has_any_file(path)


def clean_target(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_hf(repo_id: str, local_dir: Path, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=revision,
    )


def download_modelscope(repo_id: str, local_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id=repo_id,
        local_dir=str(local_dir),
    )


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (project_root / "models")

    print(f"📦 模型输出目录: {output_dir}")
    print("ℹ️ 离线识别仅使用 MOSS；不下载旧离线模型。")
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    model_specs = list(MODEL_SPECS)
    if args.include_moss:
        model_specs.append(MOSS_SPEC)

    for spec in model_specs:
        target = output_dir / spec.subdir
        print(f"\n==> [{spec.name}] {spec.repo_id}")
        print(f"    目标路径: {target}")

        complete = target / ".download-complete"
        signature = spec.repo_id + "@" + (spec.revision or "default")
        if args.force:
            print("    force 模式: 清理后重新下载")
            clean_target(target)
        elif complete.is_file() and complete.read_text() == signature and is_model_ready(target, spec.marker_file):
            print("    已存在，跳过")
            continue
        else:
            target.mkdir(parents=True, exist_ok=True)

        try:
            if spec.source == "hf":
                download_hf(spec.repo_id, target, spec.revision)
            else:
                download_modelscope(spec.repo_id, target)

            if not is_model_ready(target, spec.marker_file):
                raise RuntimeError("下载完成但未检测到有效模型文件")

            complete.write_text(signature)
            print("✅ 完成")
        except Exception as exc:
            msg = f"{spec.name}: 下载失败 - {exc}"
            print(f"❌ {msg}")
            failures.append(msg)

    print("\n==============================")
    if failures:
        print("部分模型下载失败：")
        for item in failures:
            print(f"- {item}")
        return 1

    print("✅ 全部模型下载完成")
    print("\n可用于 .env 的本地模型路径示例：")
    print(f"ASR_MODEL_PATH={output_dir / 'asr/Fun-ASR-Nano-2512'}")
    print(f"STREAMING_ASR_MODEL_PATH={output_dir / MODEL_SPECS[-1].subdir}")
    if args.include_moss:
        print(f"MOSS_MODEL_PATH={output_dir / MOSS_SPEC.subdir}")
    print(f"VAD_MODEL_PATH={output_dir / 'vad/speech_fsmn_vad_zh-cn-16k-common-pytorch'}")
    print(f"SPK_MODEL_PATH={output_dir / 'spk/speech_campplus_sv_zh-cn_16k-common'}")
    print(f"MODELS_PATH={output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
