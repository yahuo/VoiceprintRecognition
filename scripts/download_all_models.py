#!/usr/bin/env python3
"""
统一下载离线模型到 models/ 目录：
- ASR: FunAudioLLM/Fun-ASR-Nano-2512 (Hugging Face)
- VAD: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch (ModelScope)
- SPK: iic/speech_campplus_sv_zh-cn_16k-common (ModelScope)

说明：
- 本脚本不会处理 models/pyannote（仓库已内置）。
- 已存在的目标目录会自动跳过，避免重复下载。

用法:
    python scripts/download_all_models.py
    python scripts/download_all_models.py --include-upload-asr
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

UPLOAD_ASR_SPEC = ModelSpec(
    name="UPLOAD_ASR",
    source="modelscope",
    repo_id="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    subdir="asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)

UPLOAD_PUNC_SPEC = ModelSpec(
    name="UPLOAD_PUNC",
    source="modelscope",
    repo_id="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    subdir="punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
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
        "--include-upload-asr",
        action="store_true",
        help="额外下载上传链路用的 Paraformer 模型",
    )
    return parser.parse_args()


def has_any_file(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() for p in path.rglob("*"))


def is_model_ready(path: Path, marker_file: str | None) -> bool:
    if marker_file:
        return (path / marker_file).exists()
    return has_any_file(path)


def clean_target(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_hf(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
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
    print("ℹ️ 跳过 pyannote（默认使用仓库中的 models/pyannote）")
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    model_specs = list(MODEL_SPECS)
    if args.include_upload_asr:
        model_specs.append(UPLOAD_ASR_SPEC)
        model_specs.append(UPLOAD_PUNC_SPEC)

    for spec in model_specs:
        target = output_dir / spec.subdir
        print(f"\n==> [{spec.name}] {spec.repo_id}")
        print(f"    目标路径: {target}")

        if args.force:
            print("    force 模式: 清理后重新下载")
            clean_target(target)
        elif is_model_ready(target, spec.marker_file):
            print("    已存在，跳过")
            continue
        else:
            target.mkdir(parents=True, exist_ok=True)

        try:
            if spec.source == "hf":
                download_hf(spec.repo_id, target)
            else:
                download_modelscope(spec.repo_id, target)

            if not is_model_ready(target, spec.marker_file):
                raise RuntimeError("下载完成但未检测到有效模型文件")

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
    if args.include_upload_asr:
        print("UPLOAD_ASR_BACKEND=paraformer")
        print(
            "UPLOAD_ASR_MODEL_PATH="
            f"{output_dir / 'asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'}"
        )
        print(
            "PUNC_MODEL_PATH="
            f"{output_dir / 'punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'}"
        )
    print(f"VAD_MODEL_PATH={output_dir / 'vad/speech_fsmn_vad_zh-cn-16k-common-pytorch'}")
    print(f"SPK_MODEL_PATH={output_dir / 'spk/speech_campplus_sv_zh-cn_16k-common'}")
    print(f"MODELS_PATH={output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
