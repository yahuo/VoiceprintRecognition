#!/usr/bin/env bash
# 只创建新的独立环境，不修改已有服务 venv；不会下载模型或启动推理。
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:?用法: bash scripts/setup_moss.sh /absolute/path/to/new/venv_moss}
python=${MOSS_SETUP_PYTHON:-python3.11}
if [[ "$target" != /* || -e "$target" ]]; then
    echo '目标必须是尚不存在的绝对路径；拒绝覆盖已有环境。' >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    echo 'MOSS vLLM 环境需在目标 Linux/CUDA 主机或同 ABI 容器中创建。' >&2
    exit 2
fi
"$python" -m venv "$target"
"$target/bin/python" -m pip install --disable-pip-version-check uv==0.9.28
"$target/bin/uv" pip install --python "$target/bin/python" -r "$root/requirements-moss.txt"
# vLLM 声明的 post3 在 Python 3.11 有 array.array[int] 导入问题。
# 使用已验证的上游 post4，不修改 site-packages 源码，不开启 unsafe pickle。
"$target/bin/uv" pip install --python "$target/bin/python" --no-deps flashinfer-python==0.6.16.post4
"$target/bin/python" - <<'PY'
import importlib.metadata as metadata
from vllm.model_executor.models import ModelRegistry
assert 'MossTranscribeDiarizeForConditionalGeneration' in ModelRegistry.get_supported_archs()
print({name: metadata.version(name) for name in ['torch', 'vllm', 'transformers', 'flashinfer-python']})
PY
printf 'MOSS_PYTHON=%s/bin/python\n' "$target"
