#!/bin/bash

set -e

# 使用环境变量（如果有）或设置默认值
DEVICE="${DEVICE:-cpu}"        # 如果环境变量 DEVICE 没有设置，则默认为 cpu
HOST="${HOST:-0.0.0.0}"        # 如果环境变量 HOST 没有设置，则默认为 0.0.0.0
PORT="${PORT:-8000}"           # 如果环境变量 PORT 没有设置，则默认为 8000

# 执行 Python 程序，并传递参数
exec python -m app.server --device "$DEVICE" --host "$HOST" --port "$PORT"