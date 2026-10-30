#!/usr/bin/env python3
"""
生成 sherpa-onnx KWS keywords 文件（中文拼音声韵母 token）。

输入每行: 中文关键词 [:增益] [#阈值] @类别:值，空行和 # 开头的行忽略，例如:
    十二床 @bed:12床
    下一床 #0.3 @command:next_bed
    开始查体 :1.5 @section:查体

`类别:值` 原样作为命中标签返回；类别在 KWS_CONTEXT_SLOTS 中时是上下文槽位，否则只推送事件。
token 必须全部存在于模型 tokens.txt，缺失时报错退出，不生成残缺词表。
服务启用 KWS_KEYWORDS_STORE 后关键词在网页维护，本脚本只用于离线检查或生成独立的 keywords 文件。

使用方法（需要 pip install pypinyin）:
    python scripts/build_kws_keywords.py --tokens /models/kws/tokens.txt keywords_raw.txt keywords.txt
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.kws_keywords import build, load_vocabulary  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="生成 sherpa-onnx KWS keywords 文件")
    parser.add_argument("--tokens", required=True, help="模型 tokens.txt")
    parser.add_argument("input", help="原始关键词文件")
    parser.add_argument("output", help="输出 keywords 文件")
    args = parser.parse_args()

    vocabulary = load_vocabulary(args.tokens)
    with open(args.input, encoding="utf-8") as f:
        output, errors = build(f, vocabulary)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")
    print(f"已生成 {len(output)} 个关键词: {args.output}")


if __name__ == "__main__":
    main()
