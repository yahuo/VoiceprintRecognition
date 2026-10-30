#!/usr/bin/env python3
"""
生成 sherpa-onnx KWS keywords 文件（中文拼音声韵母 token）。

输入每行: 中文关键词 [:增益] [#阈值] @类别:值，空行和 # 开头的行忽略，例如:
    十二床 @bed:12床
    下一床 #0.3 @command:next_bed
    开始查体 :1.5 @section:查体

`类别:值` 原样作为命中标签返回；类别在 KWS_CONTEXT_SLOTS 中时是上下文槽位，否则只推送事件。
token 必须全部存在于模型 tokens.txt，缺失时报错退出，不生成残缺词表。

使用方法（需要 pip install pypinyin）:
    python scripts/build_kws_keywords.py --tokens /models/kws/tokens.txt keywords_raw.txt keywords.txt
"""

import argparse
import re
import sys

LINE_PATTERN = re.compile(r"^(?P<text>[^:#@]+?)\s*(?P<options>(?:\s*[:#][0-9.]+)*)\s*@(?P<label>[a-z][a-z0-9_]*:\S.*)$")


def to_tokens(text, pinyin, style):
    initials = pinyin(text, style=style.INITIALS, strict=False)
    finals = pinyin(text, style=style.FINALS_TONE, strict=False)
    return [token for ini, fin in zip(initials, finals) for token in (ini[0], fin[0]) if token]


def build(raw_lines, vocabulary):
    from pypinyin import Style, pinyin

    output, errors = [], []
    for number, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            errors.append(f"第 {number} 行格式应为 `关键词 [:增益] [#阈值] @类别:值`: {line}")
            continue
        tokens = to_tokens(match["text"].replace(" ", ""), pinyin, Style)
        missing = [token for token in tokens if token not in vocabulary]
        if missing:
            errors.append(f"第 {number} 行 token 不在模型词表中 {missing}: {line}")
            continue
        options = " ".join(match["options"].split())
        output.append(" ".join(filter(None, [" ".join(tokens), options, "@" + match["label"].strip()])))
    return output, errors


def main():
    parser = argparse.ArgumentParser(description="生成 sherpa-onnx KWS keywords 文件")
    parser.add_argument("--tokens", required=True, help="模型 tokens.txt")
    parser.add_argument("input", help="原始关键词文件")
    parser.add_argument("output", help="输出 keywords 文件")
    args = parser.parse_args()

    with open(args.tokens, encoding="utf-8") as f:
        vocabulary = {line.split()[0] for line in f if line.strip()}
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
