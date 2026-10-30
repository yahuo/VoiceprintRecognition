#!/usr/bin/env python3
"""
KWS 关键词表：按类别维护的说法与命中值，持久化为 JSON，编译成 sherpa-onnx keywords 行。

说法含汉字时用 pypinyin 转成声韵母 token；不含汉字时视为已分好的模型 token（兼容旧 keywords 文件）。
每行编译为 `token... [:增益] [#阈值] @类别:值`；token 必须全部在模型词表中，否则整表拒绝，不生成残缺词表。
"""

from datetime import datetime
import json
import os
import re
import tempfile

CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
RAW_LINE_PATTERN = re.compile(r"^(?P<text>[^:#@]+?)\s*(?P<options>(?:\s*[:#][0-9.]+)*)\s*@(?P<code>[a-z][a-z0-9_]*):(?P<value>\S.*)$")
OPTION_PATTERN = re.compile(r"([:#])([0-9.]+)")
CJK_PATTERN = re.compile(r"[一-鿿]")
# 说法里的这些字符会被 keywords 行当成增益、阈值或标签。
PHRASE_FORBIDDEN = re.compile(r"[:#@/]")
MAX_NAME_LENGTH = 32
MAX_PHRASE_LENGTH = 32
MAX_VALUE_LENGTH = 64
MAX_KEYWORDS = 2000


class KeywordsError(ValueError):
    """关键词表校验失败；errors 逐条给出类别、说法和原因，供网页定位。"""

    def __init__(self, errors: list[dict]):
        super().__init__("；".join(e["message"] for e in errors))
        self.errors = errors


def load_vocabulary(tokens_path: str) -> set[str]:
    with open(tokens_path, encoding="utf-8") as f:
        return {line.split()[0] for line in f if line.strip()}


def to_tokens(text: str) -> list[str]:
    from pypinyin import Style, pinyin

    initials = pinyin(text, style=Style.INITIALS, strict=False)
    finals = pinyin(text, style=Style.FINALS_TONE, strict=False)
    return [token for ini, fin in zip(initials, finals) for token in (ini[0], fin[0]) if token]


def phrase_tokens(phrase: str, vocabulary: set[str]) -> tuple[list[str], str | None]:
    """返回 (token, 错误)；有错误时 token 无效。"""
    if not phrase:
        return [], "说法不能为空"
    if len(phrase) > MAX_PHRASE_LENGTH:
        return [], f"说法不能超过 {MAX_PHRASE_LENGTH} 个字符"
    if PHRASE_FORBIDDEN.search(phrase):
        return [], "说法不能包含 : # @ /"
    tokens = to_tokens(phrase.replace(" ", "")) if CJK_PATTERN.search(phrase) else phrase.split()
    missing = list(dict.fromkeys(token for token in tokens if token not in vocabulary))
    if missing:
        return [], f"发音 {' '.join(missing)} 不在模型词表中，请改用说出口的汉字写法"
    return tokens, None


def _number(raw, name: str, low: float, high: float) -> tuple[float | None, str | None]:
    if raw is None or raw == "":
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{name}须为数字"
    if not low < value <= high:
        return None, f"{name}须大于 {low:g} 且不超过 {high:g}"
    return value, None


def compile_categories(categories, vocabulary: set[str]) -> tuple[list[dict], list[str]]:
    """校验并规范化类别列表，返回 (规范化类别, keywords 行)；任何一处不合法都抛出 KeywordsError。"""
    if not isinstance(categories, list):
        raise KeywordsError([{"message": "categories 须为列表"}])
    errors, normalized, lines = [], [], []
    seen_codes, seen_phrases, seen_tokens = set(), {}, {}

    def fail(message, code=None, phrase=None):
        errors.append({"category": code, "phrase": phrase, "message": message})

    for category in categories:
        if not isinstance(category, dict):
            fail("类别格式不正确")
            continue
        code = str(category.get("code") or "").strip()
        if not CODE_PATTERN.match(code):
            fail(f"类别代码 {code or '(空)'} 须以小写字母开头，只含小写字母、数字和下划线，最多 32 个字符", code)
            continue
        if code in seen_codes:
            fail(f"类别代码 {code} 重复", code)
            continue
        seen_codes.add(code)
        name = str(category.get("name") or code).strip()[:MAX_NAME_LENGTH] or code
        keywords = []
        for keyword in category.get("keywords") or []:
            if not isinstance(keyword, dict):
                fail("关键词格式不正确", code)
                continue
            phrase = " ".join(str(keyword.get("phrase") or "").split())
            value = str(keyword.get("value") or phrase).strip()
            tokens, error = phrase_tokens(phrase, vocabulary)
            if error:
                fail(f"「{phrase}」{error}", code, phrase)
                continue
            if phrase in seen_phrases:
                fail(f"「{phrase}」在 {seen_phrases[phrase]} 和 {code} 里重复", code, phrase)
                continue
            key = " ".join(tokens)
            if key in seen_tokens:
                fail(f"「{phrase}」与「{seen_tokens[key]}」发音相同", code, phrase)
                continue
            if len(value) > MAX_VALUE_LENGTH or any(ch.isspace() for ch in value):
                fail(f"「{phrase}」的命中值须为 1-{MAX_VALUE_LENGTH} 个字符且不含空白", code, phrase)
                continue
            boost, boost_error = _number(keyword.get("boost"), "增益", 0, 10)
            threshold, threshold_error = _number(keyword.get("threshold"), "阈值", 0, 1)
            if boost_error or threshold_error:
                fail(f"「{phrase}」{boost_error or threshold_error}", code, phrase)
                continue
            seen_phrases[phrase] = code
            seen_tokens[key] = phrase
            options = (f" :{boost:g}" if boost is not None else "") + (f" #{threshold:g}" if threshold is not None else "")
            lines.append(f"{key}{options} @{code}:{value}")
            keywords.append({"phrase": phrase, "value": value, "boost": boost, "threshold": threshold, "tokens": key})
        normalized.append({"code": code, "name": name, "slot": bool(category.get("slot")), "keywords": keywords})

    if not errors and not lines:
        fail("至少保留一个关键词")
    if len(lines) > MAX_KEYWORDS:
        fail(f"关键词不能超过 {MAX_KEYWORDS} 个")
    if errors:
        raise KeywordsError(errors)
    return normalized, lines


def parse_raw_lines(raw_lines, context_slots) -> list[dict]:
    """把 keywords 文件（中文原始行或已分好的 token 行）转成类别列表，作为关键词表的初始内容。"""
    categories: dict[str, dict] = {code: {"code": code, "name": code, "slot": True, "keywords": []} for code in context_slots}
    errors = []
    for number, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = RAW_LINE_PATTERN.match(line)
        if not match:
            errors.append({"category": None, "phrase": None,
                           "message": f"第 {number} 行格式应为 `关键词 [:增益] [#阈值] @类别:值`: {line}"})
            continue
        options = dict((kind, value) for kind, value in OPTION_PATTERN.findall(match["options"]))
        code = match["code"]
        category = categories.setdefault(code, {"code": code, "name": code, "slot": code in context_slots, "keywords": []})
        category["keywords"].append({
            "phrase": " ".join(match["text"].split()),
            "value": match["value"].strip(),
            "boost": options.get(":"),
            "threshold": options.get("#"),
        })
    if errors:
        raise KeywordsError(errors)
    return list(categories.values())


def build(raw_lines, vocabulary: set[str]) -> tuple[list[str], list[str]]:
    """逐行编译 keywords 文件，返回 (输出行, 错误)；供离线脚本使用，错误行跳过但会报告。"""
    output, errors = [], []
    for number, line in enumerate(raw_lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            _, lines = compile_categories(parse_raw_lines([stripped], ()), vocabulary)
        except KeywordsError as exc:
            errors.append(f"第 {number} 行 {exc}: {stripped}")
            continue
        output.extend(lines)
    return output, errors


class KeywordStore:
    """关键词表 JSON 文件；整文件原子替换，进程崩溃时不会留下半截内容。"""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, encoding="utf-8") as f:
            document = json.load(f)
        if not isinstance(document, dict) or not isinstance(document.get("categories"), list):
            raise RuntimeError(f"关键词库格式不正确: {self.path}")
        return document

    def save(self, document: dict) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".kws-keywords-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def stored_document(categories: list[dict], version: int) -> dict:
    """持久化内容：不含编译出的 token，token 由当前模型词表重新生成。"""
    return {
        "version": version,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "categories": [
            {**{k: c[k] for k in ("code", "name", "slot")},
             "keywords": [{k: kw[k] for k in ("phrase", "value", "boost", "threshold")} for kw in c["keywords"]]}
            for c in categories
        ],
    }
