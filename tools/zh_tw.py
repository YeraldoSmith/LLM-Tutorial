#!/usr/bin/env python3
"""Generate Traditional Chinese (zh-TW) from Simplified Chinese (zh-CN).

Uses OpenCC's "s2twp" profile (Taiwan wording, e.g. 程序→程式, 软件→軟體,
内存→記憶體) plus a small glossary of programming terms.  Run after editing
any zh-CN text:

    pip install opencc-python-reimplemented
    python tools/zh_tw.py            # all courses + app strings
    python tools/zh_tw.py --check    # fail if zh-TW is out of date (CI)

Hand-written zh-TW files are kept if their first line contains
"<!-- zh-TW: manual -->".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Applied after OpenCC: terms OpenCC doesn't know or gets wrong in a programming context.
GLOSSARY = [
    ("運行", "執行"), ("源代碼", "原始碼"), ("代碼", "程式碼"), ("程式碼塊", "程式碼區塊"),
    ("命令行", "命令列"), ("調試", "除錯"), ("數組", "陣列"), ("函數", "函式"), ("棧", "堆疊"),
    ("鏈表", "鏈結串列"), ("隊列", "佇列"), ("哈希", "雜湊"), ("字節", "位元組"), ("比特", "位元"),
    ("示例", "範例"), ("模塊", "模組"), ("文本", "文字"), ("調用", "呼叫"), ("迭代器", "迭代器"),
    ("打印", "印出"), ("編程", "程式設計"), ("屏幕", "螢幕"), ("鼠標", "滑鼠"), ("界面", "介面"),
    ("默認", "預設"), ("高級", "進階"), ("文件夾", "資料夾"), ("項目", "專案"), ("實例", "實例"),
    ("線程", "執行緒"), ("進程", "行程"), ("緩存", "快取"), ("服務器", "伺服器"), ("循環", "迴圈"),
    ("遞歸", "遞迴"), ("指針", "指標"), ("字符串", "字串"), ("字符", "字元"), ("布爾", "布林"),
    ("內存", "記憶體"), ("硬盤", "硬碟"), ("用戶", "使用者"), ("登錄", "登入"), ("卸載", "解除安裝"),
    ("視頻", "影片"), ("信息", "資訊"), ("網絡", "網路"), ("質量", "品質"), ("支持", "支援"),
]
MANUAL = "<!-- zh-TW: manual -->"


def converter():
    try:
        import opencc  # opencc-python-reimplemented
    except ImportError:
        sys.exit("Please run: pip install opencc-python-reimplemented")
    cc = opencc.OpenCC("s2twp")

    def convert(text: str) -> str:
        out = cc.convert(text)
        for a, b in GLOSSARY:
            out = out.replace(a, b)
        return out
    return convert


def toml_add_zh_tw(text: str, convert) -> str:
    """Insert `zh-TW = ...` after every `zh-CN = ...` entry whose table lacks one."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)zh-CN\s*=\s*('''|\"\"\"|'|\"|\[)(.*)$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent, q, rest = m.groups()
        block = [line]
        if q == "[":
            pass
        elif len(q) == 3 and not (rest.endswith(q) and len(rest) >= 3 and rest != q[:0]):
            if not rest.endswith(q) or rest == "":
                i += 1
                while i < len(lines):
                    block.append(lines[i])
                    if lines[i].rstrip().endswith(q):
                        break
                    i += 1
        i += 1
        out.extend(block)
        # does this table already have zh-TW?
        j, has = i, False
        while j < len(lines) and not lines[j].lstrip().startswith("["):
            if re.match(r"^\s*zh-TW\s*=", lines[j]):
                has = True
                break
            j += 1
        if not has:
            tw = "\n".join(block)
            tw = re.sub(r"^(\s*)zh-CN", r"\1zh-TW", tw, count=1)
            head, sep, body = tw.partition("=")
            out.extend((head + sep + convert(body)).split("\n"))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report files that would change")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    convert = converter()
    changed: list[Path] = []

    def write(path: Path, text: str):
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != text:
            changed.append(path)
            if not args.check:
                path.write_text(text, encoding="utf-8")

    for src in sorted(root.glob("courses/**/zh-CN.md")):
        dst = src.with_name("zh-TW.md")
        if dst.exists() and dst.read_text(encoding="utf-8").startswith(MANUAL):
            continue
        write(dst, convert(src.read_text(encoding="utf-8")))
    for toml in sorted(list(root.glob("courses/**/exercises.toml")) + list(root.glob("courses/*/course.toml"))):
        text = toml.read_text(encoding="utf-8")
        write(toml, toml_add_zh_tw(text, convert))
    manifest = root / "tutorial.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("name", "description"):
            if isinstance(data.get(key), dict) and "zh-CN" in data[key]:
                data[key]["zh-TW"] = convert(data[key]["zh-CN"])
        write(manifest, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    loc = root / "app" / "tutor" / "locales"
    if (loc / "zh-CN.json").exists():
        data = json.loads((loc / "zh-CN.json").read_text(encoding="utf-8"))
        tw = {k: (convert(v) if k != "_name" else "繁體中文") for k, v in data.items()}
        write(loc / "zh-TW.json", json.dumps(tw, ensure_ascii=False, indent=2) + "\n")
    readme = root / "README.md"
    if readme.exists():
        text = convert(readme.read_text(encoding="utf-8"))
        text = text.replace("**簡體中文** · [繁體中文](README.zh-TW.md)", "[简体中文](README.md) · **繁體中文**")
        write(root / "README.zh-TW.md", text)
    for p in changed:
        print(("would update " if args.check else "updated ") + str(p.relative_to(root)))
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
