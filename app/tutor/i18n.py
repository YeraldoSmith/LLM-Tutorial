"""Interface translations (locales/*.json). Missing keys fall back to English."""
from __future__ import annotations

import json
import locale as _locale
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / "locales"
ORDER = ["zh-CN", "zh-TW", "en", "ru", "ja", "ko", "es", "fr", "de"]


def available() -> list[tuple[str, str]]:
    out = []
    for code in ORDER:
        p = HERE / f"{code}.json"
        if p.exists():
            out.append((code, json.loads(p.read_text(encoding="utf-8")).get("_name", code)))
    return out


def detect_system() -> str:
    candidates = [os.environ.get(k, "") for k in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")]
    try:
        candidates.append(_locale.getlocale()[0] or "")
    except ValueError:
        pass
    for raw in candidates:
        tag = raw.split(":")[0].split(".")[0].replace("_", "-").lower()
        if not tag or tag in ("c", "posix"):
            continue
        if re.match(r"zh-(tw|hk|mo)|zh-hant", tag) or "chinese (traditional)" in tag:
            return "zh-TW"
        if tag.startswith("zh") or tag.startswith("chinese"):
            return "zh-CN"
        base = tag.split("-")[0]
        for code in ORDER:
            if code.split("-")[0] == base:
                return code
    return "en"


class Translator:
    def __init__(self, code: str):
        self.code = code if (HERE / f"{code}.json").exists() else "en"
        self.table = json.loads((HERE / f"{self.code}.json").read_text(encoding="utf-8"))
        self.fallback = json.loads((HERE / "en.json").read_text(encoding="utf-8")) if self.code != "en" else {}
        if self.code == "zh-TW":
            self.fallback = {**self.fallback, **json.loads((HERE / "zh-CN.json").read_text(encoding="utf-8"))}

    def get(self, key: str, **kw) -> str | None:
        text = self.table.get(key)
        if text is None:
            text = self.fallback.get(key)
        if text is None:
            return None
        text = text.replace("{{", "\x01").replace("}}", "\x02")
        text = re.sub(r"\{(\w+)\}", lambda m: str(kw.get(m.group(1), "")), text)
        return text.replace("\x01", "{").replace("\x02", "}")

    def __call__(self, key: str, **kw) -> str:
        text = self.get(key, **kw)
        return key if text is None else text

    @property
    def content_locale(self) -> str:
        """Which lesson language to show for this interface language."""
        return self.code if self.code in ("zh-CN", "zh-TW", "en", "ru") else "en"
