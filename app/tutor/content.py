"""Load a tutorial repository: tutorial.json -> courses -> lessons -> exercises.

Repository layout (same for every tutorial repo that uses this engine)::

    tutorial.json                       repo name, locales, course order
    courses/<course>/course.toml        title, summary, programming language
    courses/<course>/<NN-lesson>/zh-CN.md   (and zh-TW.md, en.md, ru.md ...)
    courses/<course>/<NN-lesson>/exercises.toml  tests + reference solutions

Exercise *text* lives in each language's Markdown (so GitHub readers see it);
tests and solutions live once in exercises.toml (so they can't drift).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CONTENT_LOCALES = ["zh-CN", "zh-TW", "en", "ru"]
EXPLAIN_RE = re.compile(r"答案|解析|解答|answer|explanation|solution|ответ|разбор|решение", re.I)
EX_MARK = re.compile(r"^\s*<!--\s*exercise:\s*([\w-]+)\s*-->\s*$")


def fallback_chain(locale: str) -> list[str]:
    chains = {
        "zh-CN": ["zh-CN", "zh-TW", "en"],
        "zh-TW": ["zh-TW", "zh-CN", "en"],
        "en": ["en", "zh-CN"],
        "ru": ["ru", "en", "zh-CN"],
    }
    if locale in chains:
        return chains[locale]
    return [locale, "en", "zh-CN"]


def pick(table, locale: str, default=""):
    """Pick a localized value from a {locale: value} table (or return plain values)."""
    if not isinstance(table, dict):
        return table if table is not None else default
    for loc in fallback_chain(locale):
        if table.get(loc):
            return table[loc]
    for value in table.values():
        if value:
            return value
    return default


@dataclass
class Exercise:
    id: str
    kind: str                      # code | choice | output
    title: str = ""
    prompt_md: str = ""
    hints_md: list[str] = field(default_factory=list)
    explain_md: str = ""
    options_md: list[str] = field(default_factory=list)
    starter: str = ""
    solution: str = ""
    tests: list[dict] = field(default_factory=list)
    answer: object = None
    language: str = ""
    match: str = "exact"
    requires: list[str] = field(default_factory=list)
    code_md: str = ""             # for "output" exercises: the code to read


@dataclass
class Lesson:
    course: str
    id: str
    path: Path
    locale: str                    # locale actually used
    title: str
    body_md: str                   # lesson text without the exercise blocks
    exercises: list[Exercise]
    language: str

    @property
    def key(self) -> str:
        return f"{self.course}/{self.id}"

    @property
    def number(self) -> str:
        m = re.match(r"(\d+)", self.id)
        return m.group(1) if m else ""


@dataclass
class Course:
    id: str
    path: Path
    meta: dict
    lesson_ids: list[str]

    def title(self, locale: str) -> str:
        return pick(self.meta.get("title"), locale, self.id)

    def summary(self, locale: str) -> str:
        return pick(self.meta.get("summary"), locale, "")

    @property
    def language(self) -> str:
        return self.meta.get("language", "text")

    @property
    def icon(self) -> str:
        return self.meta.get("icon", "📘")


class Library:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest_path = self.root / "tutorial.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.courses: dict[str, Course] = {}
        order = self.manifest.get("courses") or sorted(p.name for p in (self.root / "courses").iterdir() if p.is_dir())
        for cid in order:
            cpath = self.root / "courses" / cid
            meta_path = cpath / "course.toml"
            if not meta_path.exists():
                continue
            meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))
            lessons = sorted(p.name for p in cpath.iterdir() if p.is_dir() and re.match(r"\d", p.name)
                             and any(p.glob("*.md")))
            self.courses[cid] = Course(cid, cpath, meta, lessons)
        self._title_cache: dict[tuple[str, str, str], str] = {}

    # ---------------------------------------------------------------- meta
    def name(self, locale: str) -> str:
        return pick(self.manifest.get("name"), locale, self.root.name)

    def description(self, locale: str) -> str:
        return pick(self.manifest.get("description"), locale, "")

    @property
    def repo(self) -> str:
        return self.manifest.get("repo", "")

    def lesson_dir(self, course: str, lesson: str) -> Path:
        return self.courses[course].path / lesson

    def find_md(self, course: str, lesson: str, locale: str) -> tuple[Path, str] | None:
        d = self.lesson_dir(course, lesson)
        for loc in fallback_chain(locale):
            p = d / f"{loc}.md"
            if p.exists():
                return p, loc
        files = sorted(d.glob("*.md"))
        return (files[0], files[0].stem) if files else None

    def lesson_title(self, course: str, lesson: str, locale: str) -> str:
        key = (course, lesson, locale)
        if key not in self._title_cache:
            found = self.find_md(course, lesson, locale)
            title = lesson
            if found:
                with found[0].open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break
            self._title_cache[key] = title
        return self._title_cache[key]

    def exercise_data(self, course: str, lesson: str) -> dict[str, dict]:
        path = self.lesson_dir(course, lesson) / "exercises.toml"
        if not path.exists():
            return {}
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return {ex["id"]: ex for ex in data.get("exercise", [])}

    # --------------------------------------------------------------- lesson
    def lesson(self, course: str, lesson: str, locale: str) -> Lesson:
        found = self.find_md(course, lesson, locale)
        if not found:
            raise FileNotFoundError(f"{course}/{lesson}")
        path, used = found
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        title, body, segments = split_lesson(text)
        data = self.exercise_data(course, lesson)
        lang = self.courses[course].language
        exercises = []
        for ex_id, seg in segments:
            meta = data.get(ex_id, {"id": ex_id, "kind": "choice"})
            exercises.append(build_exercise(ex_id, seg, meta, used, lang))
        return Lesson(course, lesson, self.lesson_dir(course, lesson), used, title or lesson, body, exercises, lang)

    def neighbours(self, course: str, lesson: str) -> tuple[str | None, str | None]:
        ids = self.courses[course].lesson_ids
        i = ids.index(lesson)
        return (ids[i - 1] if i > 0 else None, ids[i + 1] if i + 1 < len(ids) else None)

    def resolve_link(self, base: Path, href: str) -> tuple[str, str, str] | None:
        """Map a relative link like ../02-variables/en.md to (course, lesson, anchor)."""
        if re.match(r"^[a-z]+:", href) or href.startswith("#"):
            return None
        target, _, anchor = href.partition("#")
        try:
            p = (base / target).resolve()
        except OSError:
            return None
        if p.suffix == ".md":
            p = p.parent
        try:
            rel = p.relative_to((self.root / "courses").resolve())
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) == 1 and parts[0] in self.courses:
            ids = self.courses[parts[0]].lesson_ids
            return (parts[0], ids[0], anchor) if ids else None
        if len(parts) == 2 and parts[0] in self.courses and parts[1] in self.courses[parts[0]].lesson_ids:
            return parts[0], parts[1], anchor
        return None


def split_lesson(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, body_without_exercises, [(exercise_id, markdown_segment)])."""
    lines = text.split("\n")
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    body: list[str] = []
    segments: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    in_fence = False
    for line in lines:
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
        m = EX_MARK.match(line) if not in_fence else None
        if m:
            current = []
            segments.append((m.group(1), current))
            continue
        if current is not None and not in_fence and (re.match(r"^#{1,2}\s", line) or re.match(r"^\s*<!--\s*/exercises\s*-->", line)):
            current = None
        if current is not None:
            current.append(line)
        else:
            body.append(line)
    return title, "\n".join(body).strip() + "\n", [(i, "\n".join(s).strip()) for i, s in segments]


def build_exercise(ex_id: str, segment: str, meta: dict, locale: str, course_lang: str) -> Exercise:
    lines = segment.split("\n")
    title = ""
    if lines and re.match(r"^#{3,6}\s", lines[0]):
        title = re.sub(r"^#{3,6}\s+", "", lines[0]).strip()
        lines = lines[1:]
    text = "\n".join(lines)
    hints, explain = [], ""
    def take_details(m):
        nonlocal explain
        summary, inner = m.group(1).strip(), m.group(2).strip()
        if EXPLAIN_RE.search(summary):
            explain = inner
        else:
            hints.append(inner)
        return ""
    text = re.sub(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", take_details, text, flags=re.S | re.I)
    kind = meta.get("kind", "code")
    options: list[str] = []
    code_md = ""
    if kind == "choice":
        # the last top-level bullet list holds the options
        body_lines = text.rstrip().split("\n")
        end = len(body_lines)
        while end > 0 and not body_lines[end - 1].strip():
            end -= 1
        start = end
        while start > 0 and (re.match(r"^[-*+]\s+", body_lines[start - 1]) or
                             (body_lines[start - 1].startswith("  ") and start - 1 > 0)):
            start -= 1
        opt_lines = body_lines[start:end]
        for ln in opt_lines:
            if re.match(r"^[-*+]\s+", ln):
                options.append(re.sub(r"^[-*+]\s+", "", ln).strip())
            elif options:
                options[-1] += " " + ln.strip()
        text = "\n".join(body_lines[:start])
    if kind == "output":
        m = re.search(r"(```[^\n]*\n.*?\n```)", text, flags=re.S)
        if m:
            code_md = m.group(1)
    starter = pick(meta.get("starter", ""), locale, "")
    return Exercise(
        id=ex_id, kind=kind, title=title or ex_id, prompt_md=text.strip(), hints_md=hints,
        explain_md=explain, options_md=options, starter=starter,
        solution=pick(meta.get("solution", ""), locale, ""),
        tests=list(pick(meta.get("tests", meta.get("test", [])), locale, []) or []),
        answer=pick(meta.get("answer"), locale, None) if isinstance(meta.get("answer"), dict) else meta.get("answer"),
        language=meta.get("language", course_lang), match=meta.get("match", "exact"),
        requires=list(meta.get("requires", [])), code_md=code_md,
    )


def normalize_output(text: str) -> str:
    """Compare outputs fairly: unify line endings, drop trailing spaces and blank lines."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def output_matches(actual: str, expected: str, mode: str = "exact") -> bool:
    a, e = normalize_output(actual), normalize_output(expected)
    if mode == "contains":
        return e in a
    if mode == "unordered":
        return sorted(a.split("\n")) == sorted(e.split("\n"))
    if mode == "number":
        try:
            return all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a.split(), e.split(), strict=True))
        except (ValueError, TypeError):
            return False
    return a == e


def check_output_answer(user: str, expected: str) -> bool:
    return normalize_output(user).strip() == normalize_output(str(expected)).strip()
