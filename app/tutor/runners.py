"""Find compilers/interpreters on this computer and run learners' programs.

No Qt here, so the command-line checker (tools/check.py) can reuse it.
The app turns a Plan into QProcess calls (interactive console); tests use
``run_tests`` which runs each test case with subprocess and a timeout.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .content import output_matches

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
FROZEN = getattr(sys, "frozen", False)
EXE = ".exe" if IS_WINDOWS else ""

LANGS = {
    "python": {"name": "Python", "file": "main.py"},
    "c": {"name": "C", "file": "main.c"},
    "cpp": {"name": "C++", "file": "main.cpp"},
    "java": {"name": "Java", "file": "Main.java"},
    "rust": {"name": "Rust", "file": "main.rs"},
    "go": {"name": "Go", "file": "main.go"},
    "javascript": {"name": "JavaScript", "file": "main.js"},
    "typescript": {"name": "TypeScript", "file": "main.ts"},
    "asm": {"name": "Assembly (x86-64)", "file": "main.s"},
}

# One-line install commands shown in the "toolchain doctor". Detailed, illustrated
# steps are in the course "Start here -> Installing tools".
INSTALL = {
    "python": {"windows": ["winget install -e --id Python.Python.3.13"],
               "macos": ["brew install python"],
               "linux": ["sudo apt install python3 python3-pip   # Debian / Ubuntu",
                         "sudo dnf install python3               # Fedora",
                         "sudo pacman -S python                  # Arch"]},
    "c": {"windows": ["winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT"],
          "macos": ["xcode-select --install"],
          "linux": ["sudo apt install build-essential   # Debian / Ubuntu",
                    "sudo dnf install gcc gcc-c++        # Fedora",
                    "sudo pacman -S base-devel           # Arch"]},
    "java": {"windows": ["winget install -e --id EclipseAdoptium.Temurin.25.JDK"],
             "macos": ["brew install --cask temurin"],
             "linux": ["sudo apt install openjdk-25-jdk   # Ubuntu 24.04+ / Debian 13+",
                       "sudo dnf install java-25-openjdk-devel   # Fedora"]},
    "rust": {"windows": ["winget install -e --id Rustlang.Rustup"],
             "macos": ["curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"],
             "linux": ["curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"]},
    "go": {"windows": ["winget install -e --id GoLang.Go"],
           "macos": ["brew install go"],
           "linux": ["sudo snap install go --classic   # or download from https://go.dev/dl/"]},
    "javascript": {"windows": ["winget install -e --id OpenJS.NodeJS.LTS"],
                   "macos": ["brew install node"],
                   "linux": ["curl -fsSL https://fnm.vercel.app/install | bash",
                             "fnm install --lts"]},
    "asm": {"windows": ["wsl --install", "# then, inside Ubuntu:  sudo apt install build-essential"],
            "macos": ["# Apple computers use ARM; run x86-64 lessons with the online runner,",
                      "# or in a Linux virtual machine (UTM / OrbStack / Docker)."],
            "linux": ["sudo apt install build-essential"]},
}
INSTALL["cpp"] = INSTALL["c"]
INSTALL["typescript"] = INSTALL["javascript"]


def os_key() -> str:
    return "windows" if IS_WINDOWS else "macos" if IS_MAC else "linux"


def popen_kwargs() -> dict:
    kw: dict = {}
    if IS_WINDOWS:
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW: no black console windows
    return kw


def child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1", "GOTOOLCHAIN": "local", "NO_COLOR": "1",
                "CARGO_TERM_COLOR": "never"})
    # Compiler messages in English (the error explanations match English text); keep the
    # user's character set so programs can still print UTF-8.
    if env.get("LC_ALL"):
        env["LANG"] = env.pop("LC_ALL")
    env.pop("LANGUAGE", None)
    env["LC_MESSAGES"] = "C"
    if extra:
        env.update(extra)
    return env


@dataclass
class Tool:
    lang: str
    ok: bool
    label: str = ""             # e.g. "gcc 13.3.0"
    argv: list[str] = field(default_factory=list)
    version: tuple = ()
    note: str = ""              # message key for problems (e.g. "asm-platform")
    builtin: bool = False


@dataclass
class Step:
    argv: list[str]
    kind: str                   # "compile" or "run"


@dataclass
class Plan:
    lang: str
    workdir: Path
    steps: list[Step]
    env: dict
    tool: Tool

    def cleanup(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


class ToolMissing(Exception):
    def __init__(self, tool: Tool):
        super().__init__(tool.lang)
        self.tool = tool


def _version_output(argv: list[str], flag: str = "--version") -> str | None:
    try:
        r = subprocess.run(argv + [flag], capture_output=True, timeout=20, stdin=subprocess.DEVNULL,
                           env=child_env(), **popen_kwargs())
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout + r.stderr).decode("utf-8", "replace")


def _ver(text: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    return tuple(int(x) for x in m.groups() if x is not None) if m else ()


def _which(*names: str) -> list[str]:
    found = []
    for n in names:
        p = shutil.which(n)
        if p and p not in found:
            found.append(p)
    return found


class Toolchains:
    """Detects and caches tools. ``settings`` may override paths: {"python": "/usr/bin/python3"}."""

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self._cache: dict[str, Tool] = {}
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._cache.clear()

    def get(self, lang: str) -> Tool:
        with self._lock:
            if lang not in self._cache:
                self._cache[lang] = getattr(self, "_detect_" + lang, self._detect_unknown)(lang)
            return self._cache[lang]

    # ------------------------------------------------------------ detection
    def _override(self, lang: str) -> list[str] | None:
        value = (self.settings.get("tools") or {}).get(lang)
        return [value] if value else None

    def _detect_unknown(self, lang):
        return Tool(lang, False, note="unknown-language")

    def _detect_python(self, lang):
        candidates = []
        if self._override(lang):
            candidates.append(self._override(lang))
        if not FROZEN:
            candidates.append([sys.executable])
        for p in _which("python3", "python"):
            candidates.append([p])
        if IS_WINDOWS and shutil.which("py"):
            candidates.append([shutil.which("py"), "-3"])
        for argv in candidates:
            out = _version_output(argv)
            if out and out.strip().startswith("Python 3"):
                v = _ver(out)
                if v >= (3, 8):
                    return Tool(lang, True, out.strip(), argv, v)
        if FROZEN:  # the app's own bundled Python always works for the basics
            v = sys.version_info[:3]
            return Tool(lang, True, f"Python {v[0]}.{v[1]}.{v[2]} (built-in)", [sys.executable, "--run-python"],
                        tuple(v), builtin=True)
        return Tool(lang, False, note="missing")

    def _detect_c(self, lang, cxx=False):
        names = ("g++", "clang++", "c++") if cxx else ("gcc", "clang", "cc")
        cands = [self._override(lang)] if self._override(lang) else [[p] for p in _which(*names)]
        for argv in cands:
            out = _version_output(argv)
            if out:
                first = out.strip().split("\n")[0]
                kind = "clang" if "clang" in out.lower() else "gcc"
                return Tool(lang, True, first, argv, _ver(first.split(")")[-1] or first) or _ver(first), note=kind)
        return Tool(lang, False, note="missing")

    def _detect_cpp(self, lang):
        return self._detect_c(lang, cxx=True)

    def _detect_java(self, lang):
        cands = [self._override(lang)] if self._override(lang) else [[p] for p in _which("java")]
        for argv in cands:
            out = _version_output(argv, "-version")
            if out:
                lines = [ln for ln in out.split("\n") if ln.strip() and not ln.startswith("Picked up")]
                v = _ver(lines[0] if lines else out)
                if v and v[0] == 1 and len(v) > 1:  # "1.8.0" style
                    v = (v[1],)
                if v and v[0] < 11:
                    return Tool(lang, False, lines[0] if lines else "", argv, v, note="java-old")
                return Tool(lang, True, lines[0].strip() if lines else "java", argv, v)
        return Tool(lang, False, note="missing")

    def _detect_rust(self, lang):
        cands = [self._override(lang)] if self._override(lang) else [[p] for p in _which("rustc")]
        home = Path.home() / ".cargo" / "bin" / ("rustc" + EXE)
        if not cands and home.exists():
            cands = [[str(home)]]
        for argv in cands:
            out = _version_output(argv)
            if out:
                return Tool(lang, True, out.strip(), argv, _ver(out))
        return Tool(lang, False, note="missing")

    def _detect_go(self, lang):
        cands = [self._override(lang)] if self._override(lang) else [[p] for p in _which("go")]
        for argv in cands:
            out = _version_output(argv, "version")
            if out:
                return Tool(lang, True, out.strip(), argv, _ver(out))
        return Tool(lang, False, note="missing")

    def _detect_javascript(self, lang):
        if self._override(lang):
            return Tool(lang, True, self._override(lang)[0], self._override(lang), note="node")
        for name in ("node", "deno", "bun"):
            for p in _which(name):
                out = _version_output([p])
                if out:
                    return Tool(lang, True, f"{name} {out.strip()}", [p], _ver(out), note=name)
        return Tool(lang, False, note="missing")

    def _detect_typescript(self, lang):
        js = self._detect_javascript("javascript")
        if js.ok and js.note == "node":
            v = js.version
            if v >= (23, 6) or (v[:1] == (22,) and v >= (22, 18)):
                return Tool(lang, True, js.label + " (type stripping)", js.argv, v, note="node")
        for name in ("deno", "bun"):
            for p in _which(name):
                out = _version_output([p])
                if out:
                    return Tool(lang, True, f"{name} {out.strip()}", [p], _ver(out), note=name)
        tsc = _which("tsc")
        if tsc and js.ok:
            return Tool(lang, True, f"tsc + {js.label}", [tsc[0]] + js.argv, js.version, note="tsc")
        if js.ok:
            return Tool(lang, False, js.label, js.argv, js.version, note="node-old")
        return Tool(lang, False, note="missing")

    def _detect_asm(self, lang):
        machine = platform.machine().lower()
        if not sys.platform.startswith("linux") or machine not in ("x86_64", "amd64"):
            return Tool(lang, False, note="asm-platform")
        c = self._detect_c("c")
        if not c.ok:
            return Tool(lang, False, note="missing")
        return Tool(lang, True, c.label + " (GNU as)", c.argv, c.version)

    # ------------------------------------------------------------- planning
    def prepare(self, lang: str, code: str) -> Plan:
        tool = self.get(lang)
        if not tool.ok:
            raise ToolMissing(tool)
        work = Path(tempfile.mkdtemp(prefix="tutor-"))
        src = work / LANGS[lang]["file"]
        src.write_text(code, encoding="utf-8", newline="\n")
        exe = str(work / ("main" + EXE))
        steps: list[Step] = []
        env = child_env()
        if lang == "python":
            steps.append(Step(tool.argv + (["main.py"] if tool.builtin else ["-u", "main.py"]), "run"))
        elif lang in ("c", "cpp"):
            unbuf = work / ("tutor_unbuffered.c" if lang == "c" else "tutor_unbuffered.cpp")
            unbuf.write_text("#include <stdio.h>\n"
                             "/* Added by the tutorial app: show output immediately. */\n"
                             "__attribute__((constructor)) static void tutor_unbuffered(void)"
                             " { setvbuf(stdout, NULL, _IONBF, 0); }\n", encoding="utf-8")
            std = ["-std=c2x"] if lang == "c" else ["-std=c++2b"]
            steps.append(Step(tool.argv + std + ["-Wall", "-Wextra", "-g", "-O0", "-fdiagnostics-color=never",
                                                 src.name, unbuf.name, "-o", exe, "-lm"], "compile"))
            steps.append(Step([exe], "run"))
        elif lang == "java":
            # English compiler messages and the same number formatting (3.14, not 3,14) everywhere.
            argv = tool.argv + ["-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8", "-Dfile.encoding=UTF-8",
                                "-Duser.language=en", "-Duser.country=US"]
            major = tool.version[0] if tool.version else 0
            if 21 <= major < 25 and re.search(r"^\s*void\s+main\s*\(", code, re.M) and not re.search(r"\bclass\s+\w+", code):
                argv += ["--enable-preview", "--source", str(major)]
            steps.append(Step(argv + [src.name], "run"))
        elif lang == "rust":
            edition = "2024" if tool.version >= (1, 85) else "2021"
            steps.append(Step(tool.argv + ["--edition", edition, "--color", "never", src.name, "-o", exe], "compile"))
            steps.append(Step([exe], "run"))
        elif lang == "go":
            steps.append(Step(tool.argv + ["build", "-o", exe, src.name], "compile"))
            steps.append(Step([exe], "run"))
        elif lang == "javascript":
            if tool.note == "deno":
                steps.append(Step(tool.argv + ["run", "--no-prompt", "-A", src.name], "run"))
            else:
                steps.append(Step(tool.argv + [src.name], "run"))
        elif lang == "typescript":
            if tool.note == "node":
                steps.append(Step(tool.argv + ["--disable-warning=ExperimentalWarning", src.name], "run"))
            elif tool.note == "deno":
                steps.append(Step(tool.argv + ["run", "--no-prompt", "-A", src.name], "run"))
            elif tool.note == "bun":
                steps.append(Step(tool.argv + [src.name], "run"))
            else:  # tsc then node
                steps.append(Step([tool.argv[0], "--target", "ES2022", "--module", "commonjs", "--strict", src.name], "compile"))
                steps.append(Step(tool.argv[1:] + ["main.js"], "run"))
        elif lang == "asm":
            flags = ["-nostdlib", "-static"] if re.search(r"^\s*_start\s*:", code, re.M) else ["-no-pie"]
            steps.append(Step(tool.argv + ["-g"] + flags + [src.name, "-o", exe], "compile"))
            steps.append(Step([exe], "run"))
        return Plan(lang, work, steps, env, tool)

    def has_module(self, module: str) -> bool:
        tool = self.get("python")
        if not tool.ok or tool.builtin:
            return False
        try:
            r = subprocess.run(tool.argv + ["-c", f"import {module}"], capture_output=True, timeout=60,
                               stdin=subprocess.DEVNULL, env=child_env(), **popen_kwargs())
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


def clean_stderr(text: str) -> str:
    """Remove noise that confuses beginners (JVM option banners, Node warnings...)."""
    lines = [ln for ln in text.split("\n")
             if not ln.startswith(("Picked up JAVA_TOOL_OPTIONS", "Picked up _JAVA_OPTIONS",
                                   "Picked up JDK_JAVA_OPTIONS"))
             and "ExperimentalWarning: Type Stripping" not in ln
             and "(Use `node --trace-warnings" not in ln]
    return "\n".join(lines)


# --------------------------------------------------------------------- tests
@dataclass
class TestResult:
    input: str
    expected: str
    actual: str = ""
    stderr: str = ""
    passed: bool = False
    timeout: bool = False
    exit_code: int | None = None


@dataclass
class TestReport:
    ok: bool
    compile_output: str = ""
    compile_failed: bool = False
    results: list[TestResult] = field(default_factory=list)
    error: str = ""


def _run(argv, cwd, env, stdin_text, timeout):
    try:
        r = subprocess.run(argv, cwd=cwd, env=env, input=stdin_text.encode("utf-8"), capture_output=True,
                           timeout=timeout, **popen_kwargs())
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, out, "", True
    except OSError as e:
        return -1, "", str(e), False


def run_tests(chains: Toolchains, lang: str, code: str, tests: list[dict], match: str = "exact",
              timeout: float = 10, compile_timeout: float = 180) -> TestReport:
    plan = chains.prepare(lang, code)
    try:
        for step in plan.steps:
            if step.kind != "compile":
                continue
            rc, out, err, to = _run(step.argv, plan.workdir, plan.env, "", compile_timeout)
            if rc != 0 or to:
                return TestReport(False, clean_stderr(out + err), True, error="timeout" if to else "")
        run_step = [s for s in plan.steps if s.kind == "run"][-1]
        results = []
        for t in tests or [{"input": "", "output": ""}]:
            stdin_text = t.get("input", "")
            expected = t.get("output", "")
            rc, out, err, to = _run(run_step.argv, plan.workdir, plan.env, stdin_text, timeout)
            passed = (not to) and rc == 0 and output_matches(out, expected, t.get("match", match))
            results.append(TestResult(stdin_text, expected, out, clean_stderr(err), passed, to, rc))
        return TestReport(all(r.passed for r in results), results=results)
    finally:
        plan.cleanup()


# ------------------------------------------------------------ online runner
WANDBOX = "https://wandbox.org/api"
ONLINE_LANG = {"python": "Python", "c": "C", "cpp": "C++", "java": "Java", "rust": "Rust", "go": "Go",
               "javascript": "JavaScript", "typescript": "TypeScript", "asm": "C"}
_compilers: list | None = None


def _vkey(name: str):
    return [int(x) for x in re.findall(r"\d+", name)]


def online_compiler(lang: str) -> dict | None:
    global _compilers
    if _compilers is None:
        req = urllib.request.Request(WANDBOX + "/list.json", headers={"User-Agent": "tutorial-app"})
        with urllib.request.urlopen(req, timeout=30) as r:
            _compilers = json.load(r)
    want = ONLINE_LANG.get(lang)
    cands = [c for c in _compilers if c.get("language") == want and "head" not in c.get("name", "")]
    if lang == "python":
        cands = [c for c in cands if c["name"].startswith("cpython-3")]
    if lang in ("c", "asm"):
        cands = [c for c in cands if c["name"].startswith("gcc-")]
    if lang == "cpp":
        cands = [c for c in cands if c["name"].startswith("gcc-")] or cands
    cands.sort(key=lambda c: _vkey(c["name"]), reverse=True)
    return cands[0] if cands else None


def online_run(lang: str, code: str, stdin_text: str = "", timeout: float = 60) -> tuple[int, str, str]:
    """Compile and run on wandbox.org (a free, open-source online compiler).

    Only used after the learner agrees in the app, because the code is sent
    to a third-party server. Returns (exit_code, stdout, stderr).
    """
    comp = online_compiler(lang)
    if not comp:
        raise RuntimeError("no online compiler")
    payload = {"compiler": comp["name"], "code": code, "stdin": stdin_text, "save": False}
    if lang == "c":
        payload["compiler-option-raw"] = "-std=gnu2x\n-lm"
    elif lang == "cpp":
        payload["compiler-option-raw"] = "-std=gnu++2b"
    elif lang == "asm":
        flags = "-nostdlib\n-static" if re.search(r"^\s*_start\s*:", code, re.M) else "-no-pie"
        payload["code"] = "/* assembly is in main.s */\n"
        payload["codes"] = [{"file": "main.s", "code": code}]
        payload["compiler-option-raw"] = "main.s\n" + flags
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(WANDBOX + "/compile.json", data=data, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "tutorial-app"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    status = int(res.get("status") or 0) if str(res.get("status", "0")).lstrip("-").isdigit() else 1
    stderr = (res.get("compiler_error") or "") + (res.get("program_error") or "")
    if res.get("signal"):
        stderr += f"\n[signal: {res['signal']}]"
    return status, res.get("program_output") or "", stderr


def run_tests_online(lang: str, code: str, tests: list[dict], match: str = "exact") -> TestReport:
    results = []
    for t in tests or [{"input": "", "output": ""}]:
        rc, out, err = online_run(lang, code, t.get("input", ""))
        if rc != 0 and not out and err and not results:
            return TestReport(False, err, True)
        passed = rc == 0 and output_matches(out, t.get("output", ""), t.get("match", match))
        results.append(TestResult(t.get("input", ""), t.get("output", ""), out, err, passed, False, rc))
    return TestReport(all(r.passed for r in results), results=results)


def run_builtin_python(path: str) -> None:
    """Entry used by the frozen app: ``App --run-python main.py``."""
    import runpy
    # A windowed (no console) app may start without usable std streams: attach them to the pipes.
    for name, fd, mode in (("stdin", 0, "r"), ("stdout", 1, "w"), ("stderr", 2, "w")):
        if getattr(sys, name) is None:
            try:
                setattr(sys, name, open(fd, mode, encoding="utf-8", closefd=False))
            except OSError:
                pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    sys.argv = [path]
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    runpy.run_path(path, run_name="__main__")
