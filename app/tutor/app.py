"""Start the app: find the course content, load settings, open the main window."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import i18n, runners
from .content import Library
from .storage import JsonFile, Progress, data_dir


def find_content(explicit: str | None, settings: dict) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if settings.get("content_path"):
        candidates.append(Path(settings["content_path"]))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2])                       # running from a git checkout
    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / "content")   # bundled with PyInstaller
    candidates.append(Path(sys.executable).resolve().parent / "content")
    for c in candidates:
        if (c / "tutorial.json").exists():
            return c
    raise SystemExit("Course content not found (tutorial.json). Use --content PATH.")


def bundled_app_id(default: str = "tutor") -> str:
    """Read the repo id early (before settings) so each tutorial keeps its own progress."""
    import json
    for c in (Path(__file__).resolve().parents[2], Path(getattr(sys, "_MEIPASS", "")) / "content",
              Path(sys.executable).resolve().parent / "content"):
        try:
            return json.loads((c / "tutorial.json").read_text(encoding="utf-8")).get("id", default)
        except (OSError, ValueError):
            continue
    return default


class Context:
    def __init__(self, args):
        self.data_dir = data_dir(bundled_app_id())
        self.settings = JsonFile(self.data_dir / "settings.json", {"theme": "system", "font_size": 12, "timeout": 10})
        if args.content:
            self.settings.data.pop("content_path", None)
        root = find_content(args.content, self.settings.data)
        self.library = Library(root)
        self.progress = Progress(self.data_dir)
        lang = args.lang or self.settings.data.get("ui_lang")
        self.first_run = lang is None
        self.tr = i18n.Translator(lang or i18n.detect_system())
        self.chains = runners.Toolchains(self.settings.data)

    def reload_library(self, root: Path):
        self.library = Library(root)

    def set_language(self, code: str):
        self.settings.data["ui_lang"] = code
        self.settings.save()
        self.tr = i18n.Translator(code)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--run-python"]:          # the frozen app doubles as a Python interpreter
        runners.run_builtin_python(argv[1])
        return 0
    ap = argparse.ArgumentParser(description="Interactive programming tutorial")
    ap.add_argument("--content", help="folder that contains tutorial.json")
    ap.add_argument("--lang", help="interface language, e.g. zh-CN, en, ru")
    ap.add_argument("--open", help="open a lesson, e.g. python/01-hello")
    ap.add_argument("--theme", choices=["system", "light", "dark"])
    ap.add_argument("--screenshot", help="save a screenshot and quit (for testing)")
    ap.add_argument("--size", default="1440x900")
    ap.add_argument("--run-example", type=int, help="with --screenshot: load code block N and run it")
    args = ap.parse_args(argv)

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Tutor")
    icon = Path(__file__).resolve().parent / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    ctx = Context(args)
    if args.theme:
        ctx.settings.data["theme"] = args.theme

    from .ui import theme
    from .ui.dialogs import LanguageDialog
    from .ui.main_window import MainWindow
    theme.apply(theme.resolve(ctx.settings.data.get("theme", "system")))

    if ctx.first_run and not args.screenshot:
        dlg = LanguageDialog(ctx.tr.code)
        dlg.exec()
        ctx.set_language(dlg.choice())

    holder = {}

    def make_window(first: bool):
        win = MainWindow(ctx)
        win.language_changed.connect(switch_language)
        holder["win"] = win
        return win

    def switch_language(code: str):
        old = holder["win"]
        old.save_state()
        ctx.set_language(code)
        win = make_window(False)
        win.restoreGeometry(old.saveGeometry())
        win.show()
        old.runner.stop()
        old.hide()
        old.deleteLater()

    win = make_window(True)
    if args.open:
        cid, _, lid = args.open.partition("/")
        win.open_lesson(cid, lid)
    if args.screenshot:
        w, h = (int(x) for x in args.size.split("x"))
        win.resize(w, h)
        win.show()

        def shot():
            win.grab().save(args.screenshot)
            app.quit()

        def go():
            if args.run_example is not None:
                win._anchor(__import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl(f"run:{args.run_example}"))
                win.run_code()
                QTimer.singleShot(4000, shot)
            else:
                QTimer.singleShot(300, shot)
        QTimer.singleShot(500, go)
        return app.exec()
    win.show()
    if ctx.first_run:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(win, ctx.tr("label.welcome_title"), ctx.tr("label.welcome_text"))
    code = app.exec()
    return code
