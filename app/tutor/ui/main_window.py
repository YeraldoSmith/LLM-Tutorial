"""Main window: courses on the left, lesson in the middle, editor + output on the right."""
from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSplitter, QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import __version__, errors, runners
from ..content import Exercise
from ..markdown import Renderer
from . import dialogs, theme as theme_mod
from .console import Console, ProcessRunner
from .editor import CodeEditor
from .exercises import ExercisePanel

EXT_LANG = {".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".java": "java",
            ".rs": "rust", ".go": "go", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".s": "asm"}
STARTERS = {
    "python": 'print("Hello, World!")\n',
    "c": '#include <stdio.h>\n\nint main(void) {\n    printf("Hello, World!\\n");\n    return 0;\n}\n',
    "cpp": '#include <iostream>\n\nint main() {\n    std::cout << "Hello, World!\\n";\n}\n',
    "java": 'void main() {\n    IO.println("Hello, World!");\n}\n',
    "rust": 'fn main() {\n    println!("Hello, World!");\n}\n',
    "go": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Hello, World!")\n}\n',
    "javascript": 'console.log("Hello, World!");\n',
    "typescript": 'const name: string = "World";\nconsole.log(`Hello, ${name}!`);\n',
    "asm": ('.intel_syntax noprefix\n.globl main\n\n.section .rodata\nmsg: .asciz "Hello, World!"\n\n'
            '.text\nmain:\n    push rbp\n    lea rdi, [rip + msg]\n    call puts\n    xor eax, eax\n    pop rbp\n    ret\n'),
}


class Worker(QObject):
    done = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:  # shown to the user
            self.done.emit(exc)


class MainWindow(QMainWindow):
    language_changed = Signal(str)
    _deliver = Signal(object, object)   # (callback, result): runs callbacks on the UI thread

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.tr = ctx.tr
        self.lib = ctx.library
        self.settings = ctx.settings.data
        self.progress = ctx.progress
        self.chains = ctx.chains
        self.theme = theme_mod.resolve(self.settings.get("theme", "system"))
        self.font_size = int(self.settings.get("font_size", 12))
        self.lesson = None
        self.exercise: Exercise | None = None
        self.example_lang = None
        self.pending_input: list[str] = []
        self._threads = []
        self.setWindowTitle(self.lib.name(self.tr.content_locale))
        self._deliver.connect(self._on_deliver)
        self.runner = ProcessRunner(self)
        self.runner.output.connect(lambda text, stream: self.console.write(text, stream))
        self.runner.step_started.connect(self._step_started)
        self.runner.finished.connect(self._run_finished)
        self._build()
        self._build_menu()
        self.populate_tree()
        self._restore()

    # =================================================================== UI
    def _build(self):
        tr = self.tr
        # ---- left: course tree
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 0, 6)
        title = QLabel(tr("panel.courses"))
        title.setObjectName("title")
        ll.addWidget(title)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._tree_clicked)
        self.tree.itemActivated.connect(self._tree_clicked)
        ll.addWidget(self.tree, 1)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("muted")
        ll.addWidget(self.progress_label)

        # ---- middle: lesson + exercises
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(6, 6, 6, 6)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton(tr("btn.prev"))
        self.prev_btn.clicked.connect(lambda: self.go(-1))
        self.next_btn = QPushButton(tr("btn.next"))
        self.next_btn.clicked.connect(lambda: self.go(1))
        self.done_btn = QPushButton(tr("btn.done"))
        self.done_btn.clicked.connect(self.toggle_done)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch(1)
        nav.addWidget(self.done_btn)
        ml.addLayout(nav)
        self.tabs = QTabWidget()
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._anchor)
        self.tabs.addTab(self.browser, tr("tab.lesson"))
        self.ex_panel = ExercisePanel(tr)
        self.ex_panel.open_in_editor.connect(self.start_exercise)
        self.ex_panel.show_solution.connect(lambda ex: None)
        self.ex_panel.solved.connect(self._exercise_solved)
        self.tabs.addTab(self.ex_panel, tr("tab.exercises_none"))
        ml.addWidget(self.tabs, 1)

        # ---- right: editor + console
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 6, 6, 6)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(tr("label.language")))
        self.lang_box = QComboBox()
        for code, info in runners.LANGS.items():
            self.lang_box.addItem(info["name"], code)
        self.lang_box.currentIndexChanged.connect(self._lang_changed)
        bar.addWidget(self.lang_box)
        bar.addStretch(1)
        self.run_btn = QPushButton(tr("btn.run"))
        self.run_btn.setObjectName("primary")
        self.run_btn.setToolTip("F5")
        self.run_btn.clicked.connect(self.run_code)
        self.stop_btn = QPushButton(tr("btn.stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_code)
        bar.addWidget(self.run_btn)
        bar.addWidget(self.stop_btn)
        rl.addLayout(bar)
        exbar = QHBoxLayout()
        self.ex_label = QLabel()
        self.ex_label.setObjectName("bar")
        self.ex_label.setWordWrap(True)
        self.check_btn = QPushButton(tr("btn.check"))
        self.check_btn.setObjectName("ok")
        self.check_btn.setToolTip("Ctrl+Enter")
        self.check_btn.clicked.connect(self.check_exercise)
        self.reset_btn = QPushButton(tr("btn.reset"))
        self.reset_btn.clicked.connect(self.reset_code)
        exbar.addWidget(self.ex_label, 1)
        exbar.addWidget(self.reset_btn)
        exbar.addWidget(self.check_btn)
        self.exbar_widgets = (self.ex_label, self.reset_btn, self.check_btn)
        rl.addLayout(exbar)
        vsplit = QSplitter(Qt.Vertical)
        self.editor = CodeEditor(self.theme, self.font_size)
        vsplit.addWidget(self.editor)
        cons = QWidget()
        cl = QVBoxLayout(cons)
        cl.setContentsMargins(0, 4, 0, 0)
        out_title = QLabel(tr("panel.output"))
        out_title.setObjectName("muted")
        cl.addWidget(out_title)
        self.console = Console(tr, self.theme, self.font_size)
        self.console.send_line.connect(self._send_input)
        cl.addWidget(self.console, 1)
        vsplit.addWidget(cons)
        vsplit.setSizes([420, 260])
        self.vsplit = vsplit
        rl.addWidget(vsplit, 1)

        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(left)
        self.split.addWidget(mid)
        self.split.addWidget(right)
        self.split.setStretchFactor(1, 1)
        self.split.setStretchFactor(2, 1)
        self.split.setSizes([250, 620, 560])
        self.setCentralWidget(self.split)
        self._show_exercise_bar(False)
        self.status = self.statusBar()

        QShortcut(QKeySequence("F5"), self, activated=self.run_code)
        QShortcut(QKeySequence("Shift+F5"), self, activated=self.stop_code)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.check_exercise)
        QShortcut(QKeySequence("Alt+Left"), self, activated=lambda: self.go(-1))
        QShortcut(QKeySequence("Alt+Right"), self, activated=lambda: self.go(1))

    def _build_menu(self):
        tr, mb = self.tr, self.menuBar()

        def act(menu, key, slot, shortcut=None):
            a = QAction(tr(key), self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        m = mb.addMenu(tr("menu.file"))
        act(m, "act.new", self.new_code, QKeySequence.New)
        act(m, "act.open", self.open_file, QKeySequence.Open)
        act(m, "act.save", self.save_file, QKeySequence.Save)
        act(m, "act.save_as", lambda: self.save_file(True), QKeySequence.SaveAs)
        m.addSeparator()
        act(m, "act.search", self.search, "Ctrl+Shift+F")
        m.addSeparator()
        act(m, "act.quit", self.close, QKeySequence.Quit)

        m = mb.addMenu(tr("menu.view"))
        act(m, "act.zoom_in", lambda: self.zoom(1), QKeySequence.ZoomIn)
        act(m, "act.zoom_out", lambda: self.zoom(-1), QKeySequence.ZoomOut)
        act(m, "act.zoom_reset", lambda: self.zoom(0), "Ctrl+0")
        tm = m.addMenu(tr("act.theme"))
        group = QActionGroup(self)
        for key in ("system", "light", "dark"):
            a = QAction(tr("theme." + key), self, checkable=True)
            a.setChecked(self.settings.get("theme", "system") == key)
            a.triggered.connect(lambda _=False, k=key: self.set_theme(k))
            group.addAction(a)
            tm.addAction(a)
        lm = m.addMenu(tr("menu.language"))
        lgroup = QActionGroup(self)
        from .. import i18n
        for code, name in i18n.available():
            a = QAction(name, self, checkable=True)
            a.setChecked(code == self.tr.code)
            a.triggered.connect(lambda _=False, c=code: self.language_changed.emit(c))
            lgroup.addAction(a)
            lm.addAction(a)

        m = mb.addMenu(tr("menu.tools"))
        act(m, "act.doctor", self.doctor)
        act(m, "act.update", self.update_content)
        act(m, "act.open_folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.lib.root))))
        m.addSeparator()
        act(m, "act.settings", self.open_settings, QKeySequence.Preferences)

        m = mb.addMenu(tr("menu.help"))
        act(m, "act.start_here", self.start_here)
        if self.lib.repo:
            act(m, "act.website", lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{self.lib.repo}")))
            act(m, "act.report", lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{self.lib.repo}/issues")))
        act(m, "act.about", self.about)

    # ================================================================ tree
    def populate_tree(self):
        loc = self.tr.content_locale
        self.tree.clear()
        self.items = {}
        done_total = total = 0
        for cid, course in self.lib.courses.items():
            done = sum(1 for lid in course.lesson_ids if self.progress.lesson_done(f"{cid}/{lid}"))
            done_total += done
            total += len(course.lesson_ids)
            top = QTreeWidgetItem([f"{course.icon}  {course.title(loc)}   {done}/{len(course.lesson_ids)}"])
            top.setData(0, Qt.UserRole, (cid, None))
            top.setToolTip(0, course.summary(loc))
            self.tree.addTopLevelItem(top)
            for lid in course.lesson_ids:
                mark = "  ✓" if self.progress.lesson_done(f"{cid}/{lid}") else ""
                title = self.lib.lesson_title(cid, lid, loc)
                child = QTreeWidgetItem([f"{title}{mark}"])
                child.setData(0, Qt.UserRole, (cid, lid))
                child.setToolTip(0, title)
                top.addChild(child)
                self.items[(cid, lid)] = child
        self.progress_label.setText(self.tr("label.progress", done=done_total, total=total))
        if self.lesson:
            self._select_in_tree(self.lesson.course, self.lesson.id)

    def _select_in_tree(self, cid, lid):
        item = self.items.get((cid, lid))
        if item:
            item.parent().setExpanded(True)
            self.tree.setCurrentItem(item)

    def _tree_clicked(self, item, *_):
        cid, lid = item.data(0, Qt.UserRole)
        if lid is None:
            item.setExpanded(not item.isExpanded())
            if not item.isExpanded():
                return
            ids = self.lib.courses[cid].lesson_ids
            if ids and (not self.lesson or self.lesson.course != cid):
                self.open_lesson(cid, ids[0])
            return
        self.open_lesson(cid, lid)

    # ============================================================== lesson
    def renderer(self, base: Path | None = None) -> Renderer:
        labels = {k: self.tr("md." + k) for k in ("note", "tip", "important", "warning", "caution", "run", "copy")}

        def link(href):
            if base is not None:
                target = self.lib.resolve_link(base, href)
                if target:
                    return f"lesson:{target[0]}/{target[1]}" + (f"#{target[2]}" if target[2] else "")
            return href

        def image(src):
            if base is not None and not src.startswith(("http://", "https://", "data:")):
                return QUrl.fromLocalFile(str((base / src).resolve())).toString()
            return src
        lang = self.lesson.language if self.lesson else None
        return Renderer(target="qt", labels=labels, theme=self.theme, link_resolver=link, image_resolver=image,
                        default_lang=lang)

    def open_lesson(self, cid: str, lid: str, anchor: str = ""):
        self._save_draft()
        loc = self.tr.content_locale
        self.lesson = self.lib.lesson(cid, lid, loc)
        self.settings["last"] = f"{cid}/{lid}"
        self._select_in_tree(cid, lid)
        self.render_lesson()
        if anchor:
            self.browser.scrollToAnchor(anchor)
        n = len(self.lesson.exercises)
        self.tabs.setTabText(1, self.tr("tab.exercises", n=n) if n else self.tr("tab.exercises_none"))
        done = {ex.id for ex in self.lesson.exercises if self.progress.exercise_done(f"{self.lesson.key}/{ex.id}")}
        self.ex_panel.load(self.lesson.exercises, self.theme, self.font_size, done, self.renderer(self.lesson.path))
        self.tabs.setCurrentIndex(0)
        prev, nxt = self.lib.neighbours(cid, lid)
        self.prev_btn.setEnabled(prev is not None)
        self.next_btn.setEnabled(nxt is not None)
        self._update_done_btn()
        lang = self.lesson.language if self.lesson.language in runners.LANGS else None
        if lang and not self.exercise:
            self.set_language(lang)
        self.setWindowTitle(f"{self.lesson.title} — {self.lib.name(loc)}")

    def render_lesson(self):
        if not self.lesson:
            return
        loc = self.tr.content_locale
        r = self.renderer(self.lesson.path)
        rendered = r.render(self.lesson.body_md)
        self.code_blocks = rendered.code_blocks
        c = theme_mod.COLORS[self.theme]
        extra = ""
        if self.lesson.locale != loc and self.lesson.locale.split("-")[0] != loc.split("-")[0]:
            extra = (f'<p style="color:{c["warn"]}">' +
                     html.escape(self.tr("label.lesson_loaded_from", want=loc, used=self.lesson.locale)) + "</p>")
        tail = ""
        n = len(self.lesson.exercises)
        if n:
            tail += (f'<table width="100%" cellpadding="10" style="background-color:{c["line"]};margin-top:12px">'
                     f'<tr><td>✍️ <a href="exercises:open">{html.escape(self.tr("label.exercise_section", n=n))}</a>'
                     f"</td></tr></table>")
        _, nxt = self.lib.neighbours(self.lesson.course, self.lesson.id)
        if nxt:
            ntitle = self.lib.lesson_title(self.lesson.course, nxt, loc)
            tail += (f'<p align="right" style="margin-top:16px"><a href="lesson:{self.lesson.course}/{nxt}">'
                     f"{html.escape(self.tr('btn.next'))}  {html.escape(ntitle)}</a></p>")
        doc = self.browser.document()
        doc.setDefaultStyleSheet(theme_mod.lesson_css(self.theme, self.font_size + 1))
        doc.setBaseUrl(QUrl.fromLocalFile(str(self.lesson.path) + "/"))
        font = self.browser.font()
        font.setPointSize(self.font_size + 1)
        self.browser.setFont(font)
        self.browser.setHtml(f"<html><body>{extra}{rendered.html}{tail}</body></html>")

    def _anchor(self, url: QUrl):
        s = url.toString()
        if s.startswith("run:"):
            idx = int(s[4:])
            block = self.code_blocks[idx]
            self.exercise = None
            self._show_exercise_bar(False)
            self.example_lang = block.lang
            self.set_language(block.lang)
            self.editor.setPlainText(block.code.rstrip() + "\n")
            self.status.showMessage(self.tr("label.example_loaded"), 6000)
            self.editor.setFocus()
            return
        if s.startswith("exercises:"):
            self.tabs.setCurrentIndex(1)
            return
        if s.startswith("lesson:"):
            target, _, anchor = s[7:].partition("#")
            cid, _, lid = target.partition("/")
            self.open_lesson(cid, lid, anchor)
            return
        if s.startswith("#"):
            self.browser.scrollToAnchor(s[1:])
            return
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        if url.isLocalFile():
            QDesktopServices.openUrl(url)

    def go(self, step: int):
        if not self.lesson:
            return
        prev, nxt = self.lib.neighbours(self.lesson.course, self.lesson.id)
        target = prev if step < 0 else nxt
        if target:
            self.open_lesson(self.lesson.course, target)

    def toggle_done(self):
        if not self.lesson:
            return
        key = self.lesson.key
        self.progress.set_lesson(key, not self.progress.lesson_done(key))
        self._update_done_btn()
        self.populate_tree()
        if self.progress.lesson_done(key):
            self.status.showMessage(self.tr("label.lesson_done"), 4000)

    def _update_done_btn(self):
        done = self.lesson and self.progress.lesson_done(self.lesson.key)
        self.done_btn.setText(self.tr("label.lesson_done") if done else self.tr("btn.done"))
        self.done_btn.setObjectName("" if done else "ok")
        self.done_btn.style().unpolish(self.done_btn)
        self.done_btn.style().polish(self.done_btn)

    # ============================================================== editor
    def current_lang(self) -> str:
        return self.lang_box.currentData()

    def set_language(self, lang: str):
        idx = self.lang_box.findData(lang)
        if idx >= 0 and idx != self.lang_box.currentIndex():
            self.lang_box.blockSignals(True)
            self.lang_box.setCurrentIndex(idx)
            self.lang_box.blockSignals(False)
        self.editor.set_language(lang)
        self._update_tool_status()

    def _lang_changed(self, *_):
        lang = self.current_lang()
        self.editor.set_language(lang)
        if not self.exercise and (not self.editor.toPlainText().strip() or
                                  self.editor.toPlainText() in STARTERS.values()):
            self.editor.setPlainText(STARTERS.get(lang, ""))
        self._update_tool_status()

    def _update_tool_status(self):
        lang = self.current_lang()
        QTimer.singleShot(0, lambda: self._show_tool(lang))

    def _show_tool(self, lang):
        tool = self.chains.get(lang)
        name = runners.LANGS[lang]["name"]
        if tool.ok:
            self.status.showMessage(self.tr("label.tool_ok", lang=name, label=tool.label.split("\n")[0][:80]))
        else:
            self.status.showMessage(self.tr("label.tool_missing", lang=name) + " — " + self.tr("tool." + (tool.note or "missing")))

    def new_code(self):
        self._save_draft()
        self.exercise = None
        self._show_exercise_bar(False)
        self.editor.setPlainText(STARTERS.get(self.current_lang(), ""))

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("act.open"))
        if not path:
            return
        p = Path(path)
        self.exercise = None
        self._show_exercise_bar(False)
        self.set_language(EXT_LANG.get(p.suffix.lower(), self.current_lang()))
        self.editor.setPlainText(p.read_text(encoding="utf-8", errors="replace"))
        self.current_file = p

    def save_file(self, ask=False):
        path = getattr(self, "current_file", None)
        if ask or not path:
            default = runners.LANGS[self.current_lang()]["file"]
            name, _ = QFileDialog.getSaveFileName(self, self.tr("act.save_as"), str(Path.home() / default))
            if not name:
                return
            path = Path(name)
        path.write_text(self.editor.toPlainText(), encoding="utf-8")
        self.current_file = path
        self.status.showMessage(str(path), 4000)

    def zoom(self, step: int):
        self.font_size = 12 if step == 0 else max(8, min(32, self.font_size + step))
        self.settings["font_size"] = self.font_size
        self.editor.set_font_size(self.font_size)
        self.console.set_font_size(self.font_size)
        self.render_lesson()

    def set_theme(self, key: str):
        self.settings["theme"] = key
        self.theme = theme_mod.resolve(key)
        theme_mod.apply(self.theme)
        self.editor.set_theme(self.theme)
        self.console.set_theme(self.theme)
        if self.lesson:
            self.open_lesson(self.lesson.course, self.lesson.id)

    # ============================================================= running
    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.check_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def run_code(self):
        if self.runner.running:
            return
        lang, code = self.current_lang(), self.editor.toPlainText()
        tool = self.chains.get(lang)
        self.console.system("─" * 30)
        if not tool.ok:
            self._run_online(lang, code)
            return
        try:
            plan = self.chains.prepare(lang, code)
        except runners.ToolMissing:
            self._run_online(lang, code)
            return
        self.run_lang = lang
        self._set_running(True)
        self.runner.start(plan)
        for line in self.pending_input:
            self.runner.write(line + "\n")
        self.pending_input = []
        self.console.input.setFocus()

    def _step_started(self, kind):
        self.console.system(self.tr("label.compiling") if kind == "compile" else self.tr("label.running"))

    def _send_input(self, text: str):
        if self.runner.running:
            self.runner.write(text + "\n")
        else:
            self.pending_input.append(text)

    def stop_code(self):
        self.runner.stop()

    def _run_finished(self, code, crashed, kind, stderr_all):
        self._set_running(False)
        if self.runner.stopped:
            self.console.system(self.tr("label.stopped"))
        elif kind == "compile":
            self.console.system(self.tr("label.compile_failed"))
        elif code == 0 and not crashed:
            self.console.system(self.tr("label.finished_ok"))
        else:
            self.console.system(self.tr("label.finished", code=code if not crashed else "crash"))
        if not self.runner.stopped and (code != 0 or crashed or stderr_all.strip()):
            exit_code = -11 if crashed and not runners.IS_WINDOWS else code
            self.console.explain(errors.explain(self.run_lang, stderr_all, exit_code if kind == "run" else None,
                                                False, self.tr.get))

    def _ensure_online_ok(self, lang) -> bool:
        if self.settings.get("online_ok"):
            return True
        name = runners.LANGS[lang]["name"]
        box = QMessageBox(self)
        box.setWindowTitle(self.tr("label.online_title"))
        box.setText(self.tr("label.online_consent", lang=name))
        yes = box.addButton(self.tr("btn.use_online"), QMessageBox.AcceptRole)
        box.addButton(self.tr("btn.install_help"), QMessageBox.HelpRole)
        box.addButton(self.tr("btn.cancel"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == yes:
            self.settings["online_ok"] = True
            self.ctx.settings.save()
            return True
        if box.buttonRole(box.clickedButton()) == QMessageBox.HelpRole:
            self.doctor([lang])
        return False

    def _thread(self, fn, on_done):
        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(lambda result, cb=on_done: self._deliver.emit(cb, result))
        worker.done.connect(thread.quit)
        thread.finished.connect(lambda: self._threads.remove((thread, worker)) if (thread, worker) in self._threads else None)
        self._threads.append((thread, worker))
        thread.start()

    def _on_deliver(self, callback, result):
        callback(result)

    def _run_online(self, lang, code):
        if not self._ensure_online_ok(lang):
            self._show_tool(lang)
            return
        stdin_text = "".join(line + "\n" for line in self.pending_input)
        self.pending_input = []
        self.console.system(self.tr("label.online_running"))
        self._set_running(True)
        self.stop_btn.setEnabled(False)

        def done(result):
            self._set_running(False)
            if isinstance(result, Exception):
                self.console.system(self.tr("label.online_failed", error=str(result)))
                return
            rc, out, err = result
            if out:
                self.console.write(out, "out")
            if err:
                self.console.write(err, "err")
            self.console.system(self.tr("label.finished", code=rc) if rc else self.tr("label.finished_ok"))
            if rc or err.strip():
                self.console.explain(errors.explain(lang, err, rc, False, self.tr.get))
        self._thread(lambda: runners.online_run(lang, code, stdin_text), done)

    # =========================================================== exercises
    def _show_exercise_bar(self, show: bool):
        for w in self.exbar_widgets:
            w.setVisible(show)

    def _draft_key(self) -> str | None:
        if self.exercise and self.lesson:
            return f"{self.lesson.key}/{self.exercise.id}"
        return None

    def _save_draft(self):
        key = self._draft_key()
        if key:
            self.progress.set_draft(key, self.editor.toPlainText())

    def start_exercise(self, ex: Exercise):
        self._save_draft()
        self.exercise = ex
        lang = ex.language if ex.language in runners.LANGS else self.current_lang()
        self.set_language(lang)
        draft = self.progress.draft(self._draft_key())
        self.editor.setPlainText(draft if draft is not None else ex.starter)
        self.ex_label.setText(self.tr("label.exercise_in_editor", title=ex.title))
        self._show_exercise_bar(True)
        self.editor.setFocus()

    def reset_code(self):
        if self.exercise:
            self.editor.setPlainText(self.exercise.starter)

    def check_exercise(self):
        ex = self.exercise
        if not ex or ex.kind != "code" or self.runner.running:
            return
        self._save_draft()
        code, lang = self.editor.toPlainText(), self.current_lang()
        for pkg in ex.requires:
            if not self.chains.has_module(pkg):
                QMessageBox.information(self, self.tr("btn.check"),
                                        self.tr("label.requires", pkgs=", ".join(ex.requires)) + "\n\n" +
                                        self.tr("label.pip_hint", pkgs=" ".join(ex.requires)))
                return
        tool = self.chains.get(lang)
        online = not tool.ok
        if online and not self._ensure_online_ok(lang):
            return
        timeout = float(self.settings.get("timeout", 10))
        self._set_running(True)
        self.stop_btn.setEnabled(False)
        self.console.system("─" * 30)
        self.console.system(self.tr("label.online_running") if online else self.tr("label.running"))
        lesson_key = self.lesson.key

        def work():
            if online:
                return runners.run_tests_online(lang, code, ex.tests, ex.match)
            return runners.run_tests(self.chains, lang, code, ex.tests, ex.match, timeout=timeout)
        self._thread(work, lambda report: self._show_report(report, ex, lang, lesson_key, timeout))

    def _show_report(self, report, ex, lang, lesson_key, timeout):
        self._set_running(False)
        c = theme_mod.COLORS[self.theme]
        card = self.ex_panel.cards.get(ex.id)
        if isinstance(report, Exception):
            self.console.system(self.tr("label.online_failed", error=str(report)))
            return
        if report.compile_failed:
            self.console.write(report.compile_output, "err")
            self.console.system(self.tr("label.compile_failed"))
            self.console.explain(errors.explain(lang, report.compile_output, None, False, self.tr.get))
            if card:
                card.show_report(f'<p style="color:{c["bad"]}"><b>{html.escape(self.tr("label.compile_failed"))}</b></p>')
            return
        rows = []
        first_fail = None
        for i, r in enumerate(report.results, 1):
            mark = "✅" if r.passed else "❌"
            status = self.tr("label.passed") if r.passed else (self.tr("label.timeout", s=int(timeout)) if r.timeout else self.tr("label.failed"))
            rows.append(f"<p><b>{mark} {html.escape(self.tr('label.test_n', n=i))}</b> — {html.escape(status)}</p>")
            if not r.passed:
                first_fail = first_fail or r
                def block(label, text):
                    return (f'<p style="color:{c["muted"]}">{html.escape(label)}</p>'
                            f'<pre style="background-color:{c["line"]}">{html.escape(text.rstrip() or "∅")}</pre>')
                if r.input:
                    rows.append(block(self.tr("label.input"), r.input))
                rows.append(block(self.tr("label.expected"), r.expected))
                rows.append(block(self.tr("label.actual"), r.actual))
                if r.stderr.strip():
                    rows.append(block(self.tr("label.stderr"), r.stderr[-1500:]))
        passed = sum(r.passed for r in report.results)
        total = len(report.results)
        if report.ok:
            head = f'<p style="color:{c["ok"]}"><b>{html.escape(self.tr("label.all_passed"))}</b></p>'
        else:
            head = f'<p style="color:{c["bad"]}"><b>{html.escape(self.tr("label.some_failed", passed=passed, total=total))}</b></p>'
        if card:
            card.show_report(head + "".join(rows))
        self.console.system(self.tr("label.all_passed") if report.ok else self.tr("label.some_failed", passed=passed, total=total))
        if first_fail is not None:
            if first_fail.stderr.strip():
                self.console.write(first_fail.stderr, "err")
            self.console.explain(errors.explain(lang, first_fail.stderr, first_fail.exit_code, first_fail.timeout, self.tr.get))
        if report.ok and card:
            card.mark_done()
        self.tabs.setCurrentIndex(1)
        if card:
            self.ex_panel.ensureWidgetVisible(card)

    def _exercise_solved(self, ex: Exercise):
        if not self.lesson:
            return
        self.progress.set_exercise(f"{self.lesson.key}/{ex.id}")
        if all(self.progress.exercise_done(f"{self.lesson.key}/{e.id}") for e in self.lesson.exercises):
            if not self.progress.lesson_done(self.lesson.key):
                self.progress.set_lesson(self.lesson.key, True)
                self._update_done_btn()
                self.populate_tree()

    # ============================================================= dialogs
    def doctor(self, langs=None):
        dialogs.DoctorDialog(self.tr, self.chains, self.theme, langs if isinstance(langs, list) else None, self).exec()
        self._update_tool_status()

    def open_settings(self):
        dlg = dialogs.SettingsDialog(self.tr, self.settings, self)
        if dlg.exec():
            self.settings.update(dlg.values())
            self.ctx.settings.save()
            self.chains.settings = self.settings
            self.chains.reset()
            if self.settings["font_size"] != self.font_size:
                self.font_size = self.settings["font_size"]
                self.editor.set_font_size(self.font_size)
                self.console.set_font_size(self.font_size)
                self.render_lesson()
            self._update_tool_status()

    def search(self):
        dlg = dialogs.SearchDialog(self.tr, self.lib, self.tr.content_locale, self)
        dlg.open_lesson.connect(self.open_lesson)
        dlg.exec()

    def start_here(self):
        first = next(iter(self.lib.courses.values()), None)
        if first and first.lesson_ids:
            self.open_lesson(first.id, first.lesson_ids[0])

    def about(self):
        QMessageBox.about(self, self.tr("act.about"), self.tr(
            "label.about", name=self.lib.name(self.tr.content_locale), version=__version__,
            repo=f"https://github.com/{self.lib.repo}") + "\n\n" + self.tr("label.content_source", path=str(self.lib.root)))

    def update_content(self):
        if not self.lib.repo:
            return
        if QMessageBox.question(self, self.tr("act.update"), self.tr("label.update_intro")) != QMessageBox.Yes:
            return
        from .. import updater
        dest = self.ctx.data_dir / "downloads"
        self.status.showMessage("…")

        def done(result):
            if isinstance(result, Exception):
                QMessageBox.warning(self, self.tr("act.update"), self.tr("label.update_failed", error=str(result)))
                return
            self.settings["content_path"] = str(result)
            self.ctx.settings.save()
            self.ctx.reload_library(result)
            self.lib = self.ctx.library
            self.populate_tree()
            if self.lesson and self.lesson.course in self.lib.courses and \
                    self.lesson.id in self.lib.courses[self.lesson.course].lesson_ids:
                self.open_lesson(self.lesson.course, self.lesson.id)
            QMessageBox.information(self, self.tr("act.update"), self.tr("label.update_done", n=len(self.lib.courses)))
        self._thread(lambda: updater.download_content(self.lib.repo, dest), done)

    # ============================================================ lifecycle
    def _restore(self):
        geo = self.settings.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        else:
            self.resize(1440, 900)
        sizes = self.settings.get("split")
        if sizes and len(sizes) == 3:
            self.split.setSizes(sizes)
        last = self.settings.get("last", "")
        cid, _, lid = last.partition("/")
        if cid in self.lib.courses and lid in self.lib.courses[cid].lesson_ids:
            self.open_lesson(cid, lid)
        else:
            self.start_here()
        if not self.editor.toPlainText():
            self.editor.setPlainText(STARTERS.get(self.current_lang(), ""))

    def save_state(self):
        self._save_draft()
        self.settings["geometry"] = bytes(self.saveGeometry().toBase64()).decode()
        self.settings["split"] = self.split.sizes()
        self.ctx.settings.save()

    def closeEvent(self, event):
        self.runner.stop()
        self.save_state()
        super().closeEvent(event)
