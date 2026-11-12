"""Small dialogs: first-run language picker, toolchain doctor, settings, search."""
from __future__ import annotations

import html

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
                               QTextBrowser, QVBoxLayout)

from .. import i18n, runners
from ..markdown import plain_text
from .theme import COLORS


class LanguageDialog(QDialog):
    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Language · 语言 · Язык")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("请选择界面语言 · 請選擇介面語言 · Choose your language · Выберите язык"))
        self.list = QListWidget()
        for code, name in i18n.available():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, code)
            self.list.addItem(item)
            if code == current:
                self.list.setCurrentItem(item)
        self.list.itemDoubleClicked.connect(lambda *_: self.accept())
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)
        self.resize(360, 380)

    def choice(self) -> str:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else "en"


class _DetectWorker(QObject):
    done = Signal(list)

    def __init__(self, chains, langs):
        super().__init__()
        self.chains, self.langs = chains, langs

    def run(self):
        self.done.emit([self.chains.get(lang) for lang in self.langs])


class DoctorDialog(QDialog):
    def __init__(self, tr, chains: runners.Toolchains, theme: str, langs: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.tr, self.chains, self.theme = tr, chains, theme
        self.langs = langs or list(runners.LANGS)
        self.setWindowTitle(tr("doctor.title"))
        lay = QVBoxLayout(self)
        intro = QLabel(tr("doctor.intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        again = QPushButton(tr("btn.recheck"))
        again.clicked.connect(self.refresh)
        close = QPushButton(tr("btn.close"))
        close.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(again)
        row.addWidget(close)
        lay.addLayout(row)
        self.resize(760, 560)
        self.refresh()

    def refresh(self):
        self.view.setHtml("<p>…</p>")
        self.chains.reset()
        self.thread = QThread(self)
        self.worker = _DetectWorker(self.chains, self.langs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.done.connect(self.show_tools)
        self.worker.done.connect(self.thread.quit)
        self.thread.start()

    def show_tools(self, tools):
        c = COLORS[self.theme]
        osk = runners.os_key()
        rows = []
        for tool in tools:
            name = runners.LANGS[tool.lang]["name"]
            if tool.ok:
                detail = html.escape(tool.label) + (" " + html.escape(self.tr("label.tool_builtin")) if tool.builtin else "")
                rows.append(f'<tr><td>✅</td><td><b>{name}</b></td><td>{detail}</td></tr>')
            else:
                reason = self.tr("tool." + (tool.note or "missing"))
                cmds = "\n".join(runners.INSTALL.get(tool.lang, {}).get(osk, []))
                rows.append(f'<tr><td>❌</td><td><b>{name}</b></td><td>{html.escape(reason)}<br>'
                            f'<span style="color:{c["muted"]}">{html.escape(self.tr("label.install_cmd"))}</span>'
                            f'<pre style="background-color:{c["line"]}">{html.escape(cmds)}</pre></td></tr>')
        more = html.escape(self.tr("label.install_more"))
        self.view.setHtml(f'<table cellpadding="6" width="100%">{"".join(rows)}</table><p>{more}</p>')


class SettingsDialog(QDialog):
    def __init__(self, tr, settings: dict, parent=None):
        super().__init__(parent)
        self.tr, self.settings = tr, settings
        self.setWindowTitle(tr("act.settings").rstrip("…."))
        form = QFormLayout(self)
        self.font = QSpinBox(minimum=8, maximum=32, value=settings.get("font_size", 12))
        form.addRow(tr("label.font_size"), self.font)
        self.timeout = QSpinBox(minimum=1, maximum=600, value=int(settings.get("timeout", 10)))
        form.addRow(tr("label.run_timeout"), self.timeout)
        self.online = QCheckBox(tr("label.online_title"))
        self.online.setChecked(bool(settings.get("online_ok")))
        form.addRow("", self.online)
        self.sanitize = QCheckBox(tr("label.sanitize"))
        self.sanitize.setToolTip(tr("label.sanitize_tip"))
        self.sanitize.setChecked(bool(settings.get("sanitize")))
        form.addRow("", self.sanitize)
        self.paths = {}
        tools = settings.get("tools") or {}
        for lang in ("python", "c", "cpp", "java", "rust", "go", "javascript"):
            edit = QLineEdit(tools.get(lang, ""))
            browse = QPushButton("…")
            browse.setFixedWidth(32)
            browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            form.addRow(tr("label.tool_path", lang=runners.LANGS[lang]["name"]), row)
            self.paths[lang] = edit
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("btn.ok"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("btn.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self, edit):
        path, _ = QFileDialog.getOpenFileName(self)
        if path:
            edit.setText(path)

    def values(self) -> dict:
        return {"font_size": self.font.value(), "timeout": self.timeout.value(),
                "online_ok": self.online.isChecked(), "sanitize": self.sanitize.isChecked(),
                "tools": {k: e.text().strip() for k, e in self.paths.items() if e.text().strip()}}


class SearchDialog(QDialog):
    open_lesson = Signal(str, str)

    def __init__(self, tr, library, locale: str, parent=None):
        super().__init__(parent)
        self.tr, self.library, self.locale = tr, library, locale
        self.setWindowTitle(tr("act.search").rstrip("…."))
        lay = QVBoxLayout(self)
        self.query = QLineEdit(placeholderText=tr("label.search_placeholder"))
        self.query.textChanged.connect(self.search)
        lay.addWidget(self.query)
        self.results = QListWidget()
        self.results.itemActivated.connect(self._open)
        lay.addWidget(self.results, 1)
        self.index = None
        self.resize(640, 480)

    def _build(self):
        self.index = []
        for cid, course in self.library.courses.items():
            for lid in course.lesson_ids:
                found = self.library.find_md(cid, lid, self.locale)
                if not found:
                    continue
                text = found[0].read_text(encoding="utf-8")
                self.index.append((cid, lid, course.title(self.locale), self.library.lesson_title(cid, lid, self.locale),
                                   plain_text(text)))

    def search(self, q: str):
        q = q.strip().lower()
        self.results.clear()
        if len(q) < 1:
            return
        if self.index is None:
            self._build()
        for cid, lid, ctitle, ltitle, text in self.index:
            pos = text.lower().find(q)
            in_title = q in ltitle.lower()
            if pos < 0 and not in_title:
                continue
            snippet = text[max(0, pos - 30): pos + 60].replace("\n", " ") if pos >= 0 else ""
            item = QListWidgetItem(f"{ctitle} › {ltitle}\n    …{snippet}…" if snippet else f"{ctitle} › {ltitle}")
            item.setData(Qt.UserRole, (cid, lid))
            self.results.addItem(item)
            if self.results.count() >= 200:
                break
        if self.results.count() == 0:
            self.results.addItem(self.tr("label.search_none"))

    def _open(self, item):
        data = item.data(Qt.UserRole)
        if data:
            self.open_lesson.emit(*data)
            self.accept()


def copy_text(text: str):
    QGuiApplication.clipboard().setText(text)
