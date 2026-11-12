"""Output console + process runner (compile, then run interactively)."""
from __future__ import annotations

import codecs
import html

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .. import runners
from ..markdown import Renderer
from .theme import COLORS, mono_font


class ProcessRunner(QObject):
    """Runs a Plan's steps one after another with QProcess.

    Signals: output(text, stream) with stream in {"out", "err"}; step_started(kind);
    finished(exit_code, crashed, failed_step_kind, all_stderr)."""
    output = Signal(str, str)
    step_started = Signal(str)
    finished = Signal(object, bool, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan: runners.Plan | None = None
        self.proc: QProcess | None = None
        self.index = 0
        self.stderr_all = ""
        self.stopped = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, plan: runners.Plan):
        self.plan, self.index, self.stderr_all, self.stopped = plan, 0, "", False
        self._next()

    def _next(self):
        step = self.plan.steps[self.index]
        self.step_started.emit(step.kind)
        proc = QProcess(self)
        env = QProcessEnvironment()
        for k, v in self.plan.env.items():
            env.insert(k, v)
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(self.plan.workdir))
        proc.setProgram(step.argv[0])
        proc.setArguments(step.argv[1:])
        self._dec_out = codecs.getincrementaldecoder("utf-8")("replace")
        self._dec_err = codecs.getincrementaldecoder("utf-8")("replace")
        proc.readyReadStandardOutput.connect(lambda: self._read(proc, "out"))
        proc.readyReadStandardError.connect(lambda: self._read(proc, "err"))
        proc.finished.connect(self._done)
        proc.errorOccurred.connect(self._error)
        self.proc = proc
        proc.start()

    def _read(self, proc, stream):
        if stream == "out":
            text = self._dec_out.decode(bytes(proc.readAllStandardOutput()))
        else:
            text = self._dec_err.decode(bytes(proc.readAllStandardError()))
            text = runners.clean_stderr(text) if text.startswith("Picked up") or "ExperimentalWarning" in text else text
            self.stderr_all += text
        if text:
            self.output.emit(text, stream)

    def _error(self, err):
        if err == QProcess.FailedToStart:
            self.output.emit(f"[failed to start: {self.proc.program()}]\n", "err")
            self._finish(-1, True, self.plan.steps[self.index].kind)

    def _done(self, code, status):
        proc = self.proc
        self._read(proc, "out")
        self._read(proc, "err")
        crashed = status == QProcess.CrashExit
        if runners.IS_WINDOWS and code < 0:
            code &= 0xFFFFFFFF
        step = self.plan.steps[self.index]
        if step.kind == "compile" and (code != 0 or crashed):
            self._finish(code, crashed, "compile")
            return
        self.index += 1
        if self.index < len(self.plan.steps) and not self.stopped:
            self._next()
        else:
            self._finish(code, crashed and not self.stopped, "run")

    def _finish(self, code, crashed, kind):
        plan = self.plan
        self.proc = None
        self.finished.emit(code, crashed, kind, self.stderr_all)
        if plan:
            plan.cleanup()

    def write(self, text: str):
        if self.running:
            self.proc.write(text.encode("utf-8"))

    def close_input(self):
        if self.running:
            self.proc.closeWriteChannel()

    def stop(self):
        if self.running:
            self.stopped = True
            self.proc.kill()


class Console(QWidget):
    """Read-only output area with an input line for programs that read input."""
    send_line = Signal(str)

    def __init__(self, tr, theme: str, font_size: int, parent=None):
        super().__init__(parent)
        self.tr, self.theme = tr, theme
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.view = QTextEdit(readOnly=True)
        self.view.setFont(mono_font(font_size))
        self.view.setLineWrapMode(QTextEdit.WidgetWidth)
        layout.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.input = QLineEdit(placeholderText=tr("label.stdin_placeholder"))
        self.input.setFont(mono_font(font_size))
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton(tr("btn.send"))
        self.send_btn.clicked.connect(self._send)
        self.clear_btn = QPushButton(tr("btn.clear"))
        self.clear_btn.clicked.connect(self.view.clear)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        row.addWidget(self.clear_btn)
        layout.addLayout(row)
        self.queued: list[str] = []

    def set_theme(self, theme: str):
        self.theme = theme

    def set_font_size(self, size: int):
        self.view.setFont(mono_font(size))
        self.input.setFont(mono_font(size))

    def _send(self):
        text = self.input.text()
        self.input.clear()
        self.write(text + "\n", "in")
        self.send_line.emit(text)

    def write(self, text: str, stream: str = "out"):
        c = COLORS[self.theme]
        fmt = QTextCharFormat()
        colour = {"out": c["text"], "err": c["bad"], "in": c["input"], "sys": c["muted"]}.get(stream, c["text"])
        fmt.setForeground(QColor(colour))
        if stream == "sys":
            fmt.setFontItalic(True)
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, fmt)
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

    def system(self, text: str):
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        if cursor.positionInBlock() != 0:
            self.write("\n", "out")
        self.write(text + "\n", "sys")

    def write_html(self, html_text: str):
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        if cursor.positionInBlock() != 0:
            cursor.insertText("\n")
        cursor.insertHtml(html_text)
        cursor.insertBlock()
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

    def explain(self, messages: list[str]):
        """Show plain-language explanations for an error."""
        if not messages:
            return
        c = COLORS[self.theme]
        r = Renderer(target="qt", theme=self.theme)
        body = "".join(f"<p>{r.inline(m).replace(chr(10), '<br>')}</p>" for m in messages)
        title = html.escape(self.tr("label.why"))
        family = html.escape(self.font().family())
        self.write_html(f'<table width="100%" cellpadding="8" style="background-color:{c["line"]};'
                        f'border:1px solid {c["warn"]};font-family:\'{family}\';font-style:normal">'
                        f'<tr><td><b>{title}</b>{body}</td></tr></table>')
