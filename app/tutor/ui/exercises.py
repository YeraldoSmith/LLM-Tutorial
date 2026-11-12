"""The exercises tab: one card per exercise (choice, predict-the-output, coding)."""
from __future__ import annotations

import html
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget)

from ..content import Exercise, check_output_answer
from ..markdown import Renderer
from .theme import COLORS, mono_font


def md_plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)


class ExerciseCard(QFrame):
    open_in_editor = Signal(object)      # Exercise
    show_solution = Signal(object)
    solved = Signal(object)

    def __init__(self, index: int, ex: Exercise, tr, theme: str, font_size: int, done: bool, renderer: Renderer):
        super().__init__()
        self.ex, self.tr, self.theme, self.renderer = ex, tr, theme, renderer
        self.setObjectName("card")
        self.hint_shown = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        head = QHBoxLayout()
        self.title = QLabel(html.escape(md_plain(ex.title)))
        self.title.setObjectName("title")
        self.title.setWordWrap(True)
        kind = QLabel(tr(f"label.kind.{ex.kind}"))
        kind.setObjectName("muted")
        self.done_label = QLabel("✅" if done else "")
        head.addWidget(self.title, 1)
        head.addWidget(kind)
        head.addWidget(self.done_label)
        lay.addLayout(head)

        prompt = self._label(renderer.render(ex.prompt_md).html if ex.prompt_md else "")
        lay.addWidget(prompt)

        if ex.kind == "choice":
            self.group = QButtonGroup(self)
            for i, opt in enumerate(ex.options_md):
                rb = QRadioButton(md_plain(opt))
                rb.setStyleSheet("QRadioButton { padding: 3px; }")
                self.group.addButton(rb, i)
                lay.addWidget(rb)
        elif ex.kind == "output":
            self.answer = QPlainTextEdit()
            self.answer.setPlaceholderText(tr("label.output_answer_placeholder"))
            self.answer.setFont(mono_font(font_size))
            self.answer.setFixedHeight(90)
            lay.addWidget(self.answer)

        row = QHBoxLayout()
        if ex.kind == "code":
            go = QPushButton(tr("btn.open_in_editor"))
            go.setObjectName("primary")
            go.clicked.connect(lambda: self.open_in_editor.emit(self.ex))
            row.addWidget(go)
        else:
            submit = QPushButton(tr("btn.submit"))
            submit.setObjectName("primary")
            submit.clicked.connect(self.submit)
            row.addWidget(submit)
        hint = QPushButton(tr("btn.hint"))
        hint.clicked.connect(self.next_hint)
        row.addWidget(hint)
        if ex.kind == "code" and ex.solution or ex.kind != "code":
            sol = QPushButton(tr("btn.solution"))
            sol.clicked.connect(self.reveal)
            row.addWidget(sol)
        row.addStretch(1)
        lay.addLayout(row)
        if ex.requires:
            req = QLabel(tr("label.requires", pkgs=", ".join(ex.requires)) + "  ·  " +
                         tr("label.pip_hint", pkgs=" ".join(ex.requires)))
            req.setObjectName("muted")
            req.setWordWrap(True)
            req.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(req)
        self.hints = QLabel()
        self.hints.setWordWrap(True)
        self.hints.setTextFormat(Qt.RichText)
        self.hints.hide()
        lay.addWidget(self.hints)
        self.result = QLabel()
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.RichText)
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self.result.hide()
        lay.addWidget(self.result)

    def _label(self, rich: str) -> QLabel:
        label = QLabel(rich)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        return label

    def next_hint(self):
        if self.hint_shown < len(self.ex.hints_md):
            self.hint_shown += 1
        parts = [f"<p><b>💡 {html.escape(self.tr('label.hint_n', n=i + 1))}</b></p>{self.renderer.render(h).html}"
                 for i, h in enumerate(self.ex.hints_md[: self.hint_shown])]
        if self.hint_shown >= len(self.ex.hints_md):
            parts.append(f"<p><i>{html.escape(self.tr('label.no_more_hints'))}</i></p>")
        self.hints.setText("".join(parts))
        self.hints.show()

    def submit(self):
        c = COLORS[self.theme]
        if self.ex.kind == "choice":
            chosen = self.group.checkedId()
            ok = chosen >= 0 and chosen == int(self.ex.answer if self.ex.answer is not None else -1)
        else:
            ok = check_output_answer(self.answer.toPlainText(), str(self.ex.answer or ""))
        text = self.tr("label.correct") if ok else self.tr("label.wrong")
        colour = c["ok"] if ok else c["bad"]
        extra = ""
        if ok and self.ex.explain_md:
            extra = f"<p><b>{html.escape(self.tr('label.explanation'))}</b></p>" + self.renderer.render(self.ex.explain_md).html
        self.result.setText(f'<p style="color:{colour}"><b>{html.escape(text)}</b></p>{extra}')
        self.result.show()
        if ok:
            self.mark_done()

    def mark_done(self):
        self.done_label.setText("✅")
        self.solved.emit(self.ex)

    def reveal(self):
        if QMessageBox.question(self, self.tr("btn.solution"), self.tr("label.solution_confirm")) != QMessageBox.Yes:
            return
        parts = []
        if self.ex.kind == "choice" and self.ex.answer is not None and int(self.ex.answer) < len(self.ex.options_md):
            parts.append(f"<p><b>✔ {html.escape(md_plain(self.ex.options_md[int(self.ex.answer)]))}</b></p>")
        elif self.ex.kind == "output":
            parts.append(self.renderer.render("```text\n" + str(self.ex.answer or "").rstrip("\n") + "\n```").html)
        elif self.ex.kind == "code":
            self.show_solution.emit(self.ex)
            parts.append(self.renderer.render(f"```{self.ex.language}\n{self.ex.solution.rstrip()}\n```").html)
        if self.ex.explain_md:
            parts.append(f"<p><b>{html.escape(self.tr('label.explanation'))}</b></p>" + self.renderer.render(self.ex.explain_md).html)
        self.result.setText("".join(parts))
        self.result.show()

    def show_report(self, rich: str):
        self.result.setText(rich)
        self.result.show()


class ExercisePanel(QScrollArea):
    open_in_editor = Signal(object)
    show_solution = Signal(object)
    solved = Signal(object)

    def __init__(self, tr, parent=None):
        super().__init__(parent)
        self.tr = tr
        self.setWidgetResizable(True)
        self.cards: dict[str, ExerciseCard] = {}

    def load(self, exercises: list[Exercise], theme: str, font_size: int, done_keys: set[str], renderer: Renderer):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)
        self.cards = {}
        if not exercises:
            empty = QLabel(self.tr("label.no_exercises"))
            empty.setWordWrap(True)
            empty.setObjectName("muted")
            lay.addWidget(empty)
        for i, ex in enumerate(exercises, 1):
            card = ExerciseCard(i, ex, self.tr, theme, font_size, ex.id in done_keys, renderer)
            card.open_in_editor.connect(self.open_in_editor)
            card.show_solution.connect(self.show_solution)
            card.solved.connect(self.solved)
            self.cards[ex.id] = card
            lay.addWidget(card)
        lay.addStretch(1)
        self.setWidget(inner)
