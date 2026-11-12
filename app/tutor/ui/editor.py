"""Code editor: line numbers, syntax colours, auto-indent, comment toggling."""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QSyntaxHighlighter, QTextCharFormat, QTextFormat
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget

from .. import highlight
from .theme import COLORS, mono_font

COMMENT = {"python": "# ", "shell": "# ", "asm": "# ", "sql": "-- "}
BLOCK_OPENERS = (":", "{", "(", "[")


class Highlighter(QSyntaxHighlighter):
    def __init__(self, document, lang: str, theme: str):
        super().__init__(document)
        self.lang = lang
        self.set_theme(theme)

    def set_theme(self, theme: str):
        self.formats = {}
        for kind, colour in highlight.PALETTE[theme].items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(colour))
            if kind == "com":
                fmt.setFontItalic(True)
            if kind == "kw":
                fmt.setFontWeight(QFont.DemiBold)
            self.formats[kind] = fmt
        self.rehighlight()

    def set_language(self, lang: str):
        self.lang = lang
        self.rehighlight()

    # Qt keeps an int per line; map our string states to ints.
    STATES = {None: -1, "block": 1, 'str:"""': 2, "str:'''": 3, "str:`": 4}
    REVERSE = {v: k for k, v in STATES.items()}

    def highlightBlock(self, text: str):
        prev = self.REVERSE.get(self.previousBlockState())
        tokens, state = highlight.tokenize_line(self.lang, text, prev)
        pos = 0
        for kind, chunk in tokens:
            if kind != "plain" and kind in self.formats:
                self.setFormat(pos, len(chunk), self.formats[kind])
            pos += len(chunk)
        self.setCurrentBlockState(self.STATES.get(state, -1))


class LineNumbers(QWidget):
    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event):
        self.editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self, theme: str = "light", font_size: int = 12, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.lang = "python"
        self.setFont(mono_font(font_size))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabChangesFocus(False)
        self.gutter = LineNumbers(self)
        self.highlighter = Highlighter(self.document(), self.lang, theme)
        self.blockCountChanged.connect(self._update_margin)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_line)
        self._update_margin()
        self._highlight_line()
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)

    # -------------------------------------------------------------- settings
    def set_language(self, lang: str):
        self.lang = lang
        self.highlighter.set_language(lang)

    def set_theme(self, theme: str):
        self.theme = theme
        self.highlighter.set_theme(theme)
        self._highlight_line()
        self.gutter.update()

    def set_font_size(self, size: int):
        self.setFont(mono_font(size))
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self._update_margin()

    def indent_unit(self) -> str:
        return "\t" if self.lang == "go" else "    "

    # --------------------------------------------------------------- gutter
    def gutter_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margin(self, *_):
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event):
        c = COLORS[self.theme]
        painter = QPainter(self.gutter)
        painter.fillRect(event.rect(), QColor(c["panel"]))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(c["text"] if number == current else c["gutter"]))
                painter.drawText(0, top, self.gutter.width() - 6, self.fontMetrics().height(),
                                 Qt.AlignRight, str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def _highlight_line(self):
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor(COLORS[self.theme]["line"]))
        sel.format.setProperty(QTextFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])

    # ------------------------------------------------------------- editing
    def keyPressEvent(self, event):
        cursor = self.textCursor()
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
            line = cursor.block().text()[: cursor.positionInBlock()]
            indent = line[: len(line) - len(line.lstrip())]
            stripped = line.rstrip()
            if stripped.endswith(BLOCK_OPENERS):
                indent += self.indent_unit()
            cursor.insertText("\n" + indent)
            # typing "{" + Enter between braces: put the "}" on its own line
            after = cursor.block().text()[cursor.positionInBlock():]
            if stripped.endswith("{") and after.lstrip().startswith("}"):
                pos = cursor.position()
                cursor.insertText("\n" + indent[: len(indent) - len(self.indent_unit())])
                cursor.setPosition(pos)
                self.setTextCursor(cursor)
            return
        if key == Qt.Key_Tab and not cursor.hasSelection():
            cursor.insertText(self.indent_unit())
            return
        if key in (Qt.Key_Tab, Qt.Key_Backtab) and cursor.hasSelection():
            self._shift_lines(-1 if key == Qt.Key_Backtab else 1)
            return
        if key == Qt.Key_Backtab:
            self._shift_lines(-1)
            return
        if key == Qt.Key_Backspace and not cursor.hasSelection():
            line = cursor.block().text()[: cursor.positionInBlock()]
            unit = self.indent_unit()
            if line and line.strip() == "" and line.endswith(unit) and unit != "\t":
                for _ in range(len(unit)):
                    cursor.deletePreviousChar()
                return
        if key == Qt.Key_BraceRight:
            line = cursor.block().text()[: cursor.positionInBlock()]
            unit = self.indent_unit()
            if line.strip() == "" and line.endswith(unit):
                for _ in range(len(unit)):
                    cursor.deletePreviousChar()
        if event.matches(QKeySequence.StandardKey.Find):
            return super().keyPressEvent(event)
        if key == Qt.Key_Slash and event.modifiers() & Qt.ControlModifier:
            self.toggle_comment()
            return
        super().keyPressEvent(event)

    def _selected_blocks(self):
        cursor = self.textCursor()
        doc = self.document()
        start = doc.findBlock(cursor.selectionStart())
        end = doc.findBlock(cursor.selectionEnd())
        if cursor.hasSelection() and cursor.selectionEnd() == end.position() and end != start:
            end = end.previous()
        blocks = []
        b = start
        while b.isValid():
            blocks.append(b)
            if b == end:
                break
            b = b.next()
        return blocks

    def _shift_lines(self, direction: int):
        unit = self.indent_unit()
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for block in self._selected_blocks():
            c = self.textCursor()
            c.setPosition(block.position())
            if direction > 0:
                c.insertText(unit)
            else:
                text = block.text()
                n = len(unit) if text.startswith(unit) else len(text) - len(text.lstrip(" \t")) if text[:1] in " \t" else 0
                n = min(n, len(unit))
                for _ in range(n):
                    c.deleteChar()
        cursor.endEditBlock()

    def toggle_comment(self):
        marker = COMMENT.get(self.lang, "// ")
        blocks = self._selected_blocks()
        all_commented = all(b.text().lstrip().startswith(marker.strip()) for b in blocks if b.text().strip())
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for block in blocks:
            text = block.text()
            if not text.strip():
                continue
            indent = len(text) - len(text.lstrip())
            c = self.textCursor()
            c.setPosition(block.position() + indent)
            if all_commented:
                n = len(marker) if text.lstrip().startswith(marker) else len(marker.strip())
                for _ in range(n):
                    c.deleteChar()
            else:
                c.insertText(marker)
        cursor.endEditBlock()
