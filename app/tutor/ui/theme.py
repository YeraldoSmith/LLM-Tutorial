"""Light / dark appearance for the whole app."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from .. import highlight


def system_is_dark() -> bool:
    app = QApplication.instance()
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt < 6.5
        return app.palette().color(QPalette.Window).lightness() < 128


def resolve(choice: str) -> str:
    if choice in ("light", "dark"):
        return choice
    return "dark" if system_is_dark() else "light"


COLORS = {
    "light": {"bg": "#ffffff", "panel": "#f6f7f9", "text": "#1f2328", "muted": "#59636e", "accent": "#0969da",
              "border": "#d0d7de", "ok": "#1a7f37", "bad": "#cf222e", "warn": "#9a6700", "input": "#0a7c2f",
              "editor": "#ffffff", "line": "#f3f5f8", "gutter": "#8c959f", "select": "#cfe3ff"},
    "dark": {"bg": "#0d1117", "panel": "#161b22", "text": "#e6edf3", "muted": "#9198a1", "accent": "#4493f8",
             "border": "#30363d", "ok": "#3fb950", "bad": "#f85149", "warn": "#d29922", "input": "#7ee787",
             "editor": "#0d1117", "line": "#161b22", "gutter": "#6e7681", "select": "#1f4a7a"},
}


def apply(theme: str):
    app = QApplication.instance()
    app.setStyle("Fusion")
    c = COLORS[theme]
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["panel"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["bg"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["panel"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["panel"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Link, QColor(c["accent"]))
    pal.setColor(QPalette.ToolTipBase, QColor(c["panel"]))
    pal.setColor(QPalette.ToolTipText, QColor(c["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(c["muted"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(c["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QToolTip {{ color: {c['text']}; background: {c['panel']}; border: 1px solid {c['border']}; }}
        QPushButton {{ padding: 5px 12px; border-radius: 6px; border: 1px solid {c['border']}; }}
        QPushButton:hover {{ border-color: {c['accent']}; }}
        QPushButton#primary {{ background: {c['accent']}; color: white; border-color: {c['accent']}; font-weight: bold; }}
        QPushButton#primary:disabled {{ background: {c['border']}; border-color: {c['border']}; }}
        QPushButton#ok {{ background: {c['ok']}; color: white; border-color: {c['ok']}; font-weight: bold; }}
        QFrame#card {{ background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 8px; }}
        QLabel#muted {{ color: {c['muted']}; }}
        QLabel#title {{ font-size: 15px; font-weight: bold; }}
        QLabel#bar {{ padding: 4px 8px; border-radius: 4px; background: {c['line']}; }}
        QTreeWidget {{ border: none; }}
        QTabWidget::pane {{ border-top: 1px solid {c['border']}; }}
        QSplitter::handle {{ background: {c['border']}; }}
    """)


def lesson_css(theme: str, base_pt: int) -> str:
    c = COLORS[theme]
    mono = QFontDatabase.systemFont(QFontDatabase.FixedFont).family()
    return f"""
    body {{ color: {c['text']}; }}
    h1 {{ font-size: {base_pt + 10}pt; margin-top: 4px; margin-bottom: 10px; }}
    h2 {{ font-size: {base_pt + 5}pt; margin-top: 18px; margin-bottom: 6px; color: {c['accent']}; }}
    h3 {{ font-size: {base_pt + 2}pt; margin-top: 14px; margin-bottom: 4px; }}
    h4 {{ font-size: {base_pt + 1}pt; }}
    p, li {{ line-height: 150%; }}
    code {{ font-family: '{mono}'; background-color: {c['line']}; color: {highlight.PALETTE[theme]['pre']}; }}
    pre {{ font-family: '{mono}'; font-size: {base_pt - 1}pt; }}
    a {{ color: {c['accent']}; text-decoration: none; }}
    th {{ font-weight: bold; }}
    kbd {{ font-family: '{mono}'; background-color: {c['line']}; }}
    """ + highlight.css(theme)


def mono_font(size: int) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setPointSize(size)
    f.setStyleHint(QFont.Monospace)
    return f
