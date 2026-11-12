"""Markdown -> HTML for lessons (no third-party dependencies).

Supports the subset the lessons use, which is also what GitHub shows:
headings, paragraphs, **bold**, *italic*, `code`, links, images, fenced code
blocks, block quotes, GitHub alerts (> [!TIP]), nested lists, tables,
horizontal rules, <details> and a few inline HTML tags (<kbd>, <br>, <sup>...).

Two output targets:
  * "web" – semantic HTML with classes (the website styles it);
  * "qt"  – HTML that QTextBrowser can display well (tables instead of
            boxes, links such as ``run:3`` for "load this example").
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable

from . import highlight

ALERTS = {
    "NOTE": ("ℹ️", "note"), "TIP": ("💡", "tip"), "IMPORTANT": ("⭐", "important"),
    "WARNING": ("⚠️", "warning"), "CAUTION": ("🛑", "caution"),
}
DEFAULT_LABELS = {
    "note": "Note", "tip": "Tip", "important": "Important", "warning": "Warning", "caution": "Caution",
    "run": "Try it", "copy": "Copy", "hint": "Hint", "formula": "Formula",
}
INLINE_TAGS = ("kbd", "br", "sup", "sub", "mark", "small", "u", "b", "i", "em", "strong", "s")
RUNNABLE = {"python", "c", "cpp", "java", "rust", "go", "javascript", "typescript", "asm"}

QT_COLORS = {
    "light": {"code_bg": "#f6f8fa", "code_border": "#d0d7de", "note": "#ddf4ff", "tip": "#dafbe1",
              "important": "#fbefff", "warning": "#fff8c5", "caution": "#ffebe9", "formula": "#f3f0ff",
              "head": "#57606a", "table_head": "#eef1f4", "border": "#d0d7de"},
    "dark": {"code_bg": "#161b22", "code_border": "#30363d", "note": "#0c2d4a", "tip": "#0f2e1b",
             "important": "#2c1d3f", "warning": "#3a2f0b", "caution": "#3d1517", "formula": "#221d3a",
             "head": "#8b949e", "table_head": "#21262d", "border": "#30363d"},
}


@dataclass
class CodeBlock:
    lang: str
    code: str
    info: str = ""
    runnable: bool = False


@dataclass
class Rendered:
    html: str
    code_blocks: list[CodeBlock] = field(default_factory=list)
    headings: list[tuple[int, str, str]] = field(default_factory=list)  # (level, text, id)


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text) or "section"


class Renderer:
    def __init__(self, target: str = "web", labels: dict | None = None,
                 link_resolver: Callable[[str], str] | None = None,
                 image_resolver: Callable[[str], str] | None = None,
                 theme: str = "light", default_lang: str | None = None):
        self.target = target
        self.labels = {**DEFAULT_LABELS, **(labels or {})}
        self.link_resolver = link_resolver or (lambda href: href)
        self.image_resolver = image_resolver or (lambda src: src)
        self.theme = theme
        self.default_lang = default_lang
        self.code_blocks: list[CodeBlock] = []
        self.headings: list[tuple[int, str, str]] = []
        self._ids: dict[str, int] = {}

    # ------------------------------------------------------------------ API
    def render(self, text: str) -> Rendered:
        self.code_blocks, self.headings, self._ids = [], [], {}
        text = text.replace("\r\n", "\n").replace("\t", "    ")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        body = self._blocks(text.split("\n"))
        return Rendered(body, self.code_blocks, self.headings)

    # --------------------------------------------------------------- blocks
    def _blocks(self, lines: list[str]) -> str:
        out: list[str] = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            # fenced code
            m = re.match(r"^(\s*)(`{3,}|~{3,})\s*(.*)$", line)
            if m:
                indent, fence, info = len(m.group(1)), m.group(2), m.group(3).strip()
                code_lines = []
                i += 1
                while i < n and not re.match(r"^\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", lines[i]):
                    ln = lines[i]
                    code_lines.append(ln[indent:] if ln[:indent].strip() == "" else ln.lstrip())
                    i += 1
                i += 1
                out.append(self._code("\n".join(code_lines), info))
                continue
            # heading
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                level = len(m.group(1))
                out.append(self._heading(level, m.group(2)))
                i += 1
                continue
            # horizontal rule
            if re.match(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$", line):
                out.append("<hr>")
                i += 1
                continue
            # block quote
            if stripped.startswith(">"):
                quote = []
                while i < n and lines[i].strip().startswith(">"):
                    quote.append(re.sub(r"^\s*> ?", "", lines[i]))
                    i += 1
                out.append(self._quote(quote))
                continue
            # table
            if "|" in line and i + 1 < n and re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", lines[i + 1]):
                rows = [line, lines[i + 1]]
                i += 2
                while i < n and "|" in lines[i] and lines[i].strip():
                    rows.append(lines[i])
                    i += 1
                out.append(self._table(rows))
                continue
            # list
            if re.match(r"^\s*([-*+]|\d{1,3}[.)])\s+", line):
                block = []
                base = len(line) - len(line.lstrip())
                while i < n:
                    ln = lines[i]
                    if not ln.strip():
                        # blank line: list continues if next non-blank is indented or another item
                        j = i + 1
                        while j < n and not lines[j].strip():
                            j += 1
                        if j < n and (len(lines[j]) - len(lines[j].lstrip()) > base
                                      or re.match(r"^\s{%d}([-*+]|\d{1,3}[.)])\s+" % base, lines[j])):
                            block.append("")
                            i += 1
                            continue
                        break
                    ind = len(ln) - len(ln.lstrip())
                    if ind < base:
                        break
                    if ind == base and not re.match(r"^\s*([-*+]|\d{1,3}[.)])\s+", ln):
                        # lazy continuation of a paragraph inside the item
                        if block and block[-1].strip():
                            block.append(ln)
                            i += 1
                            continue
                        break
                    block.append(ln)
                    i += 1
                out.append(self._list(block, base))
                continue
            # raw HTML block (details/summary/div)
            if re.match(r"^\s*</?(details|summary|div|p|table|br)\b", stripped, re.I):
                out.append(self._html_line(stripped))
                i += 1
                continue
            # paragraph
            para = []
            while i < n and lines[i].strip() and not self._starts_block(lines, i):
                para.append(lines[i].strip())
                i += 1
            if not para:  # safety: never loop forever
                para.append(lines[i].strip())
                i += 1
            out.append("<p>" + self._inline_join(para) + "</p>")
        return "\n".join(out)

    def _starts_block(self, lines, i) -> bool:
        line = lines[i]
        s = line.strip()
        if re.match(r"^(\s*)(`{3,}|~{3,})", line) or re.match(r"^#{1,6}\s", line) or s.startswith(">"):
            return True
        if re.match(r"^\s*([-*+]|\d{1,3}[.)])\s+", line):
            return True
        if re.match(r"^\s*</?(details|summary|div)\b", s, re.I):
            return True
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", lines[i + 1]):
            return True
        return False

    def _inline_join(self, para: list[str]) -> str:
        parts = []
        for k, p in enumerate(para):
            hard = p.endswith("\\")
            p = p[:-1] if hard else p
            parts.append(self.inline(p) + ("<br>" if hard and k < len(para) - 1 else ""))
        return "\n".join(parts)

    def _html_line(self, s: str) -> str:
        m = re.match(r"^<summary>(.*)</summary>$", s, re.I)
        if m:
            return "<summary>" + self.inline(m.group(1)) + "</summary>"
        m = re.match(r"^<(/?)(details|summary|div|p|br)(\s+open)?\s*/?>$", s, re.I)
        if m:
            return f"<{m.group(1)}{m.group(2).lower()}{' open' if m.group(3) else ''}>"
        return "<p>" + self.inline(s) + "</p>"

    def _heading(self, level: int, text: str) -> str:
        inner = self.inline(text)
        slug = slugify(text)
        if slug in self._ids:
            self._ids[slug] += 1
            slug = f"{slug}-{self._ids[slug]}"
        else:
            self._ids[slug] = 0
        self.headings.append((level, re.sub(r"<[^>]+>", "", inner), slug))
        if self.target == "qt":
            return f'<a name="{slug}"></a><h{level}>{inner}</h{level}>'
        return f'<h{level} id="{html.escape(slug)}">{inner}</h{level}>'

    def _code(self, code: str, info: str) -> str:
        words = info.split()
        lang_raw = words[0] if words else (self.default_lang or "text")
        lang = highlight.normalize(lang_raw)
        flags = set(w.lower() for w in words[1:])
        title = ""
        m = re.search(r'title="([^"]+)"', info)
        if m:
            title = m.group(1)
        if lang_raw.lower() == "formula":
            return self._formula(code)
        runnable = lang in RUNNABLE and "norun" not in flags and "no-run" not in flags
        idx = len(self.code_blocks)
        self.code_blocks.append(CodeBlock(lang, code, info, runnable))
        body = highlight.to_html(code, lang)
        label = title or (lang_raw if lang_raw not in ("text", "") else "")
        if self.target == "qt":
            c = QT_COLORS[self.theme]
            head = f'<span style="color:{c["head"]}">{html.escape(label)}</span>' if label else ""
            run = (f'&nbsp;&nbsp;<a href="run:{idx}">▶ {html.escape(self.labels["run"])}</a>' if runnable else "")
            pre = f'<pre style="margin:0">{body}</pre>'
            top = f'<tr><td style="padding:4px 8px 0 8px">{head}{run}</td></tr>' if (head or run) else ""
            return (f'<table width="100%" cellspacing="0" cellpadding="8" '
                    f'style="background-color:{c["code_bg"]};border:1px solid {c["code_border"]};margin:6px 0">'
                    f'{top}<tr><td>{pre}</td></tr></table>')
        head = ""
        if label or runnable:
            btn = (f'<button type="button" class="code-run" data-run="{idx}">▶ {html.escape(self.labels["run"])}</button>'
                   if runnable else "")
            head = (f'<div class="code-head"><span class="code-lang">{html.escape(label)}</span>'
                    f'<span class="code-actions"><button type="button" class="code-copy" data-copy="{idx}">'
                    f'{html.escape(self.labels["copy"])}</button>{btn}</span></div>')
        return (f'<div class="code" data-lang="{lang}" data-index="{idx}">{head}'
                f'<pre><code class="lang-{lang}">{body}</code></pre></div>')

    def _formula(self, code: str) -> str:
        lines = "<br>".join(html.escape(l) for l in code.strip("\n").split("\n"))
        if self.target == "qt":
            c = QT_COLORS[self.theme]
            return (f'<table width="100%" cellpadding="10" style="background-color:{c["formula"]};margin:6px 0">'
                    f'<tr><td align="center"><span style="font-family:serif;font-size:large">{lines}</span></td></tr></table>')
        return f'<div class="formula" role="math">{lines}</div>'

    def _quote(self, lines: list[str]) -> str:
        first = lines[0].strip() if lines else ""
        m = re.match(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$", first, re.I)
        if m:
            icon, kind = ALERTS[m.group(1).upper()]
            rest = ([m.group(2)] if m.group(2) else []) + lines[1:]
            inner = self._blocks(rest)
            title = f"{icon} {html.escape(self.labels[kind])}"
            if self.target == "qt":
                c = QT_COLORS[self.theme]
                return (f'<table width="100%" cellpadding="10" style="background-color:{c[kind]};margin:6px 0">'
                        f'<tr><td><b>{title}</b>{inner}</td></tr></table>')
            return f'<div class="callout callout-{kind}"><p class="callout-title">{title}</p>{inner}</div>'
        inner = self._blocks(lines)
        if self.target == "qt":
            c = QT_COLORS[self.theme]
            return (f'<table width="100%" cellpadding="8" style="margin:6px 0"><tr>'
                    f'<td width="4" style="background-color:{c["border"]}"></td><td>{inner}</td></tr></table>')
        return f"<blockquote>{inner}</blockquote>"

    def _table(self, rows: list[str]) -> str:
        def cells(row):
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|") and not row.endswith("\\|"):
                row = row[:-1]
            parts = re.split(r"(?<!\\)\|", row)
            return [p.strip().replace("\\|", "|") for p in parts]
        head = cells(rows[0])
        aligns = []
        for spec in cells(rows[1]):
            if spec.startswith(":") and spec.endswith(":"):
                aligns.append("center")
            elif spec.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("left")
        body = [cells(r) for r in rows[2:]]
        if self.target == "qt":
            c = QT_COLORS[self.theme]
            h = "".join(f'<th align="{aligns[k] if k < len(aligns) else "left"}" style="background-color:{c["table_head"]}">'
                        f"{self.inline(x)}</th>" for k, x in enumerate(head))
            b = "".join("<tr>" + "".join(f'<td align="{aligns[k] if k < len(aligns) else "left"}">{self.inline(x)}</td>'
                                         for k, x in enumerate(r)) + "</tr>" for r in body)
            return (f'<table cellspacing="0" cellpadding="6" border="1" style="border-color:{c["border"]};'
                    f'border-collapse:collapse;margin:6px 0"><tr>{h}</tr>{b}</table>')
        h = "".join(f'<th style="text-align:{aligns[k] if k < len(aligns) else "left"}">{self.inline(x)}</th>'
                    for k, x in enumerate(head))
        b = "".join("<tr>" + "".join(f'<td style="text-align:{aligns[k] if k < len(aligns) else "left"}">{self.inline(x)}</td>'
                                     for k, x in enumerate(r)) + "</tr>" for r in body)
        return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'

    def _list(self, lines: list[str], base: int) -> str:
        first = lines[0]
        ordered = bool(re.match(r"^\s*\d{1,3}[.)]\s+", first))
        start = int(re.match(r"^\s*(\d+)", first).group(1)) if ordered else 1
        items: list[list[str]] = []
        content_indent = base + 2
        loose = False
        for ln in lines:
            m = re.match(r"^(\s*)([-*+]|\d{1,3}[.)])(\s+)(.*)$", ln)
            if m and len(m.group(1)) == base:
                content_indent = base + len(m.group(2)) + len(m.group(3))
                items.append([m.group(4)])
            elif items:
                if ln.strip() == "":
                    items[-1].append("")
                else:
                    ind = len(ln) - len(ln.lstrip())
                    items[-1].append(ln[min(ind, content_indent):] if ind >= base else ln.strip())
        for it in items:
            # an item is "loose" if it has a blank line between paragraphs
            core = it[:]
            while core and core[-1] == "":
                core.pop()
            if "" in core:
                loose = True
        parts = []
        for it in items:
            while it and it[-1] == "":
                it.pop()
            # task list
            task = ""
            if it and re.match(r"^\[( |x|X)\]\s", it[0]):
                checked = it[0][1] in "xX"
                task = ("☑ " if checked else "☐ ")
                it[0] = it[0][4:]
            if not loose and not any(re.match(r"^\s*(```|~~~|>|\||([-*+]|\d{1,3}[.)])\s)", x) for x in it[1:]) and "" not in it:
                inner = self._inline_join([x.strip() for x in it])
            elif not loose:
                # first line is text, rest are nested blocks
                head_lines = []
                k = 0
                while k < len(it) and it[k].strip() and not (k > 0 and re.match(r"^\s*(```|~~~|>|([-*+]|\d{1,3}[.)])\s)", it[k])):
                    head_lines.append(it[k].strip())
                    k += 1
                inner = self._inline_join(head_lines) + self._blocks(it[k:])
            else:
                inner = self._blocks(it)
            parts.append(f"<li>{task}{inner}</li>")
        tag = "ol" if ordered else "ul"
        attr = f' start="{start}"' if ordered and start != 1 else ""
        return f"<{tag}{attr}>" + "".join(parts) + f"</{tag}>"

    # --------------------------------------------------------------- inline
    def inline(self, text: str) -> str:
        stash: list[str] = []

        def keep(s: str) -> str:
            stash.append(s)
            return f"\x00{len(stash) - 1}\x00"

        # code spans
        def code_span(m):
            body = m.group(2)
            if body.startswith(" ") and body.endswith(" ") and body.strip():
                body = body[1:-1]
            return keep("<code>" + html.escape(body, quote=False) + "</code>")
        text = re.sub(r"(`+)(.+?)\1", code_span, text)
        # escaped characters
        text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~])", lambda m: keep(html.escape(m.group(1))), text)
        # allowed inline html tags
        tag_re = r"</?(?:%s)\s*/?>" % "|".join(INLINE_TAGS)
        text = re.sub(tag_re, lambda m: keep(m.group(0).lower().replace(" ", "")), text, flags=re.I)
        # images
        def image(m):
            alt, src = m.group(1), m.group(2).split()[0]
            src = self.image_resolver(src)
            return keep(f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">')
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, text)
        # links
        def link(m):
            label, href = m.group(1), m.group(2).strip().split()[0]
            href = self.link_resolver(href)
            ext = href.startswith(("http://", "https://"))
            attrs = ' target="_blank" rel="noopener"' if ext and self.target == "web" else ""
            return keep(f'<a href="{html.escape(href)}"{attrs}>') + label + keep("</a>")
        text = re.sub(r"\[([^\]]+)\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)", link, text)
        # autolinks
        blank = ' target="_blank" rel="noopener"' if self.target == "web" else ""
        text = re.sub(r"<(https?://[^>\s]+)>", lambda m: keep(
            f'<a href="{html.escape(m.group(1))}"{blank}>{html.escape(m.group(1))}</a>'), text)
        text = html.escape(text, quote=False)
        # emphasis
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])__(?=\S)(.+?)(?<=\S)__(?![\w*])", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"<em>\1</em>", text)
        text = re.sub(r"(?<![\w])_(?=\S)(.+?)(?<=\S)_(?![\w])", r"<em>\1</em>", text)
        text = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"<del>\1</del>", text)
        # restore (repeat because stashed links contain stashed parts)
        for _ in range(3):
            text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
        return text


def render(text: str, **kwargs) -> Rendered:
    return Renderer(**kwargs).render(text)


def plain_text(md: str) -> str:
    """Rough text extraction (used for search and summaries)."""
    md = re.sub(r"```.*?```", " ", md, flags=re.S)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"[#>*_`|]", "", md)
    return re.sub(r"\s+", " ", md).strip()
