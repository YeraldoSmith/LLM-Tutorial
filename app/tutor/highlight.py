"""Small, dependency-free syntax highlighter.

Used in three places, so the colours are identical everywhere:
  * the lesson renderer (markdown.py) turns code blocks into coloured HTML;
  * the desktop editor (ui/editor.py) colours what the learner types;
  * the website build script pre-renders lessons with the same output.

The tokenizer works line by line and carries a small ``state`` between lines,
which is what Qt's QSyntaxHighlighter needs for multi-line strings/comments.
Token kinds: kw (keyword), ty (type / builtin), str, num, com (comment),
fn (function name), pre (preprocessor, decorator, macro, directive),
reg (assembler register), plain.
"""
from __future__ import annotations

import html
import re

ALIASES = {
    "py": "python", "python3": "python", "py3": "python", "pycon": "python",
    "h": "c", "c17": "c", "c23": "c",
    "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hxx": "cpp",
    "rs": "rust", "golang": "go",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript", "node": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "s": "asm", "gas": "asm", "nasm": "asm", "x86asm": "asm", "armasm": "asm", "assembly": "asm",
    "sh": "shell", "bash": "shell", "zsh": "shell", "console": "shell", "terminal": "shell",
    "powershell": "shell", "ps1": "shell", "pwsh": "shell", "cmd": "shell", "bat": "shell", "fish": "shell",
    "jsonc": "json", "yaml": "conf", "yml": "conf", "toml": "conf", "ini": "conf", "dockerfile": "shell",
    "txt": "text", "plaintext": "text", "output": "text", "formula": "text", "diff": "text",
}


def normalize(lang: str | None) -> str:
    lang = (lang or "").strip().lower()
    return ALIASES.get(lang, lang or "text")


def _words(s: str) -> frozenset[str]:
    return frozenset(s.split())


C_KW = _words("""auto break case char const continue default do double else enum extern float for goto if
inline int long register restrict return short signed sizeof static struct switch typedef union unsigned
void volatile while _Bool bool true false nullptr constexpr typeof typeof_unqual alignas alignof
static_assert thread_local _Atomic _Generic _Noreturn _Static_assert""")
C_TY = _words("""size_t ssize_t ptrdiff_t intptr_t uintptr_t int8_t int16_t int32_t int64_t uint8_t uint16_t
uint32_t uint64_t FILE NULL EOF stdin stdout stderr printf scanf puts putchar getchar malloc free calloc
realloc strlen strcpy strcmp memcpy memset fprintf fgets exit""")
CPP_KW = C_KW | _words("""and and_eq asm bitand bitor catch char8_t char16_t char32_t class co_await
co_return co_yield compl concept consteval constinit const_cast decltype delete dynamic_cast explicit
export friend module import mutable namespace new noexcept not not_eq operator or or_eq private
protected public reinterpret_cast requires static_cast template this throw try typeid typename using
virtual xor xor_eq override final""")
CPP_TY = C_TY | _words("""std string string_view vector array map unordered_map set unordered_set
pair tuple optional variant expected span unique_ptr shared_ptr weak_ptr make_unique make_shared
cout cin cerr endl print println format ranges views size_type iterator move forward""")
JAVA_KW = _words("""abstract assert boolean break byte case catch char class const continue default do
double else enum extends final finally float for goto if implements import instanceof int interface
long native new package private protected public return short static strictfp super switch
synchronized this throw throws transient try void volatile while var record sealed permits non-sealed
yield true false null module when""")
JAVA_TY = _words("""String System Object Integer Long Double Boolean Character Math List ArrayList Map
HashMap Set HashSet Scanner IO Optional Stream Collectors Arrays Collections Exception
RuntimeException Thread Runnable Record StringBuilder LocalDate Files Path Paths""")
RUST_KW = _words("""as async await break const continue crate dyn else enum extern false fn for if impl
in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe
use where while gen""")
RUST_TY = _words("""i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str String
Vec Option Some None Result Ok Err Box Rc Arc RefCell Cell HashMap HashSet BTreeMap VecDeque
Iterator IntoIterator Debug Clone Copy Display Default PartialEq Eq PartialOrd Ord Hash""")
GO_KW = _words("""break case chan const continue default defer else fallthrough for func go goto if
import interface map package range return select struct switch type var""")
GO_TY = _words("""bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune
string uint uint8 uint16 uint32 uint64 uintptr true false iota nil any comparable append cap clear
close copy delete len make max min new panic print println recover fmt""")
JS_KW = _words("""break case catch class const continue debugger default delete do else export
extends false finally for function if import in instanceof let new null return super switch this
throw true try typeof undefined var void while with yield async await of static get set""")
JS_TY = _words("""console Math JSON Object Array String Number Boolean Promise Map Set Date Error
RegExp Symbol BigInt globalThis window document process require module exports""")
TS_KW = JS_KW | _words("""type interface enum implements private protected public readonly abstract
declare namespace keyof infer is as satisfies unique asserts override accessor""")
TS_TY = JS_TY | _words("""unknown never any number string boolean bigint symbol object Record Partial
Required Readonly Pick Omit ReturnType Awaited""")
PY_KW = _words("""False None True and as assert async await break class continue def del elif else
except finally for from global if import in is lambda nonlocal not or pass raise return try while
with yield match case type""")
PY_TY = _words("""print input len range int float str list dict set tuple bool open enumerate zip map
filter sorted reversed sum min max abs round isinstance issubclass super object Exception ValueError
TypeError KeyError IndexError ZeroDivisionError NameError AttributeError RuntimeError StopIteration
self cls any all iter next hasattr getattr setattr id hash repr format chr ord bin hex oct divmod pow
bytes bytearray frozenset property staticmethod classmethod dataclass""")
SQL_KW = _words("""select from where insert into values update set delete create table drop alter add
primary key foreign references not null unique default index view join inner left right outer full
on as and or in is like between order by group having limit offset distinct union all case when then
else end count sum avg min max integer text real blob varchar char boolean date with returning""")
ASM_DIRECTIVE = re.compile(r"\.[A-Za-z_][\w.]*")
ASM_REG = re.compile(
    r"(?:[re]?(?:ax|bx|cx|dx|si|di|sp|bp|ip)|[abcd][lh]|[sd]il|[sb]pl|r(?:[89]|1[0-5])[dwb]?"
    r"|[xyz]mm\d{1,2}|cr\d|[xw](?:[12]?\d|30)|sp|lr|fp|pc|xzr|wzr|[qdsbhv]\d{1,2})$",
    re.I,
)

LANGS = {
    "python": dict(kw=PY_KW, ty=PY_TY, line=("#",), block=None, quotes="'\"", triple=True, deco="@"),
    "c": dict(kw=C_KW, ty=C_TY, line=("//",), block=("/*", "*/"), quotes="'\"", pp=True),
    "cpp": dict(kw=CPP_KW, ty=CPP_TY, line=("//",), block=("/*", "*/"), quotes="'\"", pp=True),
    "java": dict(kw=JAVA_KW, ty=JAVA_TY, line=("//",), block=("/*", "*/"), quotes="'\"", triple_dq=True, deco="@"),
    "rust": dict(kw=RUST_KW, ty=RUST_TY, line=("//",), block=("/*", "*/"), quotes='"', rust=True, deco="#"),
    "go": dict(kw=GO_KW, ty=GO_TY, line=("//",), block=("/*", "*/"), quotes="'\"`"),
    "javascript": dict(kw=JS_KW, ty=JS_TY, line=("//",), block=("/*", "*/"), quotes="'\"`"),
    "typescript": dict(kw=TS_KW, ty=TS_TY, line=("//",), block=("/*", "*/"), quotes="'\"`", deco="@"),
    "shell": dict(kw=_words("if then else elif fi for in do done while case esac function return export "
                            "sudo cd ls mkdir echo cat rm cp mv pip pip3 python python3 uv git"),
                  ty=frozenset(), line=("#",), block=None, quotes="'\""),
    "sql": dict(kw=SQL_KW, ty=frozenset(), line=("--",), block=("/*", "*/"), quotes="'", nocase=True),
    "json": dict(kw=_words("true false null"), ty=frozenset(), line=(), block=None, quotes='"'),
    "conf": dict(kw=_words("true false null yes no on off"), ty=frozenset(), line=("#", ";"), block=None, quotes="'\""),
}

IDENT = re.compile(r"[^\W\d]\w*")
NUMBER = re.compile(
    r"(?:0[xX][0-9a-fA-F_']+|0[bB][01_']+|0[oO][0-7_]+|\d[\d_']*(?:\.\d[\d_']*)?(?:[eE][+-]?\d+)?)"
    r"[a-zA-Z0-9]*"
)


def tokenize_line(lang: str, line: str, state: str | None = None):
    """Return (tokens, new_state). tokens is a list of (kind, text)."""
    lang = normalize(lang)
    if lang == "asm":
        return _asm_line(line), None
    spec = LANGS.get(lang)
    if spec is None:
        return [("plain", line)], None
    out: list[tuple[str, str]] = []
    i, n = 0, len(line)

    def emit(kind, text):
        if not text:
            return
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + text)
        else:
            out.append((kind, text))

    # Continue a construct that started on a previous line.
    if state:
        if state == "block":
            end = line.find(spec["block"][1])
            if end < 0:
                emit("com", line)
                return out, "block"
            emit("com", line[: end + 2])
            i = end + 2
        elif state.startswith("str:"):
            closer = state[4:]
            end = _find_close(line, 0, closer, escapes=closer != "`" or lang != "go")
            if end < 0:
                emit("str", line)
                return out, state
            emit("str", line[:end])
            i = end
        state = None

    if spec.get("pp") and line.lstrip().startswith("#"):
        # Preprocessor line: colour directive, keep <header> / "header" as string.
        m = re.match(r"(\s*#\s*\w+)(.*)$", line)
        if m:
            emit("pre", m.group(1))
            rest = m.group(2)
            hm = re.match(r"(\s*)(<[^>]*>|\"[^\"]*\")(.*)$", rest)
            if hm:
                emit("plain", hm.group(1))
                emit("str", hm.group(2))
                rest_tokens, _ = tokenize_line(lang, hm.group(3), None)
                out.extend(rest_tokens)
            else:
                rest_tokens, _ = tokenize_line(lang, rest, None)
                out.extend(rest_tokens)
            return out, None

    while i < n:
        ch = line[i]
        # comments
        for marker in spec["line"]:
            if line.startswith(marker, i):
                emit("com", line[i:])
                return out, None
        blk = spec.get("block")
        if blk and line.startswith(blk[0], i):
            end = line.find(blk[1], i + 2)
            if end < 0:
                emit("com", line[i:])
                return out, "block"
            emit("com", line[i : end + 2])
            i = end + 2
            continue
        # decorators / attributes / annotations
        deco = spec.get("deco")
        if deco and ch == deco:
            if deco == "#" and (line.startswith("#[", i) or line.startswith("#![", i)):
                end = line.find("]", i)
                end = n if end < 0 else end + 1
                emit("pre", line[i:end])
                i = end
                continue
            if deco != "#":
                m = IDENT.match(line, i + 1)
                if m:
                    emit("pre", line[i : m.end()])
                    i = m.end()
                    continue
        # strings
        if lang == "python":
            m = re.match(r"(?i:[rbuf]{0,2})(\"\"\"|'''|\"|')", line[i:])
            if m and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] == "_")):
                q = m.group(1)
                start = i + m.end()
                end = _find_close(line, start, q)
                if end < 0:
                    emit("str", line[i:])
                    return out, ("str:" + q) if len(q) == 3 else None
                emit("str", line[i:end])
                i = end
                continue
        elif spec.get("triple_dq") and line.startswith('"""', i):
            end = _find_close(line, i + 3, '"""')
            if end < 0:
                emit("str", line[i:])
                return out, 'str:"""'
            emit("str", line[i:end])
            i = end
            continue
        elif spec.get("rust"):
            m = re.match(r'b?r(#*)"', line[i:])
            if m and (i == 0 or not line[i - 1].isalnum()):
                closer = '"' + m.group(1)
                end = line.find(closer, i + m.end())
                if end < 0:
                    emit("str", line[i:])
                    return out, None
                emit("str", line[i : end + len(closer)])
                i = end + len(closer)
                continue
            if ch == "'":
                m = re.match(r"'(?:\\.[^']*|[^\\'])'", line[i:])
                if m:
                    emit("str", m.group(0))
                    i += m.end()
                    continue
                m = re.match(r"'[A-Za-z_]\w*", line[i:])
                if m:  # lifetime such as 'a or 'static
                    emit("pre", m.group(0))
                    i += m.end()
                    continue
        if ch in spec["quotes"]:
            end = _find_close(line, i + 1, ch, escapes=not (lang == "go" and ch == "`"))
            if end < 0:
                emit("str", line[i:])
                multiline = ch == "`"
                return out, ("str:" + ch) if multiline else None
            emit("str", line[i:end])
            i = end
            continue
        # numbers
        if ch.isdigit():
            m = NUMBER.match(line, i)
            if m:
                emit("num", m.group(0))
                i = m.end()
                continue
        # identifiers
        m = IDENT.match(line, i)
        if m:
            word = m.group(0)
            key = word.lower() if spec.get("nocase") else word
            j = m.end()
            if key in spec["kw"]:
                emit("kw", word)
            elif lang == "rust" and j < n and line[j] == "!" and not line.startswith("!=", j):
                emit("pre", word + "!")
                j += 1
            elif key in spec["ty"]:
                emit("ty", word)
            elif lang == "shell" and word.startswith("$"):
                emit("ty", word)
            elif j < n and line[j] == "(" or (lang in ("rust", "cpp") and line.startswith("::<", j)):
                emit("fn", word)
            elif word[:1].isupper() and lang in ("rust", "java", "typescript", "cpp", "go", "python") and len(word) > 1 and not word.isupper():
                emit("ty", word)
            else:
                emit("plain", word)
            i = j
            continue
        if lang == "shell" and ch == "$":
            m = re.match(r"\$\{?\w+\}?", line[i:])
            if m:
                emit("ty", m.group(0))
                i += m.end()
                continue
        emit("plain", ch)
        i += 1
    return out, state


def _find_close(line: str, start: int, quote: str, escapes: bool = True) -> int:
    """Index just after the closing quote, or -1 if the string continues."""
    i = start
    while i < len(line):
        if escapes and line[i] == "\\":
            i += 2
            continue
        if line.startswith(quote, i):
            return i + len(quote)
        i += 1
    return -1


def _asm_line(line: str):
    out: list[tuple[str, str]] = []
    code, comment = line, ""
    for marker in ("#", ";", "//"):
        # '#' only starts a comment when it's not an immediate like #10 (ARM)
        idx = code.find(marker)
        while idx >= 0 and marker == "#" and idx + 1 < len(code) and (code[idx + 1].isdigit() or code[idx + 1] == "-"):
            idx = code.find(marker, idx + 1)
        if idx >= 0:
            code, comment = code[:idx], code[idx:] + comment
    pos = 0
    first_word = True
    for m in re.finditer(r"\"(?:\\.|[^\"])*\"|'[^']*'|[^\W\d][\w.$]*:?|\.[\w.]+|0x[0-9a-fA-F]+|\d+|\s+|.", code):
        tok = m.group(0)
        if tok.isspace():
            kind = "plain"
        elif tok[0] in "\"'":
            kind = "str"
        elif tok.endswith(":") and first_word:
            kind = "fn"
        elif tok.startswith("."):
            kind = "pre"
            first_word = False
        elif tok[0].isdigit():
            kind = "num"
        elif ASM_REG.match(tok):
            kind = "reg"
        elif re.match(r"[^\W\d]", tok) and first_word:
            kind = "kw"
            first_word = False
        elif tok.lower() in ("qword", "dword", "word", "byte", "ptr", "offset", "rel"):
            kind = "ty"
        else:
            kind = "plain"
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + tok)
        else:
            out.append((kind, tok))
        pos = m.end()
    if comment:
        out.append(("com", comment))
    return out


def to_html(code: str, lang: str) -> str:
    """Highlight a whole code block, returning HTML spans (class="hl-<kind>")."""
    lines = code.split("\n")
    state = None
    parts = []
    for idx, line in enumerate(lines):
        tokens, state = tokenize_line(lang, line, state)
        for kind, text in tokens:
            esc = html.escape(text, quote=False)
            parts.append(esc if kind == "plain" else f'<span class="hl-{kind}">{esc}</span>')
        if idx != len(lines) - 1:
            parts.append("\n")
    return "".join(parts)


# Colours shared by the app and the website (light / dark).
PALETTE = {
    "light": {"kw": "#8250df", "ty": "#0550ae", "str": "#0a7c2f", "num": "#b35900", "com": "#6e7781",
              "fn": "#8a4b00", "pre": "#cf222e", "reg": "#0969da"},
    "dark": {"kw": "#d2a8ff", "ty": "#79c0ff", "str": "#7ee787", "num": "#ffa657", "com": "#8b949e",
             "fn": "#e3b341", "pre": "#ff7b72", "reg": "#a5d6ff"},
}


def css(theme: str = "light", prefix: str = "") -> str:
    pal = PALETTE[theme]
    rules = [f"{prefix}.hl-{k} {{ color: {v}; }}" for k, v in pal.items()]
    rules.append(f"{prefix}.hl-com {{ font-style: italic; }}")
    return "\n".join(rules)
