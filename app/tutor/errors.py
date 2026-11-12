"""Explain common error messages in the learner's own language.

Compilers and interpreters print errors in English. Beginners (especially those
who don't read English well) get stuck on them, so we recognise the most common
messages and add a plain-language explanation from locales/<lang>.json
(keys "err.<id>"). Named regex groups become {placeholders} in the text.
"""
from __future__ import annotations

import re

# (languages or "*", regex, message id)
PATTERNS: list[tuple[tuple[str, ...] | str, str, str]] = [
    # ---------------- full-width (Chinese/Japanese) punctuation typed by mistake
    (("python",), r"SyntaxError: invalid character '(?P<ch>.)'", "fullwidth"),
    (("c", "cpp"), r"stray '\\3\d\d' in program", "fullwidth"),
    (("c", "cpp"), r"\\U0000(?:ff[0-9a-f]{2}|20(?:1[89cd]|26)|300[12])", "fullwidth"),   # newer GCC names the character
    (("c", "cpp"), r"extended character [“”‘’（），；：。、！？] is not valid", "fullwidth"),
    (("c", "cpp"), r'invalid suffix "[，；：。（）、]', "fullwidth"),
    # Cyrillic letters that look like Latin ones (р/p, с/c, о/o, а/a, е/e) typed with a Russian layout
    (("c", "cpp"), r"\\U0000(?P<code>04[0-9a-f]{2})", "cyrillic-ident"),
    (("python",), r"NameError: name '(?P<name>[^']*[\u0400-\u04ff][^']*)' is not defined", "cyrillic-ident"),
    (("java",), r"illegal character: '\\u(?:ff|201|300)\w+'", "fullwidth"),
    (("rust",), r"unknown start of token: \\u\{(?:ff|201|300)", "fullwidth"),
    (("go",), r"invalid character U\+(?:FF|201|300)", "fullwidth"),
    (("javascript", "typescript"), r"SyntaxError: Invalid or unexpected token", "fullwidth-maybe"),
    # ---------------- Python
    (("python",), r"TypeError: '(?P<t>NoneType)' object is not (?:subscriptable|iterable)", "py-none"),
    (("python",), r"NameError: name '(?P<name>[^']+)' is not defined", "py-name"),
    (("python",), r"IndentationError: expected an indented block", "py-indent-expected"),
    (("python",), r"IndentationError: unexpected indent", "py-indent-unexpected"),
    (("python",), r"IndentationError: unindent does not match", "py-indent-mismatch"),
    (("python",), r"TabError", "py-tab"),
    (("python",), r"SyntaxError: unterminated string literal", "py-unterminated-string"),
    (("python",), r"SyntaxError: '(?P<ch>[(\[{])' was never closed", "py-never-closed"),
    (("python",), r"SyntaxError: expected ':'", "py-colon"),
    (("python",), r"SyntaxError: Missing parentheses in call to 'print'", "py-print2"),
    (("python",), r"SyntaxError: invalid syntax\. Maybe you meant '==' or ':=' instead of '='", "py-eq"),
    (("python",), r"SyntaxError: (?:invalid syntax|incomplete input)", "py-syntax"),
    (("python",), r"TypeError: can only concatenate str \(not \"(?P<t>\w+)\"\) to str", "py-concat"),
    (("python",), r"TypeError: unsupported operand type\(s\) for (?P<op>\S+): '(?P<a>\w+)' and '(?P<b>\w+)'", "py-operand"),
    (("python",), r"TypeError: '(?P<t>\w+)' object is not callable", "py-not-callable"),
    (("python",), r"TypeError: '(?P<t>\w+)' object is not subscriptable", "py-not-subscriptable"),
    (("python",), r"TypeError: (?P<f>[\w.]+)\(\) (?:takes|missing)", "py-args"),
    (("python",), r"ValueError: invalid literal for int\(\) with base 10: '(?P<v>[^']*)'", "py-int"),
    (("python",), r"ValueError: could not convert string to float: '(?P<v>[^']*)'", "py-float"),
    (("python",), r"ZeroDivisionError", "zero-division"),
    (("python",), r"IndexError: (?:list|string|tuple) index out of range", "py-index"),
    (("python",), r"KeyError: (?P<k>.+)", "py-key"),
    (("python",), r"TypeError: unhashable type: '(?P<t>\w+)'", "py-unhashable"),
    (("python",), r"RuntimeError: (?:dictionary|set) changed size during iteration", "py-changed-size"),
    (("python",), r"AttributeError: '(?P<t>\w+)' object has no attribute '(?P<a>\w+)'", "py-attr"),
    # ---------------- PyTorch (LLM tutorial)
    (("python",), r"mat1 and mat2 shapes cannot be multiplied \((?P<a>\d+x\d+) and (?P<b>\d+x\d+)\)", "torch-shape"),
    (("python",), r"Expected all tensors to be on the same device", "torch-device"),
    (("python",), r"expected (?:target dtype to be Long|scalar type Long)", "torch-target-long"),
    (("python",), r"(?:mat1 and mat2 must have the same dtype|expected (?:m1 and m2|mat1 and mat2) to have the same dtype|expected scalar type (?P<a>\w+) but found (?P<b>\w+))", "torch-dtype"),
    (("python",), r"Trying to backward through the graph a second time", "torch-backward-twice"),
    (("python",), r"element 0 of tensors does not require grad", "torch-requires-grad"),
    (("python",), r"Can't call numpy\(\) on Tensor that requires grad", "torch-numpy"),
    (("python",), r"(?:CUDA out of memory|OutOfMemoryError)", "torch-oom"),
    (("python",), r"The size of tensor a \((?P<a>\d+)\) must match the size of tensor b \((?P<b>\d+)\)", "torch-broadcast"),
    (("python",), r"ModuleNotFoundError: No module named '(?P<m>[\w.]+)'", "py-module"),
    (("python",), r"UnboundLocalError", "py-unbound"),
    (("python",), r"RecursionError", "recursion"),
    (("python",), r"FileNotFoundError: \[Errno 2\] No such file or directory: '(?P<f>[^']+)'", "file-not-found"),
    (("python",), r"EOFError: EOF when reading a line", "eof-input"),
    (("python",), r"UnicodeDecodeError", "unicode"),
    # ---------------- C / C++
    (("c", "cpp"), r"expected ';' before", "c-semicolon"),
    (("c", "cpp"), r"error: expected ';'", "c-semicolon"),
    (("c",), r"'(?P<name>\w+)' undeclared", "c-undeclared"),
    (("cpp",), r"'(?P<name>\w+)' was not declared in this scope", "c-undeclared"),
    (("c", "cpp"), r"use of undeclared identifier '(?P<name>\w+)'", "c-undeclared"),
    (("cpp",), r"'(?P<name>\w+)' is not a member of 'std'", "cpp-std-member"),
    (("c",), r"implicit declaration of function '(?P<name>\w+)'", "c-implicit"),
    (("c", "cpp", "asm"), r"undefined reference to `(?P<name>[^']+)'", "c-undefined-ref"),
    (("c", "cpp"), r"format '%f' expects argument of type 'float \*', but argument \d+ has type 'double \*'", "c-scanf-lf"),
    (("c", "cpp"), r"format '%(?P<f>\w+)' expects argument of type '[\w ]+ \*', but argument \d+ has type '(?:int|double|float|char|long|long int|long long int|unsigned int)'", "c-scanf-amp"),
    (("c", "cpp"), r"format '%(?P<f>\w+)' expects argument of type '(?P<t>[^']+)'", "c-format"),
    (("c", "cpp"), r"control reaches end of non-void function", "c-no-return"),
    (("c", "cpp"), r"suggest parentheses around assignment used as truth value", "c-assign-if"),
    (("c", "cpp"), r"'(?P<name>\w+)' is used uninitialized|'(?P<name2>\w+)' may be used uninitialized", "c-uninit"),
    (("c", "cpp"), r"expected declaration or statement at end of input|expected '}' at end of input", "c-brace"),
    (("c", "cpp"), r"missing terminating \" character", "c-string"),
    (("cpp",), r"no match for 'operator(?P<op>\S+)'", "cpp-no-operator"),
    (("cpp",), r"no matching function for call to", "cpp-no-matching"),
    (("c", "cpp"), r"fatal error: (?P<h>[\w./]+): No such file or directory", "c-header"),
    (("cpp",), r"fatal error: print: No such file", "cpp-print"),
    # ---------------- Java
    (("java",), r"error: cannot find symbol", "java-symbol"),
    (("java",), r"error: ';' expected", "c-semicolon"),
    (("java",), r"error: incompatible types: (?P<a>.+?) cannot be converted to (?P<b>\S+)", "java-incompatible"),
    (("java",), r"error: missing return statement", "c-no-return"),
    (("java",), r"error: class, interface, enum, or record expected|reached end of file while parsing", "c-brace"),
    (("java",), r"java\.lang\.NullPointerException", "java-npe"),
    (("java",), r"ArrayIndexOutOfBoundsException|StringIndexOutOfBoundsException|IndexOutOfBoundsException", "java-index"),
    (("java",), r"java\.lang\.NumberFormatException: For input string: \"(?P<v>[^\"]*)\"", "py-int"),
    (("java",), r"java\.util\.InputMismatchException", "java-input"),
    (("java",), r"java\.util\.NoSuchElementException", "eof-input"),
    (("java",), r"java\.lang\.ArithmeticException: / by zero", "zero-division"),
    (("java",), r"error: cannot find symbol\s+.*symbol:\s+(?:variable|method|class) IO", "java-io"),
    (("java",), r"UnsupportedClassVersionError|invalid source release|release version \d+ not supported", "java-version"),
    # ---------------- Rust
    (("rust",), r"error\[E0382\]", "rust-moved"),
    (("rust",), r"error\[E0499\]|error\[E0502\]", "rust-borrow"),
    (("rust",), r"error\[E0384\]", "rust-immutable"),
    (("rust",), r"error\[E0596\]", "rust-immutable"),
    (("rust",), r"error\[E0308\]", "rust-types"),
    (("rust",), r"error\[E0425\]: cannot find value `(?P<name>\w+)`", "c-undeclared"),
    (("rust",), r"error\[E0599\]", "rust-method"),
    (("rust",), r"error\[E0106\]", "rust-lifetime"),
    (("rust",), r"error\[E0277\]", "rust-trait"),
    (("rust",), r"panicked at .*index out of bounds", "rust-index"),
    (("rust",), r"called `(?:Option|Result)::unwrap\(\)` on an? `(?:None|Err)", "rust-unwrap"),
    (("rust",), r"attempt to (?:add|subtract|multiply) with overflow", "rust-overflow"),
    (("rust",), r"attempt to divide by zero", "zero-division"),
    # ---------------- Go
    (("go",), r"declared and not used: (?P<name>\w+)|(?P<name2>\w+) declared and not used", "go-unused-var"),
    (("go",), r"\"(?P<name>[\w/]+)\" imported and not used", "go-unused-import"),
    (("go",), r"undefined: (?P<name>\w+)", "c-undeclared"),
    (("go",), r"cannot use .+ as (?P<t>\S+) value", "go-types"),
    (("go",), r"missing return", "c-no-return"),
    (("go",), r"index out of range", "go-index"),
    (("go",), r"invalid memory address or nil pointer dereference", "go-nil"),
    (("go",), r"all goroutines are asleep - deadlock", "go-deadlock"),
    (("go",), r"integer divide by zero", "zero-division"),
    # ---------------- JavaScript / TypeScript
    (("javascript", "typescript"), r"ReferenceError: (?P<name>\S+) is not defined", "js-reference"),
    (("javascript", "typescript"), r"TypeError: Cannot read properties of (?P<v>undefined|null)", "js-undefined"),
    (("javascript", "typescript"), r"TypeError: (?P<name>\S+) is not a function", "js-not-function"),
    (("javascript", "typescript"), r"TypeError: Assignment to constant variable", "js-const"),
    (("javascript", "typescript"), r"SyntaxError: ", "js-syntax"),
    (("typescript",), r"TS2322|is not assignable to type", "ts-assign"),
    (("typescript",), r"ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX|TypeScript enum is not supported", "ts-strip"),
    # ---------------- Assembly
    (("asm",), r"Error: no such instruction: `(?P<i>[^']+)'", "asm-instruction"),
    (("asm",), r"Error: operand size mismatch|Error: operand type mismatch", "asm-operand"),
    (("asm",), r"relocation R_X86_64_32.*can not be used when making a PIE", "asm-pie"),
]


def explain(lang: str, text: str, exit_code: int | None, timed_out: bool, tr) -> list[str]:
    """Return localized explanations (each a short Markdown string). ``tr(key, **kw)``
    must return None when a translation is missing."""
    found: list[str] = []
    seen: set[str] = set()
    for langs, pattern, key in PATTERNS:
        if langs != "*" and lang not in langs:
            continue
        m = re.search(pattern, text or "", re.S if key == "java-io" else 0)
        if m and key not in seen:
            seen.add(key)
            values = {k: v for k, v in m.groupdict().items() if v is not None}
            if "name2" in values and "name" not in values:
                values["name"] = values["name2"]
            msg = tr("err." + key, **values)
            if msg:
                found.append(msg)
        if len(found) >= 2:
            break
    if timed_out:
        found.append(tr("err.timeout") or "")
    elif exit_code is not None and not found:
        if exit_code in (-11, 139, 3221225477):
            found.append(tr("err.segfault") or "")
        elif exit_code in (-6, 134):
            found.append(tr("err.abort") or "")
        elif exit_code in (3221225725,):
            found.append(tr("err.recursion") or "")
    return [f for f in found if f]
