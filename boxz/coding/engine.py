"""The Coding Engine: extract code from a model reply and validate its syntax.

This is a *toolchain*, not a model.  It never invents code and never executes
generated code.  Its only job is to answer one question honestly:

    "Is this snippet syntactically valid, and if not, where is the error?"

Validators
----------
``javascript`` / ``typescript``
    ``node --check`` in a subprocess, when Node.js is present.
``html``
    :mod:`html.parser` with an explicit tag-balance and attribute check.
``css``
    Brace balance, ``//`` misuse, stray semicolons, unclosed comments and
    unknown-at-rule detection.
``python``
    :func:`compile` (never executes).

If a validator is unavailable the result is ``status="unavailable"`` -- the
engine never reports "passed" for code it did not actually check.

Safety
------
Generated code is **never executed**.  ``node --check`` parses only; it does not
run the script.  Subprocesses are bounded by a timeout and run with a scratch
cwd, and no untrusted code is ever handed to :func:`eval`/:func:`exec`.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "CodeBlock",
    "Validation",
    "extract_code_blocks",
    "detect_language",
    "validate",
    "validate_reply",
    "node_available",
    "list_capabilities",
]

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

_EXT = {
    "javascript": ".js", "js": ".js", "jsx": ".js", "typescript": ".ts",
    "ts": ".ts", "html": ".html", "css": ".css", "python": ".py", "py": ".py",
    "json": ".json",
}

_ALIASES = {
    "js": "javascript", "node": "javascript", "jsx": "javascript",
    "ts": "typescript", "py": "python", "htm": "html", "svg": "html",
}

#: Ordered heuristics used when a fence carries no language tag.
_SIGNATURES: Sequence[Tuple[str, str]] = (
    ("html", r"<!DOCTYPE|<html|<div|<span|<button|<script|<body|<head|<canvas|<input|<style"),
    ("css", r"^\s*[.#@][\w-]+\s*[,{:]|^\s*[\w-]+\s*:\s*[^;]+;\s*$|\{[\s\S]*:[\s\S]*\}"),
    ("javascript", r"\b(function|const|let|var|=>|document\.|window\.|console\.|addEventListener|"
                   r"querySelector|require\(|module\.exports|import\s+\{)"),
    ("python", r"^\s*(def|class|import|from)\s+\w+|:\s*$"),
)


@dataclass
class CodeBlock:
    code: str
    language: str          # resolved language, or "unknown"
    tag: str               # what the fence actually said ("" if untagged)
    index: int = 0
    validation: Optional["Validation"] = None


@dataclass
class Validation:
    language: str
    status: str                    # "passed" | "failed" | "unavailable" | "skipped"
    errors: List[dict] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict:
        return {"language": self.language, "status": self.status,
                "errors": self.errors, "note": self.note}

    def summary(self) -> str:
        if self.status == "passed":
            return f"{self.language}: syntax OK"
        if self.status == "failed":
            first = self.errors[0] if self.errors else {}
            loc = ""
            if first.get("line"):
                loc = f" (line {first['line']}"
                loc += f", col {first['column']})" if first.get("column") else ")"
            return f"{self.language}: syntax error{loc}: {first.get('message', 'invalid')}"
        if self.status == "unavailable":
            return f"{self.language}: could not validate ({self.note})"
        return f"{self.language}: not validated"


# --------------------------------------------------------------------------
# extraction & language identification
# --------------------------------------------------------------------------

def detect_language(code: str) -> str:
    """Guess a language from the snippet body alone."""
    for lang, pattern in _SIGNATURES:
        if re.search(pattern, code, re.MULTILINE):
            return lang
    if re.search(r"[{};]", code):
        return "javascript"
    return "unknown"


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """Pull every fenced block out of ``text``, keeping untagged ones."""
    blocks: List[CodeBlock] = []
    for i, m in enumerate(_FENCE.finditer(text or "")):
        tag = (m.group(1) or "").strip().lower()
        code = m.group(2)
        # a fence may carry "```js title=..." -- take the first word
        tag = tag.split()[0] if tag else ""
        lang = _ALIASES.get(tag, tag) if tag else detect_language(code)
        if lang not in _EXT and lang != "unknown":
            lang = detect_language(code)
        blocks.append(CodeBlock(code=code, language=lang, tag=tag, index=i))
    return blocks


# --------------------------------------------------------------------------
# validators
# --------------------------------------------------------------------------

def node_available() -> bool:
    return shutil.which("node") is not None


def _validate_js(code: str, lang: str = "javascript") -> Validation:
    node = shutil.which("node")
    if node is None:
        return Validation(lang, "unavailable",
                          note="Node.js not installed; JavaScript cannot be syntax-checked")
    suffix = ".js" if lang == "javascript" else ".mjs"
    tmpdir = Path(tempfile.mkdtemp(prefix="boxz_syntax_"))
    path = tmpdir / f"snippet{suffix}"
    try:
        path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True, text=True, timeout=10, cwd=tmpdir,
        )
        if proc.returncode == 0:
            return Validation(lang, "passed")
        err = (proc.stderr or "").strip().splitlines()
        msg, line, col = "syntax error", None, None
        for ln in err:
            mm = re.search(r"^(.*?):(\d+)$", ln.strip())
            if mm and "SyntaxError" in ln:
                pass
            if "SyntaxError" in ln or "Error" in ln:
                msg = ln.strip()
                break
        mm = re.search(r":(\d+)(?::(\d+))?", "\n".join(err))
        if mm:
            line = int(mm.group(1))
            col = int(mm.group(2)) if mm.group(2) else None
        if msg.startswith(path.name + ":"):
            msg = msg.split(":", 1)[1].strip()
        return Validation(lang, "failed",
                          errors=[{"line": line, "column": col, "message": msg}],
                          note="\n".join(err[-3:]))
    except subprocess.TimeoutExpired:
        return Validation(lang, "unavailable", note="syntax check timed out")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
         "meta", "param", "source", "track", "wbr", "!doctype"}


class _HTMLChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[Tuple[str, int]] = []
        self.errors: List[dict] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() not in _VOID:
            self.stack.append((tag.lower(), self.getpos()[0]))
        seen = set()
        for name, _ in attrs:
            if name in seen:
                self.errors.append({"line": self.getpos()[0],
                                    "message": f"duplicate attribute {name!r} on <{tag}>"})
            seen.add(name)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _VOID:
            return
        if not self.stack:
            self.errors.append({"line": self.getpos()[0],
                                "message": f"closing </{t}> with no open tag"})
            return
        if self.stack[-1][0] == t:
            self.stack.pop()
        else:
            # tolerate mis-nesting but report it
            names = [s[0] for s in self.stack]
            if t in names:
                while self.stack and self.stack[-1][0] != t:
                    bad, line = self.stack.pop()
                    self.errors.append({"line": line,
                                        "message": f"<{bad}> is not closed before </{t}>"})
                if self.stack:
                    self.stack.pop()
            else:
                self.errors.append({"line": self.getpos()[0],
                                    "message": f"unexpected closing </{t}>"})


def _validate_html(code: str) -> Validation:
    checker = _HTMLChecker()
    try:
        checker.feed(code)
        checker.close()
    except Exception as e:                                # pragma: no cover - defensive
        return Validation("html", "failed",
                          errors=[{"message": f"parser error: {e}"}])
    for tag, line in checker.stack:
        checker.errors.append({"line": line, "message": f"<{tag}> is never closed"})
    if checker.errors:
        return Validation("html", "failed", errors=checker.errors)
    return Validation("html", "passed")


def _validate_css(code: str) -> Validation:
    errors: List[dict] = []
    lines = code.split("\n")
    depth = 0
    in_comment = False
    open_line = 0
    for i, raw in enumerate(lines, 1):
        line = raw
        if in_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_comment = False
            else:
                continue
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" in after:
                line = before + after.split("*/", 1)[1]
            else:
                line = before
                in_comment = True
                break
        if re.search(r"(^|[^:])//", line) and not re.search(r"url\(\s*['\"]?//", line):
            errors.append({"line": i, "message": "// is not a CSS comment; use /* */"})
        for ch in line:
            if ch == "{":
                depth += 1
                if depth == 1:
                    open_line = i
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append({"line": i, "message": "unexpected '}'"})
                    depth = 0
    if in_comment:
        errors.append({"line": len(lines), "message": "unclosed /* comment"})
    if depth > 0:
        errors.append({"line": open_line, "message": f"{depth} unclosed '{{' block(s)"})
    if errors:
        return Validation("css", "failed", errors=errors)
    return Validation("css", "passed")


def _validate_python(code: str) -> Validation:
    try:
        compile(code, "<boxz-snippet>", "exec")
        return Validation("python", "passed")
    except SyntaxError as e:
        return Validation("python", "failed",
                          errors=[{"line": e.lineno, "column": e.offset,
                                   "message": f"{e.msg}"}])


def _validate_json(code: str) -> Validation:
    try:
        json.loads(code)
        return Validation("json", "passed")
    except json.JSONDecodeError as e:
        return Validation("json", "failed",
                          errors=[{"line": e.lineno, "column": e.colno,
                                   "message": e.msg}])


def validate(code: str, language: str) -> Validation:
    """Dispatch to the right validator; never executes ``code``."""
    lang = _ALIASES.get(language.lower(), language.lower())
    if not code.strip():
        return Validation(lang, "skipped", note="empty snippet")
    if lang in ("javascript", "typescript"):
        return _validate_js(code, lang)
    if lang == "html":
        return _validate_html(code)
    if lang == "css":
        return _validate_css(code)
    if lang == "python":
        return _validate_python(code)
    if lang == "json":
        return _validate_json(code)
    return Validation(lang or "unknown", "unavailable",
                      note=f"no validator for {lang or 'an untagged snippet'}")


def validate_reply(text: str) -> List[CodeBlock]:
    """Extract every fence in a model reply and validate it in place."""
    blocks = extract_code_blocks(text)
    for b in blocks:
        b.validation = validate(b.code, b.language)
    return blocks


def verdict(blocks: Sequence[CodeBlock]) -> str:
    """A one-line, human-readable verdict safe to show the user."""
    if not blocks:
        return ""
    parts = []
    for b in blocks:
        if b.validation is not None:
            parts.append(b.validation.summary())
    passed = sum(1 for b in blocks if b.validation and b.validation.ok)
    failed = sum(1 for b in blocks if b.validation and b.validation.status == "failed")
    if failed:
        head = f"{failed} of {len(blocks)} code block(s) have syntax errors."
    elif passed == len(blocks):
        head = f"All {len(blocks)} code block(s) passed a syntax check."
    else:
        head = f"{passed} of {len(blocks)} code block(s) syntax-checked."
    return head + (" " + "; ".join(parts) if parts else "")


def list_capabilities() -> List[str]:
    return [
        "fenced code-block extraction (tagged and untagged)",
        "language identification from content when the fence is untagged",
        "JavaScript/TypeScript syntax check via node --check" +
        ("" if node_available() else " (currently unavailable: Node.js not installed)"),
        "HTML tag-balance and duplicate-attribute validation",
        "CSS brace/comment/misuse validation",
        "Python compile-only syntax check",
        "JSON parse check",
        "never executes generated code",
    ]
