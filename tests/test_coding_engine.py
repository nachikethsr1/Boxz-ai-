"""Tests for the Coding Engine.

The engine must never claim a snippet passed a check that did not run.  These
tests therefore assert on the *tri-state* status (passed / failed /
unavailable), not merely on a boolean.
"""
from __future__ import annotations

import pytest

from boxz.coding.engine import (detect_language, extract_code_blocks, node_available,
                                validate, validate_reply, verdict)


VALID_JS = "const x = 1;\nfunction f(a) { return a + 1; }\nconsole.log(f(x));"
INVALID_JS = "const x = 1;\nfunction f(a { return a + 1; }"
VALID_HTML = '<div class="wrap"><ul><li>one</li></ul></div>'
INVALID_HTML = '<div class="wrap"><p>hi</div>'
VALID_CSS = ".btn { color: red; padding: 4px; }"
INVALID_CSS = ".btn { color: red;"
VALID_PY = "def f(x):\n    return x + 1\n"


# -------------------------------------------------------------- javascript
@pytest.mark.skipif(not node_available(), reason="Node.js not installed")
def test_js_valid_passes():
    v = validate(VALID_JS, "javascript")
    assert v.status == "passed", v.summary()


@pytest.mark.skipif(not node_available(), reason="Node.js not installed")
def test_js_invalid_fails_with_location():
    v = validate(INVALID_JS, "javascript")
    assert v.status == "failed"
    assert v.errors and v.errors[0]["line"] == 2
    assert "SyntaxError" in v.errors[0]["message"] or "Unexpected" in v.errors[0]["message"]


def test_js_reports_unavailable_without_node(monkeypatch):
    """Simulate a device with no Node.js: the engine must say so."""
    monkeypatch.setattr("boxz.coding.engine.shutil.which", lambda _: None)
    v = validate(VALID_JS, "javascript")
    assert v.status == "unavailable"
    assert "Node.js" in v.note
    assert not v.ok


# -------------------------------------------------------------------- html
def test_html_valid_passes():
    assert validate(VALID_HTML, "html").status == "passed"


def test_html_unclosed_fails():
    v = validate(INVALID_HTML, "html")
    assert v.status == "failed"
    assert "not closed" in v.errors[0]["message"]


def test_html_duplicate_attribute_fails():
    v = validate('<img src="a.png" src="b.png">', "html")
    assert v.status == "failed"
    assert "duplicate attribute" in v.errors[0]["message"]


def test_html_void_elements_are_fine():
    assert validate('<div><br><hr><img src="x.png"></div>', "html").status == "passed"


# --------------------------------------------------------------------- css
def test_css_valid_passes():
    assert validate(VALID_CSS, "css").status == "passed"


def test_css_unbalanced_brace_fails():
    v = validate(INVALID_CSS, "css")
    assert v.status == "failed"
    assert "unclosed" in v.errors[0]["message"]


def test_css_rejects_js_comments():
    v = validate(".a { color: red; } // nope", "css")
    assert v.status == "failed"
    assert "not a CSS comment" in v.errors[0]["message"]


# ------------------------------------------------------------------ python
def test_python_valid_and_invalid():
    assert validate(VALID_PY, "python").status == "passed"
    assert validate("def f(x)\n    return x\n", "python").status == "failed"


# ------------------------------------------------------- extraction / detect
def test_detects_language_when_fence_untagged():
    blocks = extract_code_blocks(
        "```\ndocument.getElementById('a');\n```\n"
        "```\n.card { color: blue; }\n```\n"
        "```\n<div>x</div>\n```\n")
    assert [b.language for b in blocks] == ["javascript", "css", "html"]


def test_preserves_explicit_tag():
    blocks = extract_code_blocks("```javascript\nlet a = 1;\n```")
    assert blocks[0].tag == "javascript"
    assert blocks[0].language == "javascript"


def test_multiple_blocks_all_validated():
    reply = ("Here:\n```js\nlet a = 1;\n```\nand:\n```css\n.x { b: c; }\n```\n"
             "and broken:\n```js\nlet = ;\n```\n")
    blocks = validate_reply(reply)
    assert len(blocks) == 3
    statuses = [b.validation.status for b in blocks]
    assert statuses[0] == "passed"
    assert statuses[1] == "passed"
    assert statuses[2] == "failed"
    assert "1 of 3" in verdict(blocks)


def test_no_code_yields_no_blocks():
    assert validate_reply("No code here, just prose.") == []
    assert verdict([]) == ""


def test_unknown_language_is_unavailable_not_passed():
    v = validate("SELECT 1;", "sql")
    assert v.status == "unavailable"
    assert not v.ok


def test_engine_never_executes_code():
    """A snippet with a side effect must not produce a side effect."""
    import tempfile
    from pathlib import Path

    marker = Path(tempfile.gettempdir()) / "boxz_should_not_exist.txt"
    if marker.exists():
        marker.unlink()
    code = f"require('fs').writeFileSync({str(marker)!r}, 'x');"
    v = validate(code, "javascript")
    assert v.status in ("passed", "failed")      # syntax checked
    assert not marker.exists(), "the coding engine executed the snippet!"


def test_detect_language_fallback():
    assert detect_language("console.log(1)") == "javascript"
