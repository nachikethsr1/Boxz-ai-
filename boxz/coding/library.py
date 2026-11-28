"""Validate a generated project with the existing Coding Engine.

This module is the only place that decides what "validated" means, and it is
deliberately blunt about it:

* every generated file is checked with :func:`boxz.coding.engine.validate`
  using **that file's own language**, never a guess;
* the HTML validator only balances tags, so an HTML file is reported as
  ``html_structure`` rather than "the page works";
* nothing is executed and no browser is involved, so the report never claims
  a project was tested -- only that its files were parsed.

``ProjectValidation`` carries per-file results plus a single honest sentence
for the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .engine import Validation, validate  # noqa: F401  (re-exported)
from .customize import GeneratedProject
from .templates import Template

__all__ = ["FileReport", "ProjectValidation", "validate_project"]


#: What each validator actually covers, in plain words.  Kept here so the
#: report cannot drift away from the engine's real behaviour.
_COVERAGE = {
    "javascript": "parsed with `node --check` (syntax only, never executed)",
    "typescript": "not supported in v1",
    "html": "tag balance and duplicate attributes only; inline <script> and "
            "<style> are NOT parsed",
    "css": "brace and comment balance only",
    "python": "compiled, never executed",
    "json": "parsed",
}


@dataclass
class FileReport:
    path: str
    language: str
    status: str                       # passed | failed | unavailable | skipped
    summary: str = ""
    errors: List[dict] = field(default_factory=list)
    coverage: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "status": self.status,
            "summary": self.summary,
            "errors": self.errors,
            "coverage": self.coverage,
        }


@dataclass
class ProjectValidation:
    template_id: str
    files: List[FileReport] = field(default_factory=list)
    browser_tested: bool = False      # always False: nothing here runs a browser

    @property
    def ok(self) -> bool:
        """True only when every file actually passed a real check."""
        return bool(self.files) and all(f.ok for f in self.files)

    @property
    def failed(self) -> List[FileReport]:
        return [f for f in self.files if f.status == "failed"]

    @property
    def unchecked(self) -> List[FileReport]:
        return [f for f in self.files if f.status in ("unavailable", "skipped")]

    def verdict(self) -> str:
        """One sentence a user can trust, including what was *not* checked."""
        if not self.files:
            return "Nothing was validated: the project produced no files."
        passed = sum(1 for f in self.files if f.ok)
        total = len(self.files)
        if self.failed:
            names = ", ".join(f.path for f in self.failed)
            return (f"{len(self.failed)} of {total} file(s) failed a syntax "
                    f"check: {names}. This is a template bug, not a problem "
                    f"with your request.")
        if self.unchecked:
            names = ", ".join(f"{f.path} ({f.status})" for f in self.unchecked)
            return (f"{passed} of {total} file(s) syntax-checked; could not "
                    f"check {names}.")
        html = [f for f in self.files if f.language == "html"]
        suffix = ""
        if html:
            suffix = (" HTML was checked for tag balance only: its structure, "
                      "not its runtime behaviour.")
        return (f"All {total} file(s) passed a syntax check."
                f"{suffix} Nothing was run in a browser.")

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "ok": self.ok,
            "browser_tested": self.browser_tested,
            "verdict": self.verdict(),
            "files": [f.to_dict() for f in self.files],
        }


def validate_project(
    project: GeneratedProject,
    template: Optional[Template] = None,
) -> ProjectValidation:
    """Run the Coding Engine over every generated file.

    Language comes from the project's own per-file record, which came from the
    template metadata -- not from ``detect_language``, because a template
    author already knows what language they wrote.
    """
    report = ProjectValidation(template_id=project.template_id)
    for path, language, body in project.items():
        v: Validation = validate(body, language)
        report.files.append(FileReport(
            path=path,
            language=language,
            status=v.status,
            summary=v.summary(),
            errors=list(v.errors),
            coverage=_COVERAGE.get(language, "no validator"),
        ))
    return report
