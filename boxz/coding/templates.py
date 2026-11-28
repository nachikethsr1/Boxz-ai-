"""A template: real source files plus typed, placeholder-driven parameters.

Design rules that the rest of the system depends on:

* **No placeholder is guessed.**  A parameter names its own ``__TOKEN__`` and
  the customizer replaces only that token.  There is no find-and-replace on
  values, so the string "red" in a comment is never rewritten.
* **Every declared parameter is used and every placeholder is declared.**
  ``validate_placeholders()`` enforces both directions at load time, which
  turns a typo in a JSON file into a loud error instead of a blank canvas.
* **Ranges are enforced, not documented.**  A value outside ``[min, max]`` is
  clamped by the customizer and reported, never silently written.

A template lives in its own directory so it can be copied, reviewed or deleted
as one unit::

    data/templates/js_snake/
        template.json
        index.html
        index.js
        style.css
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "ParamSpec",
    "TemplateFile",
    "Template",
    "load_template",
    "load_library",
]


PLACEHOLDER_RE = r"__[A-Z][A-Z0-9_]*__"
_KNOWN_TYPES = {"string", "int", "float", "bool", "color", "enum"}


@dataclass
class ParamSpec:
    """One typed, named parameter of a template."""

    name: str
    placeholder: str
    type: str = "string"
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    allowed: Optional[List[Any]] = None
    aliases: List[str] = field(default_factory=list)
    nl_values: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # -- coercion --------------------------------------------------------
    def coerce(self, value: Any) -> Any:
        """Return a value of this parameter's type, or raise ValueError.

        Coercion is strict on purpose: silently turning "banana" into 0 hides
        a bug in the caller forever.  The one convenience allowed is a string
        number for a numeric parameter, because extracted values often arrive
        as text.
        """
        t = self.type
        if t == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "yes", "on", "1", "enabled", "enable"):
                    return True
                if low in ("false", "no", "off", "0", "disabled", "disable"):
                    return False
            if value in (0, 1):
                return bool(value)
            raise ValueError(f"{self.name}: expected a boolean, got {value!r}")

        if t == "int":
            try:
                out = int(str(value).strip().replace("_", "").replace(",", ""))
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected an integer, got {value!r}")
            return self._clamp(out)

        if t == "float":
            try:
                out = float(str(value).strip().replace(",", ""))
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected a number, got {value!r}")
            return self._clamp(out)

        if t == "color":
            return _normalise_color(value, self.name)

        if t == "enum":
            s = str(value).strip()
            if self.allowed:
                if s in self.allowed:
                    return s
                low = s.lower()
                for a in self.allowed:
                    if str(a).lower() == low:
                        return a
                raise ValueError(
                    f"{self.name}: {s!r} is not one of {self.allowed}"
                )
            return s

        # string
        return str(value)

    def _clamp(self, out: float) -> float:
        if self.min is not None and out < self.min:
            raise ValueError(f"{self.name}: {out} is below the minimum {self.min}")
        if self.max is not None and out > self.max:
            raise ValueError(f"{self.name}: {out} is above the maximum {self.max}")
        return out

    def resolve_word(self, word: str) -> Any:
        """Map a natural-language word ("bigger", "red") onto a value."""
        low = word.strip().lower()
        if low in self.nl_values:
            return self.coerce(self.nl_values[low])
        return None

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "placeholder": self.placeholder,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "allowed": self.allowed,
            "aliases": self.aliases,
            "nl_values": self.nl_values,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        if d.get("type", "string") not in _KNOWN_TYPES:
            raise ValueError(
                f"parameter {d.get('name')!r}: unknown type {d.get('type')!r}; "
                f"expected one of {sorted(_KNOWN_TYPES)}"
            )
        return cls(
            name=d["name"],
            placeholder=d["placeholder"],
            type=d.get("type", "string"),
            default=d.get("default"),
            min=d.get("min"),
            max=d.get("max"),
            step=d.get("step"),
            allowed=d.get("allowed"),
            aliases=list(d.get("aliases", [])),
            nl_values=dict(d.get("nl_values", {})),
            description=d.get("description", ""),
        )


@dataclass
class TemplateFile:
    """A file the template emits, plus the language used to validate it."""

    path: str
    language: str
    role: str = ""            # "page" | "script" | "style" | "module" | ""

    def to_dict(self) -> dict:
        return {"path": self.path, "language": self.language, "role": self.role}


@dataclass
class Template:
    id: str
    path: Path
    title: str = ""
    language: str = ""
    category: str = ""
    task: str = ""
    summary: str = ""
    files: List[TemplateFile] = field(default_factory=list)
    parameters: List[ParamSpec] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    when_to_select: List[str] = field(default_factory=list)
    when_not_to_select: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    licence: str = "CC0-1.0 (project original)"
    version: str = "1.0.0"
    status: str = "untested"          # "tested" | "untested"
    validation_note: str = ""

    # -- lookups ---------------------------------------------------------
    def param_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def param(self, name: str) -> Optional[ParamSpec]:
        for p in self.parameters:
            if p.name == name:
                return p
            if name in p.aliases:
                return p
        return None

    def defaults(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.parameters}

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def validate_placeholders(self) -> None:
        """Both directions: declared->used, and used->declared.

        Raises ValueError, naming the template and the offending token.
        """
        declared = {p.placeholder for p in self.parameters}
        for f in self.files:
            body = self.read(f.path)
            used = set(__import__("re").findall(PLACEHOLDER_RE, body))
            unknown = used - declared
            if unknown:
                raise ValueError(
                    f"template {self.id!r} file {f.path!r} uses undeclared "
                    f"placeholder(s) {sorted(unknown)}"
                )
        # every declared placeholder must appear in at least one file
        blob = "".join(self.read(f.path) for f in self.files)
        for p in self.parameters:
            if p.placeholder not in blob:
                raise ValueError(
                    f"template {self.id!r} declares parameter {p.name!r} with "
                    f"placeholder {p.placeholder} but no file contains it"
                )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "language": self.language,
            "category": self.category,
            "task": self.task,
            "summary": self.summary,
            "files": [f.to_dict() for f in self.files],
            "parameters": [p.to_dict() for p in self.parameters],
            "keywords": self.keywords,
            "when_to_select": self.when_to_select,
            "when_not_to_select": self.when_not_to_select,
            "known_limitations": self.known_limitations,
            "licence": self.licence,
            "version": self.version,
            "status": self.status,
            "validation_note": self.validation_note,
        }


def _normalise_color(value: Any, name: str) -> str:
    """Accept ``#rrggbb``, ``#rgb`` or a small set of CSS colour names."""
    s = str(value).strip().lower()
    named = {
        "black": "#000000", "white": "#ffffff", "red": "#ff4136",
        "green": "#2ecc40", "blue": "#0074d9", "yellow": "#ffdc00",
        "orange": "#ff851b", "purple": "#b10dc9", "pink": "#f012be",
        "gray": "#aaaaaa", "grey": "#aaaaaa", "cyan": "#7fdbff",
        "teal": "#39cccc", "navy": "#001f3f", "lime": "#01ff70",
    }
    if s in named:
        return named[s]
    if s.startswith("#"):
        hexpart = s[1:]
        if len(hexpart) == 3 and all(c in "0123456789abcdef" for c in hexpart):
            return "#" + "".join(c * 2 for c in hexpart)
        if len(hexpart) == 6 and all(c in "0123456789abcdef" for c in hexpart):
            return "#" + hexpart
    raise ValueError(f"{name}: {value!r} is not a colour (#rrggbb or a name)")


def load_template(path: str | Path) -> Template:
    """Load one ``template.json`` and its files.  Validates on load."""
    p = Path(path)
    meta = json.loads(p.read_text(encoding="utf-8"))
    tpl = Template(
        id=meta["id"],
        path=p.parent,
        title=meta.get("title", ""),
        language=meta.get("language", ""),
        category=meta.get("category", ""),
        task=meta.get("task", ""),
        summary=meta.get("summary", ""),
        files=[TemplateFile(**f) for f in meta.get("files", [])],
        parameters=[ParamSpec.from_dict(d) for d in meta.get("parameters", [])],
        keywords=list(meta.get("keywords", [])),
        when_to_select=list(meta.get("when_to_select", [])),
        when_not_to_select=list(meta.get("when_not_to_select", [])),
        known_limitations=list(meta.get("known_limitations", [])),
        licence=meta.get("licence", "CC0-1.0 (project original)"),
        version=meta.get("version", "1.0.0"),
        status=meta.get("status", "untested"),
        validation_note=meta.get("validation_note", ""),
    )
    if not tpl.files:
        raise ValueError(f"template {tpl.id!r} declares no files")
    for f in tpl.files:
        if not (tpl.path / f.path).exists():
            raise ValueError(
                f"template {tpl.id!r} declares file {f.path!r} that does not exist"
            )
    tpl.validate_placeholders()
    return tpl


def load_library(root: str | Path = "data/templates",
                 only: Optional[Iterable[str]] = None) -> Dict[str, Template]:
    """Load every ``*/template.json`` under ``root``, keyed by template id."""
    base = Path(root)
    if not base.is_dir():
        return {}
    wanted = set(only) if only is not None else None
    out: Dict[str, Template] = {}
    for meta_path in sorted(base.glob("*/template.json")):
        tpl = load_template(meta_path)
        if wanted is not None and tpl.id not in wanted:
            continue
        if tpl.id in out:
            raise ValueError(f"duplicate template id {tpl.id!r}")
        out[tpl.id] = tpl
    return out
