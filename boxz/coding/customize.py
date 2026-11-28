"""Turn a template plus a value map into real, validated source files.

Three rules, in order of importance:

1. **Only declared placeholders are touched.**  Substitution replaces the
   literal token ``__FOO__``.  Nothing else in the file is rewritten, so the
   word "red" in a comment or a variable name is never disturbed.
2. **Every value is typed and range-checked** by :meth:`ParamSpec.coerce`.
   A bad value raises; it is never written into the file.
3. **No placeholder survives.**  After substitution the output is scanned for
   ``__ANYTHING__``; a leftover is a hard error, because shipping
   ``var x = __COLOR__;`` produces a blank page and a confusing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .templates import PLACEHOLDER_RE, ParamSpec, Template

__all__ = [
    "GeneratedProject",
    "CustomizationResult",
    "apply_values",
    "render_template",
    "Patch",
    "apply_patch",
]


class CustomizationError(ValueError):
    """A value could not be applied.  The message says which one and why."""


@dataclass
class GeneratedProject:
    """The files a template produced, ready to write or validate."""

    template_id: str
    files: Dict[str, str] = field(default_factory=dict)      # path -> body
    languages: Dict[str, str] = field(default_factory=dict)  # path -> language
    values: Dict[str, Any] = field(default_factory=dict)

    def items(self) -> Iterable[Tuple[str, str, str]]:
        """(path, language, body) in declaration order."""
        for path, body in self.files.items():
            yield path, self.languages[path], body

    def entry(self) -> str:
        """The file a user should open first: the page if there is one."""
        for path in self.files:
            if path.endswith(".html"):
                return path
        return next(iter(self.files))

    def to_artifacts(self) -> dict:
        """Shape returned by the HTTP API."""
        return {
            "template_id": self.template_id,
            "entry": self.entry(),
            "values": self.values,
            "files": [{"path": p, "language": lang, "bytes": len(b),
                       "content": b}
                      for p, lang, b in self.items()],
        }


@dataclass
class CustomizationResult:
    project: GeneratedProject
    applied: Dict[str, Any] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)   # name -> reason

    @property
    def ok(self) -> bool:
        return not self.rejected


def _stringify(param: ParamSpec, value: Any) -> str:
    """How a coerced value is written into source code."""
    if param.type == "bool":
        return "true" if value else "false"
    if param.type in ("int",):
        return str(int(value))
    if param.type == "float":
        text = repr(float(value))
        return text
    # strings and colours go inside quotes that the template supplies
    return str(value)


def apply_values(
    tpl: Template,
    values: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
) -> CustomizationResult:
    """Render ``tpl`` with ``values`` merged over the template defaults.

    ``strict=True`` raises on the first bad value; ``strict=False`` records it
    in ``rejected`` and falls back to the default, which is what an interactive
    session wants (the user is told, and still gets a working page).
    """
    final: Dict[str, Any] = tpl.defaults()
    applied: Dict[str, Any] = {}
    rejected: Dict[str, str] = {}

    for name, raw in (values or {}).items():
        param = tpl.param(name)
        if param is None:
            msg = f"{tpl.id}: unsupported parameter {name!r}"
            if strict:
                raise CustomizationError(msg)
            rejected[name] = msg
            continue
        try:
            coerced = param.coerce(raw)
        except ValueError as exc:
            if strict:
                raise CustomizationError(str(exc)) from exc
            rejected[name] = str(exc)
            continue
        final[param.name] = coerced
        applied[param.name] = coerced

    files: Dict[str, str] = {}
    languages: Dict[str, str] = {}
    for f in tpl.files:
        body = tpl.read(f.path)
        for param in tpl.parameters:
            if param.placeholder in body:
                body = body.replace(param.placeholder,
                                    _stringify(param, final[param.name]))
        leftover = re.findall(PLACEHOLDER_RE, body)
        if leftover:
            raise CustomizationError(
                f"{tpl.id}: file {f.path!r} still contains {sorted(set(leftover))} "
                f"after substitution -- a declared parameter is missing"
            )
        files[f.path] = body
        languages[f.path] = f.language

    return CustomizationResult(
        project=GeneratedProject(template_id=tpl.id, files=files,
                                 languages=languages, values=final),
        applied=applied,
        rejected=rejected,
    )


def render_template(tpl: Template, values: Optional[Dict[str, Any]] = None) -> GeneratedProject:
    """Convenience wrapper returning just the project (strict)."""
    return apply_values(tpl, values, strict=True).project


# ---------------------------------------------------------------------------
# follow-up patches
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    """One change to a session's parameter values.

    ``op`` is ``set`` (absolute value), ``scale`` (multiply a number) or
    ``offset`` (add to a number).  Follow-ups like "make it bigger" are a
    ``scale``, while "make it blue" is a ``set``.
    """

    param: str
    op: str = "set"
    value: Any = None
    factor: float = 1.0


def apply_patch(
    tpl: Template,
    current: Dict[str, Any],
    patch: Patch,
) -> Tuple[Dict[str, Any], str]:
    """Apply one patch to ``current``; return the new values and a note.

    The returned note is human-readable and is what the reply quotes back, so
    the user can see exactly what changed and what was clamped.
    """
    param = tpl.param(patch.param)
    if param is None:
        supported = ", ".join(p.name for p in tpl.parameters)
        raise CustomizationError(
            f"{tpl.id}: {patch.param!r} is not a supported parameter. "
            f"Supported: {supported}"
        )
    note = ""
    if patch.op == "set":
        # An absolute value is clamped like a relative one.  ``coerce`` raises
        # outside the range, which used to escape as a ValueError: "set the
        # grid to 500" would crash the request instead of quietly giving a
        # 30x30 board and saying so.
        if param.type in ("int", "float"):
            try:
                raw = param.coerce(patch.value)
            except ValueError:
                value = float(str(patch.value).strip())
                if param.min is not None and value < param.min:
                    note = f" (clamped to the minimum {param.min})"
                    value = float(param.min)
                elif param.max is not None and value > param.max:
                    note = f" (clamped to the maximum {param.max})"
                    value = float(param.max)
                else:
                    raise
                raw = param.coerce(int(value) if param.type == "int" else value)
        else:
            raw = param.coerce(patch.value)
    elif patch.op in ("scale", "offset"):
        if param.type not in ("int", "float"):
            raise CustomizationError(
                f"{param.name}: {patch.op} needs a numeric parameter, "
                f"but {param.name} is {param.type}"
            )
        base = param.coerce(current.get(param.name, param.default))
        raw = base * patch.factor if patch.op == "scale" else base + patch.factor
        if param.type == "int":
            raw = int(round(raw))
        # a relative change that lands outside the range is clamped, and the
        # user is told -- silently ignoring "much bigger" is worse than saying so
        if param.min is not None and raw < param.min:
            note = f" (clamped to the minimum {param.min})"
            raw = param.min
        if param.max is not None and raw > param.max:
            note = f" (clamped to the maximum {param.max})"
            raw = param.max
        raw = param.coerce(raw)
    else:
        raise CustomizationError(f"unknown operation {patch.op!r}")

    updated = dict(current)
    updated[param.name] = raw
    return updated, note
