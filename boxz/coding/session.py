"""Per-conversation coding state: which template, which values.

A session is deliberately tiny and JSON-serialisable so it can live inside
``ChatEngine`` without touching the model:

    {"template_id": "js_snake", "values": {"step_ms": 70, ...}, "version": "1.0.0"}

Follow-ups ("make it bigger") are applied to ``values`` and never re-select a
template.  A brand-new coding request replaces the state; a conversational
turn leaves it untouched, so "what is CSS?" does not destroy the snake the
user is editing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .customize import CustomizationError, Patch, apply_patch
from .templates import Template

__all__ = ["CodingSession", "FollowUp", "parse_followup"]

#: Verbs that mark a follow-up rather than a new request.
_FOLLOWUP_START = re.compile(
    r"^\s*(make|change|set|turn|add|remove|increase|decrease|reduce|speed|"
    r"slow|use|switch|rename|move|put|give|let)\b", re.I)

#: "make it bigger", "make everything smoother", "a bit faster"
_MAGNITUDE = {
    "much": 2.0, "a lot": 2.0, "way": 2.0, "slightly": 1.15, "a bit": 1.15,
    "a little": 1.15, "little": 1.15, "bit": 1.15, "very": 1.5, "really": 1.5,
}

#: Words that map onto a relative change of a numeric parameter.
_RELATIVE_WORDS = {
    "bigger": ("size", "scale", 1.4),
    "larger": ("size", "scale", 1.4),
    "smaller": ("size", "scale", 0.7),
    "tiny": ("size", "scale", 0.6),
    "huge": ("size", "scale", 1.8),
    "faster": ("speed", "scale", 0.75),      # step_ms down == faster
    "quicker": ("speed", "scale", 0.75),
    "slower": ("speed", "scale", 1.4),
    "more": ("count", "scale", 1.6),
    "fewer": ("count", "scale", 0.6),
    "less": ("count", "scale", 0.6),
}


@dataclass
class FollowUp:
    """A parsed follow-up instruction."""

    patch: Optional[Patch] = None
    text: str = ""
    ambiguous: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "ambiguous": self.ambiguous,
                "note": self.note,
                "patch": ({"param": self.patch.param, "op": self.patch.op,
                           "value": self.patch.value, "factor": self.patch.factor}
                          if self.patch else None)}


@dataclass
class CodingSession:
    """Active template plus its current parameter values."""

    template_id: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)
    version: str = ""
    turns: int = 0

    @property
    def active(self) -> bool:
        return self.template_id is not None

    def start(self, template: Template, values: Optional[Dict[str, Any]] = None) -> None:
        self.template_id = template.id
        self.version = template.version
        self.values = dict(values if values is not None else template.defaults())

    def add_turn(self) -> None:
        self.turns += 1

    def clear(self) -> None:
        self.template_id = None
        self.values = {}
        self.version = ""
        self.turns = 0

    def apply(self, template: Template, follow: FollowUp) -> Tuple[bool, str]:
        """Apply a follow-up.  Returns (changed, human note)."""
        if follow.patch is None:
            return False, follow.note or "I could not tell what to change."
        try:
            new_values, clamp_note = apply_patch(template, self.values, follow.patch)
        except CustomizationError as exc:
            return False, str(exc)
        param = template.param(follow.patch.param)
        name = param.name if param else follow.patch.param
        before = self.values.get(name)
        self.values = new_values
        after = new_values.get(name)
        note = f"{name}: {_fmt(before)} -> {_fmt(after)}{clamp_note}"
        return True, note

    def to_dict(self) -> dict:
        return {"template_id": self.template_id, "values": self.values,
                "version": self.version, "turns": self.turns}

    @classmethod
    def from_dict(cls, d: dict) -> "CodingSession":
        return cls(template_id=d.get("template_id"), values=dict(d.get("values", {})),
                   version=d.get("version", ""), turns=int(d.get("turns", 0)))


# ---------------------------------------------------------------------------
# follow-up parsing
# ---------------------------------------------------------------------------

_COLOR_WORDS = {
    "red": "#ff4136", "blue": "#0074d9", "green": "#2ecc40", "yellow": "#ffdc00",
    "orange": "#ff851b", "purple": "#b10dc9", "pink": "#f012be",
    "white": "#ffffff", "black": "#000000", "gray": "#aaaaaa", "grey": "#aaaaaa",
    "cyan": "#7fdbff", "teal": "#39cccc", "lime": "#01ff70",
}


def _fmt(value: Any) -> str:
    """Human-friendly number formatting for the change note."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _resolve_param(template: Template, word: str) -> Optional[str]:
    """Map a word from the request onto a real parameter name."""
    word = word.lower().strip()
    if word in ("bigger", "larger", "smaller", "huge", "tiny", "size"):
        for cand in ("size", "particle_radius", "paddle_h", "cell_px",
                     "font_px", "display_font_px", "count_font_px", "grid_size"):
            if template.param(cand):
                return cand
        return None
    if word in ("faster", "quicker", "slower", "speed"):
        for cand in ("step_ms", "speed", "ball_speed", "duration_ms", "rotation_speed"):
            if template.param(cand):
                return cand
        return None
    if word in ("more", "fewer", "less", "many", "amount", "count"):
        for cand in ("particle_count", "count", "start_length", "win_score"):
            if template.param(cand):
                return cand
        return None
    if word in _COLOR_WORDS or word in ("colour", "color"):
        for cand in ("snake_color", "color", "accent_color", "particle_color",
                     "ball_color", "paddle_color", "count_color", "bg_color"):
            if template.param(cand):
                return cand
        return None
    direct = template.param(word)
    return direct.name if direct else None


def parse_followup(text: str, template: Template) -> FollowUp:
    """Turn "make it bigger" / "change the background to blue" into a Patch."""
    raw = (text or "").strip()
    low = raw.lower()

    if not _FOLLOWUP_START.match(raw):
        return FollowUp(ambiguous=True, text=raw,
                        note="not phrased as a change to the current project")

    # --- colour: "make it blue", "background to red", "colour = #00ff00" ---
    hexmatch = re.search(r"#[0-9a-f]{3,6}\b", low)
    if hexmatch:
        target = _resolve_param(template, "color")
        if target:
            return FollowUp(patch=Patch(param=target, op="set", value=hexmatch.group(0)),
                            text=raw)
    for colour in _COLOR_WORDS:
        if re.search(r"(?<![a-z])" + colour + r"(?![a-z])", low):
            # prefer the parameter the sentence names, else the main colour
            target = None
            for word in ("background", "bg", "text", "border", "accent"):
                if word in low:
                    named = template.param(word)
                    if named:
                        target = named.name
                        break
            target = target or _resolve_param(template, "color")
            if target:
                return FollowUp(patch=Patch(param=target, op="set",
                                            value=_COLOR_WORDS[colour]), text=raw)

    # --- explicit parameter mention: "make step_ms 60" / "set grid to 20" ---
    for name in list(template.param_names()):
        if name in low:
            num = re.search(r"(-?\d+(?:\.\d+)?)", low)
            if num:
                return FollowUp(patch=Patch(param=name, op="set",
                                            value=num.group(1)), text=raw)

    # --- relative words: "make it bigger", "much faster" ---
    magnitude = 1.0
    for word, factor in _MAGNITUDE.items():
        if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low):
            magnitude = factor
            break
    for word, (_hint, _op, factor) in _RELATIVE_WORDS.items():
        if re.search(r"(?<![a-z])" + word + r"(?![a-z])", low):
            target = _resolve_param(template, _hint)
            if not target:
                continue
            param = template.param(target)
            if param is None or param.type not in ("int", "float"):
                continue
            # "faster" on a millisecond interval means a smaller number, so the
            # direction is carried by the factor itself.
            return FollowUp(patch=Patch(param=target, op="scale",
                                        factor=factor * magnitude), text=raw)

    # --- "add a score" style: unsupported feature on this template ---
    if re.match(r"^\s*add\b", raw, re.I):
        return FollowUp(ambiguous=True, text=raw,
                        note=("adding a feature is not a supported parameter "
                              "change for this template"))

    return FollowUp(ambiguous=True, text=raw,
                    note="I could not map that to a supported parameter")
