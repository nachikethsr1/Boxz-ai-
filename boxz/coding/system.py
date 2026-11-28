"""The coding system: request -> template -> files -> validated project.

This is the layer the chat engine talks to.  It owns the decision order and
nothing else:

1. **Do we already have an active template?**  Then an instruction phrased as a
   change ("make it faster") is applied to it, and the project is rebuilt.  If
   it is clearly a *new* request ("make a pong game") the session is replaced.
2. **Otherwise classify** with the deterministic matcher.
3. ``match``       -> render, validate, return a project
4. ``clarify``     -> return the 2-3 closest real templates so the user can pick
5. ``unsupported`` -> say which technology we cannot do, and why
6. ``none``        -> hand the turn back to the neural model

The ``source`` field on every result says exactly where the code came from:
``template``, ``template+followup``, or ``model``.  Nothing in this module
ever claims a project was browser-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .customize import CustomizationError, apply_values
from .library import ProjectValidation, validate_project
from .requests import RequestCorpus, parse_coding_requests
from .selection import Selection, TemplateSelector
from .session import CodingSession, FollowUp, parse_followup
from .templates import Template, load_library

__all__ = ["CodingResult", "CodingSystem", "build_default_system"]

_REASON_TEXT = {
    "language_not_supported":
        "I only build JavaScript, HTML and CSS projects right now. That "
        "language is outside the template library.",
    "technology_not_supported":
        "That technology (3D/WebGL, a game engine, a native app) is not in the "
        "template library, so I would be guessing.",
    "framework_not_supported":
        "I do not have templates for that framework. I can build plain "
        "HTML/CSS/JavaScript projects.",
}


@dataclass
class CodingResult:
    """Everything one coding turn produced."""

    handled: bool = False                  # did the coding system answer?
    outcome: str = "none"                  # match | clarify | unsupported | none
    source: str = ""                       # template | template+followup | model
    template_id: Optional[str] = None
    text: str = ""                         # what to say to the user
    project: Optional[Any] = None          # GeneratedProject
    validation: Optional[ProjectValidation] = None
    session: Optional[CodingSession] = None
    selection: Optional[Selection] = None
    followup: Optional[FollowUp] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "handled": self.handled,
            "outcome": self.outcome,
            "source": self.source,
            "template_id": self.template_id,
            "text": self.text,
            "session": self.session.to_dict() if self.session else None,
        }
        if self.project is not None:
            d["project"] = self.project.to_artifacts()
        if self.validation is not None:
            d["validation"] = self.validation.to_dict()
        if self.selection is not None:
            d["selection"] = self.selection.to_dict()
        return d


_NEW_REQUEST = re.compile(
    r"^\s*(make|create|build|write|start|generate|give me|i want|can you make|"
    r"now make|also make|instead|new)\b", re.I)


def _is_explicit_new_request(text: str) -> bool:
    """True only when the user clearly asks for a *different* project.

    A follow-up that mentions a feature of the current project ("add a score")
    must not be mistaken for a request for a different template.
    """
    if not _NEW_REQUEST.match(text or ""):
        return False
    # "make it faster" starts with "make" but is not a new project
    return not re.match(r"^\s*(make|turn|set|change)\s+(it|this|that|them)\b",
                        text or "", re.I)


_CHANGE_START = re.compile(
    r"^\s*(add|remove|improve|change|make|turn|set|increase|decrease|"
    r"reduce|speed|slow|rename|move|put|give|let|use|switch)\b", re.I)


def _is_change_instruction(text: str) -> bool:
    """True for "add a score" / "make it smoother", false for "what is CSS?"."""
    return bool(_CHANGE_START.match(text or ""))


def _sentence(text: str) -> str:
    """Capitalise the first letter and end with a full stop."""
    t = (text or "").strip()
    if not t:
        return ""
    t = t[:1].upper() + t[1:]
    return t if t.endswith((".", "!", "?")) else t + "."


def _fmt_value(value: Any) -> str:
    """Print 3.15, not 3.1500000000000004."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _describe_files(project) -> str:
    files = ", ".join(f"{p}" for p, _l, _b in project.items())
    return f"Files: {files}."


class CodingSystem:
    """Deterministic template selection, customization and validation."""

    def __init__(
        self,
        templates: Dict[str, Template],
        corpus: RequestCorpus,
        *,
        min_score: float = 0.16,
        min_margin: float = 0.35,
    ) -> None:
        self.templates = templates
        self.corpus = corpus
        self.selector = TemplateSelector(corpus, templates,
                                         min_score=min_score,
                                         min_margin=min_margin)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(
        cls,
        requests_file: str | Path = "coding_requests.txt",
        templates_root: str | Path = "data/templates",
        **kwargs: Any,
    ) -> "CodingSystem":
        corpus = parse_coding_requests(requests_file)
        lib = load_library(templates_root)
        if not lib:
            raise ValueError(f"no templates found under {templates_root}")
        return cls(lib, corpus, **kwargs)

    # -- public API ------------------------------------------------------
    def available(self) -> List[dict]:
        return [
            {"id": t.id, "title": t.title, "category": t.category,
             "summary": t.summary, "parameters": [p.name for p in t.parameters],
             "status": t.status}
            for t in sorted(self.templates.values(), key=lambda t: t.id)
        ]

    def handle(self, message: str, session: Optional[CodingSession] = None) -> CodingResult:
        """Run one user turn through the coding system."""
        session = session or CodingSession()
        text = (message or "").strip()
        if not text:
            return CodingResult(handled=False, outcome="none", session=session)

        # ---- 1. an active template turns this into a follow-up? --------
        if session.active and session.template_id in self.templates:
            tpl = self.templates[session.template_id]
            follow = parse_followup(text, tpl)
            if follow.patch is not None:
                return self._followup(text, tpl, session, follow)

            # Not a parameter change.  Is it a *new project*, or an
            # instruction about the current one?  The distinction matters:
            # "add a score" must not silently replace the user's snake with a
            # click counter just because the word "counter" matched.
            fresh = self.selector.classify(text)
            new_project = (fresh.outcome == "match"
                           and fresh.template_id
                           and fresh.template_id != session.template_id
                           and _is_explicit_new_request(text))
            if not new_project:
                # A genuine change request ("add a score") gets an explanation.
                # Anything else -- a question, a topic change -- must fall
                # through to the model, keeping the project intact but not
                # answering a question about CSS with a list of parameters.
                if not follow.note or not _is_change_instruction(text):
                    return CodingResult(handled=False, outcome="none",
                                        session=session, followup=follow)
                return CodingResult(
                    handled=True, outcome="clarify", source="template",
                    template_id=tpl.id, session=session, followup=follow,
                    text=(f"I kept your {tpl.title}. {_sentence(follow.note)} "
                          f"Supported changes here: "
                          f"{', '.join(p.name for p in tpl.parameters)}."))

        # ---- 2. fresh classification ----------------------------------
        selection = self.selector.classify(text)

        if selection.outcome == "unsupported":
            reason = selection.reason
            return CodingResult(
                handled=True, outcome="unsupported", source="template",
                text=_REASON_TEXT.get(reason, "That request is outside the "
                                              "template library."),
                session=session, selection=selection)

        if selection.outcome == "clarify":
            return self._clarify(session, selection)

        if selection.outcome != "match" or not selection.template_id:
            # not a coding request at all -> the model answers
            return CodingResult(handled=False, outcome="none",
                                session=session, selection=selection)

        return self._build(selection.template_id, {}, session, selection,
                           source="template")

    # -- internals -------------------------------------------------------
    def _followup(self, text: str, tpl: Template, session: CodingSession,
                  follow: FollowUp) -> CodingResult:
        changed, note = session.apply(tpl, follow)
        if not changed:
            return CodingResult(
                handled=True, outcome="clarify", source="template",
                template_id=tpl.id, session=session, followup=follow,
                text=f"{note} Supported changes: "
                     f"{', '.join(p.name for p in tpl.parameters)}.")
        result = self._build(tpl.id, session.values, session, None,
                             source="template+followup")
        result.followup = follow
        result.text = _sentence(note) + " " + result.text
        return result

    def _build(self, template_id: str, values: Dict[str, Any],
               session: CodingSession, selection: Optional[Selection],
               source: str) -> CodingResult:
        tpl = self.templates[template_id]
        try:
            custom = apply_values(tpl, values, strict=False)
        except CustomizationError as exc:            # pragma: no cover
            return CodingResult(handled=True, outcome="clarify", source=source,
                                template_id=template_id, session=session,
                                text=f"I could not build that: {exc}")

        session.start(tpl, custom.project.values)
        report = validate_project(custom.project, tpl)

        parts = [f"{tpl.title}. {_describe_files(custom.project)}",
                 report.verdict()]
        if custom.rejected:
            rejected = "; ".join(custom.rejected.values())
            parts.append(f"I ignored some values that did not fit: {rejected}")
        parts.append("Open " + custom.project.entry() + " in a browser to try it.")

        return CodingResult(
            handled=True, outcome="match", source=source,
            template_id=template_id, project=custom.project,
            validation=report, session=session, selection=selection,
            text=" ".join(parts))

    def _clarify(self, session: CodingSession, selection: Selection) -> CodingResult:
        cands = [c for c in selection.candidates if c[0] in self.templates][:3]
        if not cands:
            return CodingResult(
                handled=True, outcome="clarify", source="template",
                session=session, selection=selection,
                text=("I do not have a template for that yet. Tell me the "
                      "kind of thing you want (a game, a page, an animation) "
                      "and I will check again."))
        names = ", ".join(f"{self.templates[tid].title} ({tid})" for tid, _ in cands)
        return CodingResult(
            handled=True, outcome="clarify", source="template",
            session=session, selection=selection,
            text=(f"That could be more than one thing. Did you mean {names}? "
                  f"Tell me which one."))


def build_default_system(
    requests_file: str | Path = "coding_requests.txt",
    templates_root: str | Path = "data/templates",
) -> Optional[CodingSystem]:
    """Build the system from the repo, or return None if the data is absent.

    Returning ``None`` (rather than raising) keeps Boxz usable on a machine
    where the template library was not shipped -- the model still answers.
    """
    try:
        return CodingSystem.from_files(requests_file, templates_root)
    except (FileNotFoundError, ValueError):
        return None
