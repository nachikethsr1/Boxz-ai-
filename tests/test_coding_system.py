"""End-to-end tests for the coding system: session state and preservation.

The two things most likely to break silently are (a) a follow-up replacing the
project instead of editing it, and (b) the coding system stealing a maths or
chat turn.  Both are covered here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from boxz.coding.session import CodingSession, parse_followup
from boxz.coding.system import CodingSystem, build_default_system

REQUESTS = Path("coding_requests.txt")
TEMPLATES = Path("data/templates")


@pytest.fixture(scope="module")
def system():
    sys_ = build_default_system(REQUESTS, TEMPLATES)
    assert sys_ is not None, "coding system failed to load"
    return sys_


@pytest.fixture
def session():
    return CodingSession()


# ------------------------------------------------------------ basic build
def test_match_builds_a_project(system, session):
    r = system.handle("Make a snake game where the tail grows", session)
    assert r.handled and r.outcome == "match"
    assert r.template_id == "js_snake"
    assert r.source == "template"
    assert r.project is not None and len(r.project.files) == 3


def test_built_project_passes_validation(system, session):
    r = system.handle("Make a snake game", session)
    assert r.validation is not None
    assert r.validation.ok
    assert all(f.ok for f in r.validation.files)


def test_validation_never_claims_browser_testing(system, session):
    r = system.handle("Make a snake game", session)
    assert r.validation.browser_tested is False
    assert "Nothing was run in a browser" in r.validation.verdict()
    assert "tag balance" in r.validation.verdict()


def test_session_records_the_template(system, session):
    system.handle("Make a snake game", session)
    assert session.active and session.template_id == "js_snake"
    assert session.values["step_ms"] == 120


# ------------------------------------------------------------ follow-ups
def test_followup_changes_the_same_project(system, session):
    system.handle("Make a snake game", session)
    r = system.handle("Make it faster.", session)
    assert r.template_id == "js_snake"
    assert r.source == "template+followup"
    assert session.values["step_ms"] == 90
    assert "var STEP_MS = 90;" in r.project.files["game.js"]


def test_followup_preserves_other_parameters(system, session):
    system.handle("Make a snake game", session)
    session.values["snake_color"] = "#0074d9"
    system.handle("Make it faster.", session)
    assert session.values["snake_color"] == "#0074d9"
    assert session.values["step_ms"] == 90


def test_colour_followup(system, session):
    system.handle("Make a snake game", session)
    r = system.handle("Change the color to blue.", session)
    assert session.values["snake_color"] == "#0074d9"
    assert "--accent: #0074d9;" in r.project.files["style.css"]


def test_repeated_followups_accumulate(system, session):
    system.handle("Make a snake game", session)
    system.handle("Make it faster.", session)
    system.handle("Make it faster.", session)
    assert session.values["step_ms"] == 68


def test_clamped_followup_is_reported(system, session):
    system.handle("Make a snake game", session)
    r = system.handle("Make it much bigger.", session)
    assert session.values["cell_px"] == 40
    assert "clamped" in r.text.lower()


def test_unsupported_feature_is_refused_not_built(system, session):
    system.handle("Make a snake game", session)
    r = system.handle("Add a score.", session)
    assert r.template_id == "js_snake"          # project kept
    assert "not a supported parameter" in r.text
    assert session.values["step_ms"] == 120      # nothing changed


def test_new_request_replaces_the_project(system, session):
    system.handle("Make a snake game", session)
    r = system.handle("now make a pong game", session)
    assert r.template_id == "js_pong"
    assert session.template_id == "js_pong"
    assert session.values["win_score"] == 7


def test_followup_without_session_does_not_invent_a_project(system, session):
    r = system.handle("Make it faster.", session)
    assert r.outcome in ("clarify", "none")
    assert r.project is None


# ------------------------------------------------------------ preservation
@pytest.mark.parametrize("question", [
    "482736 + 918264",
    "what is 17% of 2480",
    "solve 4x - 9 = 27",
])
def test_maths_never_enters_the_coding_system(system, session, question):
    r = system.handle(question, session)
    assert not r.handled, f"{question!r} was claimed by the coding system"


@pytest.mark.parametrize("question", [
    "hi", "hello bro", "thanks for helping me", "how are you?",
])
def test_conversation_never_enters_the_coding_system(system, session, question):
    assert not system.handle(question, session).handled


def test_question_during_a_coding_session_falls_through(system, session):
    """'What is CSS?' must not be answered with a parameter list."""
    system.handle("Make a snake game", session)
    r = system.handle("What is CSS?", session)
    assert not r.handled
    assert session.template_id == "js_snake"     # project survives


def test_maths_during_a_coding_session_still_falls_through(system, session):
    system.handle("Make a snake game", session)
    assert not system.handle("482736 + 918264", session).handled


# ------------------------------------------------------------ unsupported
def test_unsupported_language_is_declined(system, session):
    r = system.handle("Write a small Python game with keyboard controls.", session)
    assert r.outcome == "unsupported"
    assert r.project is None
    assert "JavaScript" in r.text


def test_unsupported_framework_is_declined(system, session):
    r = system.handle("Create a React dashboard with reusable components.", session)
    assert r.outcome == "unsupported"
    assert "framework" in r.text.lower()


def test_vague_request_offers_real_candidates(system, session):
    r = system.handle("I want a browser game with some fun mechanics.", session)
    assert r.outcome == "clarify"
    assert r.project is None
    # it must name templates that actually exist
    assert any(tid in r.text for tid in
               ("js_snake", "js_pong", "js_click_counter", "js_timer_app"))


# ------------------------------------------------------------ contract
def test_result_is_json_serialisable(system, session):
    import json
    r = system.handle("Make a snake game", session)
    blob = json.dumps(r.to_dict())
    assert "index.html" in blob
    assert "js_snake" in blob


def test_source_is_always_labelled(system, session):
    r = system.handle("Make a snake game", session)
    assert r.source in ("template", "template+followup")
    assert r.to_dict()["source"] == r.source


def test_session_survives_serialisation(system, session):
    system.handle("Make a snake game", session)
    system.handle("Make it faster.", session)
    restored = CodingSession.from_dict(session.to_dict())
    assert restored.template_id == "js_snake"
    assert restored.values["step_ms"] == 90


def test_available_lists_the_library(system):
    listing = system.available()
    assert len(listing) >= 6
    for item in listing:
        assert item["id"] and item["title"]
        assert isinstance(item["parameters"], list)
