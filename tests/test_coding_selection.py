"""Router tests: does a request reach the right template, and does it refuse?

The corpus in ``coding_requests.txt`` is the specification.  These tests turn
it into a regression suite: every request for an implemented template must
route to that template, and the edge-case blocks must produce the documented
outcome rather than a confident guess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from boxz.coding.requests import parse_coding_requests
from boxz.coding.selection import TemplateSelector
from boxz.coding.templates import load_library

REQUESTS = Path("coding_requests.txt")
TEMPLATES = Path("data/templates")

IMPLEMENTED = {"js_snake", "js_pong", "js_click_counter", "js_timer_app",
               "canvas_particle_effect", "css_button_animation"}


@pytest.fixture(scope="module")
def setup():
    corpus = parse_coding_requests(REQUESTS)
    lib = load_library(TEMPLATES)
    return corpus, lib, TemplateSelector(corpus, lib)


# ------------------------------------------------------- corpus accuracy
def test_accuracy_on_implemented_templates(setup):
    """Every request in the corpus for a real template must route to it."""
    corpus, _lib, selector = setup
    total = correct = 0
    misses = []
    for block in corpus.templates:
        if block.id not in IMPLEMENTED:
            continue
        for request in block.requests:
            total += 1
            result = selector.classify(request)
            if result.outcome == "match" and result.template_id == block.id:
                correct += 1
            else:
                misses.append((block.id, result.outcome, result.template_id, request))
    assert total > 50, total
    rate = correct / total
    assert rate >= 0.95, (
        f"{correct}/{total} = {rate:.1%}; misses: {misses[:5]}")


def test_never_routes_to_an_unimplemented_template(setup):
    corpus, lib, selector = setup
    for block in corpus.templates:
        for request in block.requests:
            r = selector.classify(request)
            if r.template_id is not None:
                assert r.template_id in lib, (request, r.template_id)


# ------------------------------------------------------- direct phrasings
@pytest.mark.parametrize("phrase,template", [
    ("Make a snake game where the tail grows", "js_snake"),
    ("Create a two-paddle Pong game with a ball", "js_pong"),
    ("Make a stopwatch with start, pause, lap, and reset", "js_timer_app"),
    ("Create a Canvas particle effect with tiny dots", "canvas_particle_effect"),
    ("Make a button that grows when I hover over it", "css_button_animation"),
    ("Create a click counter with plus, minus, and reset", "js_click_counter"),
])
def test_representative_requests(setup, phrase, template):
    _corpus, _lib, selector = setup
    assert selector.classify(phrase).template_id == template


# ------------------------------------------------------- edge cases
@pytest.mark.parametrize("phrase", [
    "Write a small Python game with keyboard controls.",
    "Create a Java program that manages a student list.",
    "Build a Rust command-line application for tracking tasks.",
    "Write a CUDA program for GPU-based image processing.",
])
def test_unsupported_languages(setup, phrase):
    _corpus, _lib, selector = setup
    r = selector.classify(phrase)
    assert r.outcome == "unsupported", (phrase, r.to_dict())
    assert r.reason == "language_not_supported"


@pytest.mark.parametrize("phrase", [
    "Make a 3D scene with Three.js and a rotating camera.",
    "Build a native Android application with Kotlin.",
    "Make a virtual reality scene with interactive 3D objects.",
])
def test_unsupported_technologies(setup, phrase):
    _corpus, _lib, selector = setup
    r = selector.classify(phrase)
    assert r.outcome == "unsupported", (phrase, r.to_dict())


@pytest.mark.parametrize("phrase", [
    "Create a React dashboard with reusable components.",
    "Build a Vue application with several routed pages.",
    "Make a Node.js server with an Express API.",
    "Build a Django web application with user login.",
])
def test_unsupported_frameworks(setup, phrase):
    _corpus, _lib, selector = setup
    r = selector.classify(phrase)
    assert r.outcome == "unsupported", (phrase, r.to_dict())
    assert r.reason == "framework_not_supported"


@pytest.mark.parametrize("phrase", [
    "Make me a JavaScript game but I have not decided what kind yet.",
    "I want a browser game with some fun mechanics.",
    "Can you build a small game for my website?",
])
def test_vague_game_requests_ask_for_clarification(setup, phrase):
    _corpus, _lib, selector = setup
    r = selector.classify(phrase)
    assert r.outcome == "clarify", (phrase, r.to_dict())


@pytest.mark.parametrize("phrase", [
    "hi", "hello bro", "how are you?", "thanks for helping me",
])
def test_conversation_is_not_coding(setup, phrase):
    _corpus, _lib, selector = setup
    assert selector.classify(phrase).outcome == "none"


# ------------------------------------------------------- no guessing
def test_ambiguous_request_returns_two_candidates(setup):
    _corpus, _lib, selector = setup
    r = selector.classify("make a game with a ball")
    assert r.outcome in ("clarify", "match")
    if r.outcome == "clarify":
        assert r.runner_up


def test_empty_request_is_not_a_match(setup):
    _corpus, _lib, selector = setup
    r = selector.classify("")
    assert not r.matched and r.outcome == "none"


def test_scores_are_deterministic(setup):
    _corpus, _lib, selector = setup
    text = "Make a snake game where the tail grows"
    a = selector.classify(text).to_dict()
    b = selector.classify(text).to_dict()
    assert a == b


def test_unimplemented_template_requests_do_not_crash(setup):
    """Requests for the other 60 templates must degrade gracefully."""
    corpus, lib, selector = setup
    for block in corpus.templates:
        if block.id in IMPLEMENTED:
            continue
        for request in block.requests[:2]:
            r = selector.classify(request)
            assert r.outcome in ("match", "clarify", "unsupported", "none")
            if r.template_id is not None:
                assert r.template_id in lib


# ------------------------------------------------- English vs technology
@pytest.mark.parametrize("phrase", [
    # "react" as a verb, "go"/"on" as English, Windows as a UI element
    "Make the page react to the mouse.",
    "I need a personal site but I am not sure what should go on it.",
    "Make windows pop up on click",
    "Add some clicking actions to the interface.",
])
def test_english_words_are_not_tech_names(setup, phrase):
    _corpus, _lib, selector = setup
    r = selector.classify(phrase)
    assert r.outcome != "unsupported", (phrase, r.reason)


@pytest.mark.parametrize("phrase", [
    "Make a C program that simulates a simple queue.",
    "Write the app in Go",
    "Create a React dashboard with reusable components.",
    "Build a website with React",
    "Make a Windows desktop application with a native interface.",
])
def test_real_tech_mentions_are_still_refused(setup, phrase):
    _corpus, _lib, selector = setup
    assert selector.classify(phrase).outcome == "unsupported", phrase


def test_wish_with_no_object_asks_a_question(setup):
    _corpus, _lib, selector = setup
    r = selector.classify("i want to make something cool")
    assert r.outcome == "clarify"
    assert r.reason == "task_not_specified"


def test_greeting_is_not_a_clarification(setup):
    """Greetings must reach the model; answering with a template menu is wrong."""
    _corpus, _lib, selector = setup
    assert selector.classify("hi").outcome == "none"
    assert selector.classify("good morning").outcome == "none"


# --------------------------------------------------- keyword vs vector fight
def test_incidental_keyword_does_not_win(setup):
    """"counts clicks on each colored button" is a counter, not a button."""
    _corpus, _lib, selector = setup
    r = selector.classify("Build a page that counts how many times each "
                          "colored button has been clicked.")
    assert r.template_id == "js_click_counter"


def test_hedged_request_asks_instead_of_building(setup):
    """'a button or something' is a wish, not a specification."""
    _corpus, _lib, selector = setup
    for phrase in ("Add a popup or something when I press a button.",
                   "Make an arcade game with a character and some challenges."):
        r = selector.classify(phrase)
        assert r.outcome == "clarify", (phrase, r.to_dict())


def test_no_clarify_case_is_silently_built(setup):
    """The worst possible failure: building an unrelated project."""
    corpus, lib, selector = setup
    wrong = []
    for block in corpus.clarifications:
        for phrase in block.requests:
            r = selector.classify(phrase)
            if r.outcome == "match":
                wrong.append((phrase, r.template_id))
    assert not wrong, wrong
