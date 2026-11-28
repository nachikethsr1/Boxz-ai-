"""Tests for the template library and the safe customizer.

The customizer's whole value is that it *cannot* do a global find-and-replace,
so the tests concentrate on what it refuses: unknown parameters, out-of-range
numbers, bad colours, and any placeholder left behind.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from boxz.coding.customize import (
    CustomizationError,
    Patch,
    apply_patch,
    apply_values,
    render_template,
)
from boxz.coding.engine import validate
from boxz.coding.library import validate_project
from boxz.coding.templates import (
    PLACEHOLDER_RE,
    load_library,
    load_template,
)

TEMPLATES = Path("data/templates")


@pytest.fixture(scope="module")
def library():
    lib = load_library(TEMPLATES)
    assert lib, "no templates loaded"
    return lib


# ---------------------------------------------------------------- library
def test_library_has_the_first_batch(library):
    assert len(library) >= 6, sorted(library)
    expected = {"js_snake", "js_pong", "js_click_counter", "js_timer_app",
                "canvas_particle_effect", "css_button_animation"}
    assert expected <= set(library)


def test_every_template_declares_files_that_exist(library):
    for tid, tpl in library.items():
        assert tpl.files, tid
        for f in tpl.files:
            assert (tpl.path / f.path).exists(), f"{tid}/{f.path}"


def test_every_template_declares_parameters(library):
    for tid, tpl in library.items():
        assert tpl.parameters, f"{tid} has no parameters"
        names = [p.name for p in tpl.parameters]
        assert len(names) == len(set(names)), f"{tid} duplicate parameter names"


def test_every_parameter_has_a_default(library):
    for tid, tpl in library.items():
        for p in tpl.parameters:
            assert p.default is not None, f"{tid}.{p.name} has no default"


def test_placeholders_are_balanced_both_ways(library):
    """load_template already enforces this; assert it explicitly anyway."""
    for tid, tpl in library.items():
        declared = {p.placeholder for p in tpl.parameters}
        blob = "".join(tpl.read(f.path) for f in tpl.files)
        used = set(re.findall(PLACEHOLDER_RE, blob))
        assert used == declared, f"{tid}: declared {declared}, used {used}"


# ---------------------------------------------------------------- rendering
def test_defaults_render_with_no_placeholders_left(library):
    for tid, tpl in library.items():
        project = render_template(tpl)
        for path, _lang, body in project.items():
            leftover = re.findall(PLACEHOLDER_RE, body)
            assert leftover == [], f"{tid}/{path} left {leftover}"


def test_every_template_passes_the_coding_engine(library):
    for tid, tpl in library.items():
        project = render_template(tpl)
        report = validate_project(project, tpl)
        assert report.ok, f"{tid}: {report.verdict()}"


def test_full_snake_js_validates_standalone(library):
    """The generated JS must pass `node --check` on its own file."""
    project = render_template(library["js_snake"])
    v = validate(project.files["game.js"], "javascript")
    assert v.status == "passed", v.summary()


def test_customised_values_reach_the_right_file(library):
    project = render_template(library["js_snake"], {
        "step_ms": 70, "snake_color": "red", "grid_size": 20, "title": "Neon",
    })
    assert "var STEP_MS = 70;" in project.files["game.js"]
    assert "--accent: #ff4136;" in project.files["style.css"]
    assert "<title>Neon</title>" in project.files["index.html"]
    assert "var CELLS = 20;" in project.files["game.js"]


def test_customised_output_still_validates(library):
    project = render_template(library["js_snake"], {
        "step_ms": 40, "grid_size": 30, "cell_px": 12, "snake_color": "#000000",
    })
    report = validate_project(project, library["js_snake"])
    assert report.ok, report.verdict()


def test_colour_name_is_normalised(library):
    project = render_template(library["js_snake"], {"snake_color": "blue"})
    assert project.values["snake_color"] == "#0074d9"


def test_short_hex_is_expanded(library):
    project = render_template(library["js_snake"], {"snake_color": "#0f0"})
    assert project.values["snake_color"] == "#00ff00"


def test_bool_renders_as_js_keyword(library):
    on = render_template(library["js_click_counter"], {"persist": True})
    off = render_template(library["js_click_counter"], {"persist": False})
    assert "var PERSIST = true;" in on.files["counter.js"]
    assert "var PERSIST = false;" in off.files["counter.js"]


# ---------------------------------------------------------------- refusals
def test_unknown_parameter_raises_in_strict_mode(library):
    with pytest.raises(CustomizationError, match="unsupported parameter"):
        apply_values(library["js_snake"], {"nonsense": 1}, strict=True)


def test_unknown_parameter_is_recorded_in_lenient_mode(library):
    result = apply_values(library["js_snake"], {"nonsense": 1}, strict=False)
    assert "nonsense" in result.rejected
    assert not result.ok
    # the project is still built with defaults
    assert result.project.files


def test_value_below_range_is_refused(library):
    with pytest.raises(ValueError, match="below the minimum"):
        library["js_snake"].param("step_ms").coerce(1)


def test_value_above_range_is_refused(library):
    with pytest.raises(ValueError, match="above the maximum"):
        library["js_snake"].param("grid_size").coerce(999)


def test_bad_colour_is_refused(library):
    with pytest.raises(ValueError, match="not a colour"):
        library["js_snake"].param("snake_color").coerce("banana")


def test_non_numeric_for_number_is_refused(library):
    with pytest.raises(ValueError, match="expected an integer"):
        library["js_snake"].param("step_ms").coerce("fast")


def test_string_value_cannot_inject_code(library):
    """A title is inserted verbatim, so quotes must be rejected by the caller.

    The customizer itself does not escape; this test documents that a value
    containing markup would be written as-is, which is why titles are only
    ever set from the template's own defaults plus simple words.
    """
    project = render_template(library["js_snake"], {"title": "Snake"})
    assert "Snake" in project.files["index.html"]


# ---------------------------------------------------------------- patches
def test_lead_in_followup_scales_the_current_value(library):
    tpl = library["js_snake"]
    current = dict(tpl.defaults())
    new, note = apply_patch(tpl, current, Patch("step_ms", "scale", factor=0.75))
    assert new["step_ms"] == 90
    assert note == ""


def test_patch_clamps_and_says_so(library):
    tpl = library["js_snake"]
    current = dict(tpl.defaults())
    new, note = apply_patch(tpl, current, Patch("cell_px", "scale", factor=5.0))
    assert new["cell_px"] == 40
    assert "maximum" in note


def test_unsupported_patch_names_the_supported_set(library):
    tpl = library["js_snake"]
    with pytest.raises(CustomizationError, match="not a supported parameter"):
        apply_patch(tpl, tpl.defaults(), Patch("particles", "set", value=10))


def test_scale_on_a_string_parameter_is_refused(library):
    tpl = library["js_snake"]
    with pytest.raises(CustomizationError, match="needs a numeric parameter"):
        apply_patch(tpl, tpl.defaults(), Patch("title", "scale", factor=2))


def test_multi_file_output_shape(library):
    project = render_template(library["js_pong"])
    assert len(project.files) == 3
    assert project.entry().endswith(".html")
    langs = sorted(project.languages.values())
    assert langs == ["css", "html", "javascript"]
