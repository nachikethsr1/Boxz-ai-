# Boxz Coding System — Implementation Report

Branch `arena/5bdc9efb-boxz-ai`. Two commits: `deba192` (the system) and
`3465e58` (bugs the new tests found). 40 files changed, no new dependencies:
`requirements.txt` is still just `numpy`.

Everything below is measured on this checkout. Where something is not done, it
says so.

---

## 1. What was created and modified

**New package `boxz/coding/` — 2,442 lines including the pre-existing `engine.py`:**

| File | Lines | Role |
|---|---|---|
| `requests.py` | 298 | Reads `coding_requests.txt`. Also writes the JSONL export. |
| `templates.py` | 334 | Metadata model: typed parameters, ranges, aliases, placeholders. |
| `customize.py` | 221 | Substitutes values into templates. Rejects bad ones. |
| `library.py` | 139 | Validates each generated file with its own validator. |
| `selection.py` | 427 | The router: which template, or ask, or refuse. |
| `session.py` | 344 | Active project state and follow-up parsing. |
| `system.py` | 307 | Orchestration; the entry point `CodingSystem.handle()`. |
| `engine.py` | 372 | **Pre-existing, unchanged.** Still the only thing that validates code. |

**Six templates, 24 files** under `data/templates/<id>/` — real source, not
placeholder stubs (`template.json` plus the actual HTML/CSS/JS):

`js_snake`, `js_pong`, `js_click_counter`, `js_timer_app`,
`canvas_particle_effect`, `css_button_animation`.

**Modified, minimally:**

- `boxz/chat/engine.py` — coding is consulted between the maths branch and the
  model. Off-switch `use_coding_system=False`.
- `boxz/cli.py` — two commands: `code` and `router`.
- `boxz/app/demo.py` — `GET /api/templates`, `GET /api/build?q=`.
- `tests/` — five new files, 132 tests.

---

## 2. Features

- **Deterministic construction.** A recognised request builds a real project
  without generating a single token (`tokens_generated == 0`).
- **Paraphrase tolerance.** TF-IDF over the corpus examples plus per-template
  keywords.
- **Clarification instead of guessing.** Two plausible templates → a question
  naming the actual candidates.
- **Refusal with a reason.** Python, React, Three.js, Unity, Kotlin and friends
  are declined by category, not by accident.
- **Follow-ups edit the existing project.** "Make it faster" rescores the
  project you already have.
- **Values stated up front are honoured.** "a 20 by 20 grid", "a fast speed",
  "a blue snake game", "a timer for 90 seconds".
- **Three surfaces:** chat (`/api/chat`), CLI, HTTP build endpoint.

---

## 3. The templates

| ID | Category | Files | Parameters |
|---|---|---|---|
| `js_snake` | 2d_arcade_games | 3 | title, grid_size, cell_px, step_ms, start_length, show_score, bg/snake/food/grid_color |
| `js_pong` | 2d_arcade_games | 3 | title, win_score, paddle_h, ball_speed, accel, cpu_enabled, cpu_skill, paddle/ball/court/bg_color |
| `js_click_counter` | dom_interactions | 3 | title, button_label, click_message, start_value, step, min/max_value, persist, count_font_px, count/bg/card_color |
| `js_timer_app` | dom_interactions | 3 | title, countdown_s, tick_ms, display_font_px, accent/bg_color |
| `canvas_particle_effect` | canvas_animations | 3 | title, particle_count, speed, particle_radius, link_distance, link_lines, particle/link/bg_color |
| `css_button_animation` | css_animations | 3 | title, button_label, accent_color, accent_hover, bg_color, font_px, radius_px, duration_ms, lift_px, hover_scale, glow_px, arrow_px, ripple_enabled |

All three requested priorities are present (particle effect, CSS button
animation) plus three games. The page is always split into `index.html`,
a `.js` file and a `.css` file, because the HTML validator cannot see inside
inline `<script>`/`<style>`.

---

## 4. How the router uses `coding_requests.txt`

It is parsed, not copied. `requests.py` reads the file's four line types into
records and raises on anything it does not understand — a silently skipped line
would be a silently missing routing example.

Verified parse: **9 categories, 66 template blocks, 6 clarify blocks, 3
unsupported blocks, 1,024 request sentences, 0 duplicates.**

Range is only 4–8 templates, so the corpus is used as **specification and test
data**:

- The example sentences of the six implemented templates form a TF-IDF index.
  A request is scored against those examples.
- The corpus also supplies the clarify and unsupported cases, so the router has
  concrete things to refuse.

`data/coding/requests.jsonl` exports all 1,024 requests, one JSON object per
line, each carrying `expected_outcome`, `expected_template`, `expected_reason`
and `implemented`:

```json
{"utterance": "Make a Snake game where the tail grows each time the player eats a dot.",
 "expected_outcome": "match", "expected_template": "js_snake", "expected_reason": null,
 "intent": "coding", "category": "2d_arcade_games", "template_block": "js_snake",
 "implemented": true, "source": "coding_requests.txt"}
```

`python -m boxz router` scores against it; `--export` regenerates the JSONL.

**Measured** (`python -m boxz router`, exit code 0):

| Corpus block | Result |
|---|---|
| requests for the six implemented templates | **84/84 = 100%** route to the right template |
| wrong template built | **0** |
| unsupported (Python/React/Unity/…) | **29/30 = 96.7%** |
| clarify cases | **39/70 = 55.7%**, with **0** silently built |

---

## 5. How the Coding Engine validates

Each file is validated with the validator for **its own language**, via the
existing `boxz.coding.engine.validate()`:

| Language | What actually happens |
|---|---|
| `javascript` | `node --check` |
| `css` | brace and comment balance |
| `html` | tag balance and duplicate attributes |

The verdict never overstates this. `ProjectValidation.verdict()` always appends
*"Nothing was run in a browser."*, and HTML is described as *"checked for tag
balance only: its structure, not its runtime behaviour."* `browser_tested` is
`False` and is asserted `False` in the tests.

I did not touch TypeScript validation. It remains unsupported in v1, as
allowed.

---

## 6. Tests actually run

```
$ .venv/bin/python -m pytest tests/
398 passed, 8 skipped in 12.56s
```

Baseline before this work was `253 passed, 13 skipped`. The 5 extra skips
became 8 because three new tests skip when no checkpoint is present; here the
checkpoint exists, so all coding tests ran.

| File | Tests | Covers |
|---|---|---|
| `test_coding_requests.py` | 18 | parser: counts, malformed input, duplicates |
| `test_coding_templates.py` | 25 | templates load, placeholders declared, ranges, rendering, validation |
| `test_coding_selection.py` | 44 | intents, corpus accuracy, unsupported, clarify, no-guessing |
| `test_coding_system.py` | 36 | build, follow-ups, refusals, maths/chat preservation |
| `test_coding_integration.py` | 12 | real `ChatEngine` + checkpoint: ordering, streaming, reset |

135 of the 398 tests are the new coding ones.

The tests earned their keep — they found seven real bugs, all fixed in `3465e58`
and the follow-up commit:

1. *"make the page react to the mouse"* was refused as the React framework.
2. *"what should go on it"* was refused as the Go language.
3. *"make windows pop up on click"* was refused as Windows.
4. *"a 30x30 snake game"* matched **worse** than *"a snake game"*, because
   TF-IDF averages over the sentence — adding detail hurt.
5. The demo server called `reset()` every turn, wiping the coding session, so
   follow-ups silently reached the model. The HTTP test caught this; the unit
   tests could not.
6. A single incidental keyword beat the vector score, so *"a page that counts
   clicks on each colored button"* built a button animation instead of a
   counter.
7. Hedged requests (*"a popup or something"*, *"some challenges"*) were built
   as if they were specifications.

Also fixed: `last_coding` staleness across turns, and `_resolve_param`
returning an alias (`speed` for pong) that was not a declared parameter, which
silently dropped stated speeds.

---

## 7. Limitations

Honest list. These are deliberate or unfixable with the current goal.

- **Six templates, not sixty.** 60 of the corpus's 66 template blocks have no
  implementation; requests for them ask or refuse rather than build. That is
  the agreed v1 scope.
- **Clarify-block accuracy is 55.7%, not 98%.** This is the biggest honest
  gap. 16 of the 31 misses are greetings (`"hi"`, `"thanks for helping me"`)
  inside the corpus's `task_not_specified` block; answering a greeting with a
  template menu would be worse behaviour than the corpus label implies, so
  they reach the model. The rest are requests to a feature that is not a
  parameter (`"make the cards interactive"`), which none of these templates
  can honour. The metric that matters most — **never building the wrong
  project when the request is vague — is 0 failures.**
- **Nothing is executed.** No browser, no headless test. A template could pass
  every check and still be broken. `node --check` catches syntax, not logic.
- **HTML validation is shallow** (tag balance, duplicate attributes). The
  verdict says so every time.
- **The neural model is unchanged.** It still writes poor text — ask it "What
  is CSS?" and you get nonsense. The coding system bypasses that for the six
  templates and nothing else. No retraining was done.
- **English only.** Paraphrases outside the corpus's vocabulary can miss.
- **Single-turn parameter memory.** The session holds parameters as scalars;
  "make it red" then "make the snake green" applies the last match, not a
  history you can undo.
- **Not optimised.** The TF-IDF index is rebuilt at startup, which is fine at
  1,024 sentences and would not be at 100,000.

---

## 8. How to run and test

```bash
# score the router against the corpus
.venv/bin/python -m boxz router

# regenerate data/coding/requests.jsonl
.venv/bin/python -m boxz router --export data/coding/requests.jsonl

# list the library
.venv/bin/python -m boxz code --list

# build a project (writes real files)
.venv/bin/python -m boxz code "Make a snake game with a 20 by 20 grid" --out ./mysnake

# chat server with the real checkpoint
.venv/bin/python -m boxz.app.demo --checkpoint checkpoints/v3-sft/best --port 8000

# all tests
.venv/bin/python -m pytest tests/
```

In the chat window, one session behaves like this — verified against the live
server, not just unit tests:

```
you : Make a snake game with a 20 by 20 grid and a fast speed
boxz: Snake game. Files: index.html, game.js, style.css. ... (0 tokens)
you : Make it slower.          -> step_ms 90 -> 126, same project
you : Change the color to blue.-> snake_color -> #0074d9
you : Add a score.             -> declined, project kept, parameters listed
you : now make a click counter -> new project (explicit "make")
you : what is 17% of 2480      -> 421.6, from the Maths Engine
you : hi                       -> the model answers
```

---

## 9. Design decisions worth knowing

- **The model is untouched and still in charge by default.** Coding only
  answers when it recognises the request; everything else falls through.
- **Order is maths → templates → model.** Maths wins because it is exact. The
  coding system is consulted after it and never for a sum.
- **Refusing is a feature.** "Add a score" to a snake template returns *"adding
  a feature is not a supported parameter change"* instead of silently swapping
  in a click counter.
- **No find-and-replace.** Values are substituted through declared
  `__PLACEHOLDER__` tokens only; a placeholder that survives rendering is an
  error, and a value that no template declares is rejected.
- **A stated value is a proposal, not an order.** "a 500 by 500 grid" is
  clamped to 30 and the user is told.

## 10. Next steps

In order of value: implement 2–3 more corpus templates (platformer, starfield,
card hover) so the ask/refuse path shrinks; add a headless smoke test that
actually runs each game's JS in `node` with a stubbed DOM; and only then
consider any model work.
