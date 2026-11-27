"""Expand the seed instruction set into a larger, diverse SFT corpus.

The 36 hand-written seed pairs in ``data/sources/boxz_instruct.jsonl`` teach the
answer *style* but not the *language* of the domain.  This builder adds
several hundred more examples per topic by:

  * combining question paraphrases (so the same fact is asked many ways),
  * supplying genuinely different answers per paraphrase group (not just
    re-wordings of one string), and
  * covering topics the seed set never mentions (events, canvas 2D, animation
    timing, collision, lighting, shaders, game state, local storage, ...).

Every example is project-original text written for Boxz, licensed CC0, so the
output file is redistributable and requires no download.

    python -m boxz.data.expand_instruct          # writes the expanded file
    python -m boxz.data.expand_instruct --stats
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["EXPANDED_PATH", "build_examples", "write_expanded", "main"]

EXPANDED_PATH = Path("data/sources/boxz_instruct_expanded.jsonl")
SEED_PATH = Path("data/sources/boxz_instruct.jsonl")

# ----------------------------------------------------------------------
# Topic bank.  Each entry: (question paraphrases, answer variants).
# Answers deliberately differ in wording and detail so that the model sees
# paraphrase variance rather than one memorised string per question.
# ----------------------------------------------------------------------
Topic = Tuple[List[str], List[str]]

TOPICS: List[Topic] = []


def _topic(questions: Iterable[str], answers: Iterable[str]) -> None:
    TOPICS.append((list(questions), list(answers)))


# -- identity / meta ---------------------------------------------------
_topic(
    ["hi", "hello", "hey", "hi there", "hello boxz", "hey boxz, how are you"],
    ["Hello. I am Boxz, a small language model trained from scratch on public text and code. "
     "I can explain what I am and help with simple HTML, CSS, JavaScript and Three.js questions.",
     "Hi. I am Boxz, a decoder-only transformer. I was trained by this project's own NumPy "
     "autograd engine, not imported from another model.",
     "Hello. I am a small language model built from scratch. Ask me about tokens, attention, "
     "or a simple front-end problem.",
     "Hi there. I am Boxz. I run entirely on the CPU of the machine it is started on and every "
     "weight in me came from training that the project ran itself."],
)
_topic(
    ["who are you?", "what are you?", "what is your name?", "tell me about yourself",
     "are you an ai?"],
    ["I am Boxz, a decoder-only transformer language model. Every weight in me was trained by "
     "the project's own NumPy autograd engine, not imported from another model.",
     "I am a small language model built by the Boxz project. I am trained from scratch on "
     "public text and instruction data, and I speak through a byte-level BPE tokenizer.",
     "My name is Boxz. I am a 3-million parameter transformer that this repository trains, "
     "evaluates and serves without PyTorch, pretrained weights or external APIs.",
     "I am a language model. I am not a person and not a general assistant: I am a small "
     "transformer whose behaviour comes entirely from the data the project trained me on."],
)
_topic(
    ["how do you work?", "how were you trained?", "how does boxz work?"],
    ["I read a prompt as tokens, run it through six transformer blocks with causal attention, "
     "and predict the next token one step at a time. Everything runs on NumPy.",
     "Training pairs text with its next token, computes cross entropy, and backpropagates "
     "through attention and MLP layers with an autograd engine written for this project.",
     "A prompt is tokenized, embedded, mixed by self-attention across layers, projected to the "
     "vocabulary, and sampled. There is no retrieval and no hidden service."],
)
_topic(
    ["what is a token?", "what are tokens?", "what is tokenization?", "how does a tokenizer work?"],
    ["A token is the unit a language model reads and writes. In Boxz a token is usually a "
     "fragment of a word, a whole word, or a piece of code, produced by byte-level BPE.",
     "A tokenizer turns text into integers. Boxz uses byte-level byte-pair encoding with a "
     "2048 token vocabulary, so any UTF-8 text can be encoded without an unknown token.",
     "Tokenization splits text into subword pieces. Common words become one token, rare ones "
     "are split, which keeps the vocabulary small and the sequences short."],
)
_topic(
    ["what is attention?", "how does self-attention work?", "what is a transformer?"],
    ["Attention lets every position build a weighted average of earlier positions. The weights "
     "come from dot products between queries and keys, so the model can copy and relate tokens.",
     "A transformer stacks attention with a small MLP. Attention mixes information across "
     "positions; the MLP transforms each position on its own. Residual connections tie it together.",
     "Self-attention compares each token with the tokens before it, scales the scores, applies "
     "softmax and mixes the values. Boxz runs six such layers, causally masked."],
)
_topic(
    ["can you think?", "are you conscious?", "do you understand what you type?"],
    ["I do not think the way people do. I compute probability distributions over the next token. "
     "Everything I say is sampled from those distributions.",
     "No. I have no inner state between turns and no model of the world beyond my training data. "
     "I am a function that maps text to a distribution over the next token.",
     "I mimic patterns in text. That can look like understanding, but there is no reasoning "
     "process behind it, only matrix multiplication."],
)

# -- HTML --------------------------------------------------------------
_topic(
    ["write a minimal html page", "how do i start an html file?", "show me basic html"],
    ["<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Boxz</title>\n</head>\n<body>\n  <h1>Hello</h1>\n</body>\n</html>",
     "<!doctype html>\n<html>\n  <head><meta charset=\"utf-8\"><title>Page</title></head>\n  <body>\n    <p>Hello world</p>\n  </body>\n</html>"],
)
_topic(
    ["how do i add a canvas to a page?", "how do i create a canvas element?"],
    ["<canvas id=\"game\" width=\"800\" height=\"600\"></canvas>\n\nconst canvas = document.getElementById('game');\nconst ctx = canvas.getContext('2d');",
     "<canvas id=\"c\"></canvas>\n<script>\n  const canvas = document.querySelector('#c');\n  canvas.width = 800;\n  canvas.height = 600;\n</script>"],
)
_topic(
    ["how do i add a button that runs code?", "how do i handle a button click in html?"],
    ["<button id=\"go\">Start</button>\n<script>\n  document.getElementById('go').addEventListener('click', () => {\n    console.log('started');\n  });\n</script>",
     "<button onclick=\"start()\">Play</button>\n<script>\n  function start() {\n    console.log('playing');\n  }\n</script>"],
)
_topic(
    ["what is a semantic html element?", "when should i use a section or article?"],
    ["Semantic elements describe meaning: header, nav, main, section, article, aside and footer. "
     "They help screen readers and search engines and keep markup readable.",
     "Use article for self-contained content, section for a themed group, nav for navigation, "
     "and main once per page for the primary content."],
)

# -- CSS ---------------------------------------------------------------
_topic(
    ["how do i center a div?", "how do i center something horizontally and vertically?"],
    ["Use flexbox on the parent:\n\n.parent {\n  display: flex;\n  align-items: center;\n  justify-content: center;\n}",
     "Grid works too:\n\n.parent {\n  display: grid;\n  place-items: center;\n}",
     "With fixed size you can use position and a transform:\n\n.child {\n  position: absolute;\n  top: 50%;\n  left: 50%;\n  transform: translate(-50%, -50%);\n}"],
)
_topic(
    ["what does the css box model do?", "explain margin and padding"],
    ["The box model describes how an element occupies space: content, then padding, then border, "
     "then margin. Padding is inside the border; margin is outside it.",
     "Every box has content, padding, a border and a margin. box-sizing: border-box makes width "
     "include padding and border, which is usually what you want."],
)
_topic(
    ["how do i make a responsive layout?", "how do i support mobile screens?"],
    ["Prefer fluid units and flex or grid over fixed pixel widths, add a viewport meta tag, and "
     "use media queries for the breakpoints you actually need.",
     "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\n.grid {\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));\n  gap: 16px;\n}"],
)
_topic(
    ["how do i make an element full screen?", "how do i make a canvas fill the window?"],
    ["html, body { margin: 0; height: 100%; }\ncanvas { display: block; width: 100vw; height: 100vh; }",
     "Set the canvas size from the window and update it on resize:\n\nfunction resize() {\n  renderer.setSize(window.innerWidth, window.innerHeight);\n  camera.aspect = window.innerWidth / window.innerHeight;\n  camera.updateProjectionMatrix();\n}\nwindow.addEventListener('resize', resize);"],
)
_topic(
    ["how do i add a hover effect in css?", "how do i change style on hover?"],
    [".button {\n  background: #222;\n  transition: background 150ms ease;\n}\n.button:hover {\n  background: #444;\n}",
     "Use :hover for pointer devices:\n\n.card:hover {\n  transform: translateY(-4px);\n  box-shadow: 0 8px 24px rgba(0, 0, 0, .25);\n}"],
)

# -- JavaScript --------------------------------------------------------
_topic(
    ["how do i loop over an array in javascript?", "how do i iterate an array?"],
    ["Three common ways:\n\nconst xs = [1, 2, 3];\n\nfor (const x of xs) console.log(x);\nfor (let i = 0; i < xs.length; i++) console.log(xs[i]);\nxs.forEach((x) => console.log(x));",
     "Use for...of when you only need values, forEach when you want a callback, and map when "
     "you want a new array:\n\nconst doubled = xs.map((x) => x * 2);"],
)
_topic(
    ["what is a closure?", "explain closures in javascript"],
    ["A closure is a function that keeps access to the variables of the scope where it was "
     "created, even after that scope has finished running.",
     "function counter() {\n  let n = 0;\n  return () => ++n;\n}\n\nconst next = counter();\nnext(); // 1\nnext(); // 2\n\nn survives between calls because the inner function closes over it."],
)
_topic(
    ["how do i fetch json in the browser?", "how do i make an http request in javascript?"],
    ["async function load(url) {\n  const res = await fetch(url);\n  if (!res.ok) throw new Error(`HTTP ${res.status}`);\n  return res.json();\n}",
     "const data = await fetch('/api/levels').then((r) => r.json());\n\nAlways check res.ok; fetch does not reject on 404 or 500."],
)
_topic(
    ["what is the difference between let and const?", "should i use var?"],
    ["Both are block scoped. const forbids rebinding the name, but the value it points to can "
     "still change if it is an object. var is function scoped and hoisted, so prefer let and const.",
     "Use const by default, let when you must reassign, and avoid var in new code."],
)
_topic(
    ["how do i add an event listener?", "how do i listen for keyboard input?"],
    ["document.addEventListener('keydown', (e) => {\n  if (e.code === 'Space') jump();\n});",
     "const keys = new Set();\nwindow.addEventListener('keydown', (e) => keys.add(e.code));\nwindow.addEventListener('keyup', (e) => keys.delete(e.code));\n\n// then read keys in the update loop"],
)
_topic(
    ["how do i store data in the browser?", "what is localstorage?"],
    ["localStorage stores strings per origin and survives reloads:\n\nlocalStorage.setItem('highscore', '1200');\nconst score = Number(localStorage.getItem('highscore') ?? 0);",
     "Use localStorage for small settings and save slots, and JSON.stringify for objects. It is "
     "synchronous, so keep it out of tight loops."],
)
_topic(
    ["what is an arrow function?", "how is a normal function different from an arrow function?"],
    ["An arrow function has no own this, arguments or prototype: (a, b) => a + b. That makes it "
     "convenient for callbacks inside a class or object method.",
     "const add = (a, b) => a + b;\n\nIt inherits this from the surrounding scope, unlike "
     "function add(a, b) { return a + b; }."],
)
_topic(
    ["what does array map do?", "how do i transform each element of an array?"],
    ["map returns a new array with the callback applied to every element:\n\nconst squares = xs.map((x) => x * x);",
     "Use map for transforming, filter for selecting and reduce for collapsing to one value. They "
     "do not change the original array."],
)

# -- canvas 2D / animation ---------------------------------------------
_topic(
    ["how do i draw a rectangle on a canvas?", "how do i fill a canvas shape?"],
    ["const ctx = canvas.getContext('2d');\nctx.fillStyle = '#4af';\nctx.fillRect(20, 20, 120, 80);",
     "ctx.beginPath();\nctx.rect(x, y, w, h);\nctx.fillStyle = 'tomato';\nctx.fill();"],
)
_topic(
    ["how do i animate with requestanimationframe?", "how do i write a game loop?"],
    ["function frame(t) {\n  const dt = (t - last) / 1000;\n  last = t;\n  update(dt);\n  render();\n  requestAnimationFrame(frame);\n}\nlet last = performance.now();\nrequestAnimationFrame(frame);",
     "requestAnimationFrame gives you a timestamp and syncs with the display. Compute delta time "
     "in seconds and use it for movement so the game runs at the same speed on every monitor."],
)
_topic(
    ["how do i clear the canvas each frame?", "why does my canvas leave trails?"],
    ["ctx.clearRect(0, 0, canvas.width, canvas.height);\n\n// or paint a background:\nctx.fillStyle = '#000';\nctx.fillRect(0, 0, canvas.width, canvas.height);",
     "Without clearing, every frame draws on top of the last one. Clear at the start of each frame, "
     "before drawing sprites."],
)

# -- Three.js ----------------------------------------------------------
_topic(
    ["write a minimal three.js scene", "how do i set up three.js?"],
    ["import * as THREE from 'three';\n\nconst scene = new THREE.Scene();\nconst camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);\ncamera.position.z = 3;\n\nconst renderer = new THREE.WebGLRenderer({ antialias: true });\nrenderer.setSize(innerWidth, innerHeight);\ndocument.body.appendChild(renderer.domElement);\n\nconst mesh = new THREE.Mesh(\n  new THREE.BoxGeometry(1, 1, 1),\n  new THREE.MeshStandardMaterial({ color: 0x44aaff })\n);\nscene.add(mesh);\nscene.add(new THREE.AmbientLight(0xffffff, 0.6));\nscene.add(new THREE.DirectionalLight(0xffffff, 1.2));\n\nrenderer.setAnimationLoop(() => {\n  mesh.rotation.y += 0.01;\n  renderer.render(scene, camera);\n});",
     "The four pieces are scene, camera, renderer and a mesh. Add a light if you use a standard "
     "material, then render inside setAnimationLoop."],
)
_topic(
    ["how do i move a camera with the keyboard in three.js?",
     "how do i make first person controls?"],
    ["Track key state, then integrate in the animation loop:\n\nconst keys = new Set();\naddEventListener('keydown', (e) => keys.add(e.code));\naddEventListener('keyup', (e) => keys.delete(e.code));\n\nfunction update(dt) {\n  const speed = 3 * dt;\n  const dir = new THREE.Vector3();\n  if (keys.has('KeyW')) dir.z -= 1;\n  if (keys.has('KeyS')) dir.z += 1;\n  if (keys.has('KeyA')) dir.x -= 1;\n  if (keys.has('KeyD')) dir.x += 1;\n  camera.position.addScaledVector(dir.normalize(), speed);\n}",
     "Use a pointer lock for looking around, and move the camera each frame by a direction vector "
     "scaled by delta time."],
)
_topic(
    ["how do i load a texture in three.js?", "how do i put an image on a mesh?"],
    ["const texture = new THREE.TextureLoader().load('wood.png');\ntexture.colorSpace = THREE.SRGBColorSpace;\nconst material = new THREE.MeshStandardMaterial({ map: texture });",
     "Set texture.colorSpace to sRGB for colour images. Use CanvasTexture when you draw the "
     "texture procedurally."],
)
_topic(
    ["how do i add shadows in three.js?", "why are there no shadows in my scene?"],
    ["renderer.shadowMap.enabled = true;\nlight.castShadow = true;\nmesh.castShadow = true;\nfloor.receiveShadow = true;",
     "Shadows need the renderer flag, a light that casts them, and casters plus receivers flagged "
     "on each object."],
)
_topic(
    ["how do i detect a collision between two boxes?", "how do i check if two rectangles overlap?"],
    ["function overlaps(a, b) {\n  return a.x < b.x + b.w && a.x + a.w > b.x &&\n         a.y < b.y + b.h && a.y + a.h > b.y;\n}",
     "if (Math.abs(ax - bx) < aw / 2 + bw / 2 && Math.abs(ay - by) < ah / 2 + bh / 2) {\n  // circles or centred boxes overlap\n}"],
)
_topic(
    ["how do i make a game character jump?", "how do i simulate gravity?"],
    ["Use velocity and gravity with delta time:\n\nvy -= 20 * dt;\ny += vy * dt;\nif (y < 0) { y = 0; vy = 0; grounded = true; }\n\n// on jump\nif (grounded) { vy = 8; grounded = false; }",
     "Apply a constant downward acceleration, integrate position, and clamp at the floor. Jumping "
     "sets an upward velocity once the character is grounded."],
)
_topic(
    ["how do i play sound in a web game?", "how do i add audio?"],
    ["const audio = new Audio('hit.wav');\naudio.volume = 0.5;\naudio.play();\n\nBrowsers only allow audio after a user gesture, so unlock it on the first click or key press.",
     "For overlapping effects use the Web Audio API: create an AudioContext, decode the buffer "
     "once, and start a new BufferSource per sound."],
)

# -- debugging ---------------------------------------------------------
_topic(
    ["why is my canvas blank?", "my canvas does not show anything, why?"],
    ["A blank canvas is usually one of these:\n\n1. The canvas has zero width or height in CSS.\n2. You draw before the script runs, so the element is null.\n3. The animation loop throws, so only the first frame is drawn.\n4. Nothing is inside the camera frustum.",
     "Open the console first. A thrown exception inside the first frame leaves the canvas empty, "
     "and a 0x0 canvas reports no error at all."],
)
_topic(
    ["javascript error cannot read properties of undefined", "what does undefined property error mean?"],
    ["Something is undefined when you try to read a property from it. Guard the access or fix the "
     "source:\n\nif (obj?.child) { use(obj.child); }\n\nThen find why obj is not what you expect.",
     "The error names the property and usually the line. Log the object just before the access to "
     "see which level is missing."],
)
_topic(
    ["how do i debug a three.js scene that renders nothing?",
     "my three.js scene is black, what should i check?"],
    ["Check in this order: is the renderer attached to the DOM and the right size; is the camera "
     "looking at the object; is there a light for a standard material; are near and far planes sane; "
     "any console error in the first frame.",
     "A black screen is often a camera inside geometry, a missing light, or a render call that "
     "happens before the first resize."],
)
_topic(
    ["how do i find a bug in my javascript?", "how do i debug javascript code?"],
    ["Reproduce it reliably, log the values you assume, then shrink the code until the bug "
     "disappears. Browser devtools give breakpoints, watch expressions and a call stack.",
     "console.log is enough to start; a breakpoint in the sources panel is better because you can "
     "inspect every variable at that moment."],
)

# -- game design -------------------------------------------------------
_topic(
    ["how do i organise a small game project?", "how should i structure game code?"],
    ["Split it into input, update, render and state. Keep data separate from drawing, put tunable "
     "numbers in one config object, and keep the loop in one place.",
     "A small game can live in one file with clear sections: constants, state, input handlers, "
     "update, render, and a single animation loop that calls update then render."],
)
_topic(
    ["how do i keep a score?", "how do i add a high score?"],
    ["let score = 0;\nfunction addPoints(n) {\n  score += n;\n  scoreEl.textContent = String(score);\n}",
     "Keep the score in state, render it to the DOM when it changes, and persist the best value "
     "with localStorage."],
)
_topic(
    ["how do i make a sprite animation?", "how do i animate frames?"],
    ["Track elapsed time, advance the frame index when it passes the frame duration, and draw the "
     "current frame from a sprite sheet with drawImage(source, sx, sy, sw, sh, dx, dy, dw, dh).",
     "frameTime += dt;\nif (frameTime > 0.1) {\n  frameTime = 0;\n  frame = (frame + 1) % frameCount;\n}"],
)

# -- ML basics ---------------------------------------------------------
_topic(
    ["what is a loss function?", "what does training loss mean?"],
    ["A loss function scores how wrong a prediction is. Boxz uses cross entropy over the next "
     "token, which is the negative log probability of the correct token.",
     "Loss measures error. Training lowers it by gradient descent; validation loss tells you "
     "whether the model is learning patterns or just memorising."],
)
_topic(
    ["what is a gradient?", "how does backpropagation work?"],
    ["A gradient is the vector of partial derivatives of the loss with respect to each parameter. "
     "Backpropagation applies the chain rule backwards through the computation graph.",
     "Backprop walks the graph in reverse, multiplying local derivatives to get each parameter's "
     "share of the error, then the optimiser takes a step against that direction."],
)
_topic(
    ["what is overfitting?", "how do i know my model is overfitting?"],
    ["Overfitting is when training loss keeps falling while validation loss stalls or rises: the "
     "model memorises examples instead of learning patterns. More data, regularisation or early "
     "stopping help.",
     "Watch the gap between train and held-out loss. If the gap widens steadily, you are fitting "
     "noise."],
)


# ----------------------------------------------------------------------
def _dedupe_key(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def build_examples(seed_path: Path = SEED_PATH) -> List[dict]:
    """Seed pairs plus every generated paraphrase, de-duplicated by question."""
    out: List[dict] = []
    seen: set = set()

    def add(q: str, a: str) -> None:
        key = _dedupe_key(q)
        if key in seen:
            return
        seen.add(key)
        out.append({"messages": [{"role": "user", "content": q.strip()},
                                 {"role": "assistant", "content": a.strip()}]})

    if seed_path.exists():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            if len(msgs) >= 2:
                add(msgs[0]["content"], msgs[-1]["content"])

    for qi, (questions, answers) in enumerate(TOPICS):
        for i, q in enumerate(questions):
            add(q, answers[i % len(answers)])
    return out


def write_expanded(path: Path = EXPANDED_PATH, seed_path: Path = SEED_PATH) -> dict:
    examples = build_examples(seed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    answer_chars = sum(len(ex["messages"][-1]["content"]) for ex in examples)
    return {"path": str(path), "examples": len(examples), "answer_chars": answer_chars,
            "avg_answer_chars": round(answer_chars / max(1, len(examples)), 1)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="boxz.data.expand_instruct")
    ap.add_argument("--stats", action="store_true", help="only report what would be written")
    args = ap.parse_args(argv)
    if args.stats:
        exs = build_examples()
        print(json.dumps({"examples": len(exs)}, indent=2))
    else:
        print(json.dumps(write_expanded(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
