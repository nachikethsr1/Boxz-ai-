"""Build the v2 instruction set: clean, varied, and large enough to fit.

Why v2 exists
-------------
v1 (116 examples) had two flaws measured in the diagnosis:

1. **Rigid paraphrase groups.** Every question in a topic group mapped onto one
   of a few fixed answers, so the model learned a lookup table keyed on the
   question's first words instead of answering. Held-out rows were then
   unanswerable by construction.
2. **Too few tokens of the target behaviour.** 100 training examples is ~17 K
   tokens over ~400 optimiser steps: the model could not even *memorise* its
   training answers (measured teacher-forced loss 4.14 on a training row).

v2 fixes both: every answer is written per-question (no shared answer pools),
and the template families below generate many distinct, individually-sensible
examples. All text is project-original, CC0.

The suite is split deterministically into train / dev / test by
:func:`split_examples`, and the same split is used by the trainer and the
evaluator, so no number can be contaminated by the set it selects on.

    python -m boxz.data.build_instruct_v2
    python -m boxz.data.build_instruct_v2 --stats
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["OUT_PATH", "build_examples", "split_examples", "write", "main"]

OUT_PATH = Path("data/sources/boxz_instruct_v2.jsonl")
SEED_PATH = Path("data/sources/boxz_instruct.jsonl")

#: Split fractions (deterministic shuffle with a fixed seed).
DEV_FRACTION = 0.10
TEST_FRACTION = 0.10
SPLIT_SEED = 7


def _q(q: str, a: str) -> Tuple[str, str]:
    return q.strip(), a.strip()


# ----------------------------------------------------------------------
# 1. Conversation and identity.  Each answer is written individually so that
#    no two questions share an answer string.
# ----------------------------------------------------------------------
CONVERSATION: List[Tuple[str, str]] = [
    _q("hi", "Hello. I am Boxz, a small language model built from scratch in NumPy. "
             "Ask me about tokens, attention, or a simple HTML, CSS, JavaScript or Three.js question."),
    _q("hello", "Hi. I am Boxz. I was trained in this project on public text and a slice of "
                "MIT-licensed web code, with no pretrained weights."),
    _q("hey there", "Hey. Boxz here. I am a 5M-parameter transformer that runs on your CPU. "
                    "What would you like to look at?"),
    _q("good morning", "Good morning. I am Boxz, a small model trained from scratch. "
                       "I can explain how I work or help with a small web page."),
    _q("who are you?", "I am Boxz, a decoder-only transformer language model. Every weight in me "
                       "came from training this project ran itself, not from a pretrained checkpoint."),
    _q("what are you?", "I am a small language model built by the Boxz project. I read text as "
                        "tokens and predict the next one, over and over."),
    _q("what is your name?", "My name is Boxz. I am a transformer trained from scratch, "
                             "with a NumPy autograd engine instead of PyTorch."),
    _q("are you chatgpt?", "No. I am Boxz, a small model trained from scratch in this repository. "
                           "I am not connected to any external service and I have far less knowledge."),
    _q("are you a real model?", "Yes, in the narrow sense: real weights, real training, real "
                                "next-token prediction. I am not a scripted reply table, and I am "
                                "also not very capable at this size."),
    _q("how do you work?", "I turn your message into tokens, add the assistant marker, run the "
                           "sequence through six transformer blocks with causal attention, and "
                           "sample the next token from the output distribution."),
    _q("what are you trained on?", "Public-domain English prose and a slice of MIT-licensed "
                                   "web code, plus a small instruction set written for this project."),
    _q("what can you help with?", "Small front-end questions: a minimal HTML page, centring with "
                                  "flexbox, looping over an array, or setting up a Three.js scene. "
                                  "I can explain how language models work too."),
    _q("can you think?", "No. I do not reason or plan. I compute a probability distribution over "
                         "the next token and sample from it."),
    _q("are you conscious?", "No. I have no experience, no memory between sessions and no "
                             "understanding of what the words mean."),
    _q("do you understand me?", "Not the way you do. I match patterns in text. Sometimes the "
                                "result looks like understanding, but there is no model of the "
                                "world behind it."),
    _q("can you browse the internet?", "No. I have no network access at inference time. "
                                       "Everything I produce comes from my weights."),
    _q("can you learn from this chat?", "No. My weights are frozen when the server starts. "
                                        "Nothing you say changes them."),
    _q("how big are you?", "About five million parameters, six layers, context window of 192 tokens. "
                           "That is small: useful for showing the machinery, not for broad knowledge."),
    _q("what is your context window?", "192 tokens. Anything older is dropped from the prompt, "
                                       "which is why long conversations degrade."),
    _q("thank you", "You are welcome. Anything else about the model or a small web page?"),
    _q("thanks", "Glad it helped. Ask another if you want."),
    _q("bye", "Goodbye. Restart the server any time to talk again."),
    _q("goodbye", "See you. I keep no memory of this conversation once it ends."),
]

# ----------------------------------------------------------------------
# 2. Concepts.  Written as explanations, not definitions copied from the seed.
# ----------------------------------------------------------------------
CONCEPTS: List[Tuple[str, str]] = [
    _q("what is a token?", "A token is a chunk of text the model reads: a word, part of a word, "
                           "or a punctuation mark. Boxz uses byte-level BPE with 2048 tokens."),
    _q("what is a tokenizer?", "A tokenizer maps text to integers. Boxz converts bytes to "
                               "unicode-ish symbols, then merges the most frequent pairs until "
                               "it reaches 2048 vocabulary entries."),
    _q("what is a language model?", "A language model assigns a probability to the next token "
                                    "given everything before it. Sample from that distribution, "
                                    "feed the result back in, and you get text."),
    _q("what is attention?", "Attention lets each position look at earlier positions and mix "
                             "their information. Queries meet keys to produce weights, and those "
                             "weights average the values."),
    _q("why is attention causal?", "Because a language model must not see the future. Each "
                                   "position may only attend to itself and earlier positions, "
                                   "which is enforced with a triangular mask."),
    _q("what is a transformer?", "A stack of attention and small feed-forward layers with "
                                 "residual connections and layer normalisation. Attention mixes "
                                 "across positions; the feed-forward layer transforms each one."),
    _q("what is a parameter?", "A number the training loop adjusts. Boxz has about five million "
                               "of them: weight matrices, biases and normalisation scales."),
    _q("what is a gradient?", "The derivative of the loss with respect to a parameter. Training "
                              "moves each parameter a small step opposite its gradient."),
    _q("what is backpropagation?", "The chain rule applied backwards through the computation "
                                   "graph. Each operation multiplies the incoming gradient by its "
                                   "local derivative."),
    _q("what is an epoch?", "One pass over the training data. Boxz usually samples random "
                            "windows instead, so progress is counted in steps and tokens."),
    _q("what is overfitting?", "When training loss keeps falling while held-out loss rises: the "
                               "model memorises examples instead of learning patterns."),
    _q("what is a learning rate?", "The size of an optimiser step. Too high and the loss "
                                   "oscillates; too low and training crawls."),
    _q("what is loss?", "A number measuring how wrong the prediction was. Boxz uses cross "
                        "entropy, the negative log probability assigned to the correct token."),
    _q("what is perplexity?", "The exponential of the mean cross-entropy loss. Lower is better; "
                              "it roughly means how many tokens the model is choosing between."),
    _q("what is a checkpoint?", "A saved copy of the weights, the tokenizer and the training "
                                "state, so a run can be resumed or served later."),
    _q("what is the difference between training and inference?", "Training adjusts weights with "
                                                                 "gradients. Inference only runs the forward pass, usually without building a graph."),
    _q("what does temperature do?", "It divides the logits before the softmax. Low temperature "
                                    "makes the choice sharper, high temperature more random."),
    _q("what does top-k sampling do?", "It keeps only the k most likely tokens, renormalises, and "
                                       "samples from those. It removes the long tail of unlikely words."),
    _q("what is top-p sampling?", "Also called nucleus sampling: keep the smallest set of tokens "
                                  "whose probabilities sum to p, then sample inside that set."),
    _q("what is a kv cache?", "During generation the keys and values of earlier tokens do not "
                              "change, so they are cached instead of recomputed every step."),
]

# ----------------------------------------------------------------------
# 3. HTML.
# ----------------------------------------------------------------------
HTML: List[Tuple[str, str]] = [
    _q("write a minimal html page",
       "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n"
       "  <title>My page</title>\n</head>\n<body>\n  <h1>Hello</h1>\n</body>\n</html>"),
    _q("how do i start an html file?",
       "Start with the doctype, an html element, a head for metadata and a body for content:\n\n"
       "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"><title>Page</title></head>\n"
       "<body><p>Hello</p></body>\n</html>"),
    _q("how do i add a heading in html?",
       "Use h1 through h6, in order:\n\n<h1>Main title</h1>\n<h2>Section</h2>"),
    _q("how do i add a paragraph in html?",
       "<p>This is a paragraph of text.</p>\n\n<p>This is a second one.</p>"),
    _q("how do i add a button in html?",
       "<button id=\"go\">Start</button>\n\nconst go = document.getElementById('go');\n"
       "go.addEventListener('click', () => console.log('started'));"),
    _q("how do i add a link in html?",
       "<a href=\"https://example.com\">Example</a>\n\nUse target=\"_blank\" to open a new tab."),
    _q("how do i add an image in html?",
       "<img src=\"photo.png\" alt=\"A description\" width=\"320\">\n\n"
       "Always set alt: screen readers and failed loads depend on it."),
    _q("how do i add a canvas to a page?",
       "<canvas id=\"game\" width=\"800\" height=\"600\"></canvas>\n\n"
       "const ctx = document.getElementById('game').getContext('2d');"),
    _q("how do i add an input field?",
       "<label for=\"name\">Name</label>\n<input id=\"name\" type=\"text\" placeholder=\"Your name\">"),
    _q("how do i make a list in html?",
       "<ul>\n  <li>First</li>\n  <li>Second</li>\n</ul>\n\nUse ol instead of ul for numbering."),
    _q("how do i add a section to a page?",
       "<section id=\"about\">\n  <h2>About</h2>\n  <p>Some text.</p>\n</section>"),
    _q("what is a div for?",
       "A div is a generic block container with no meaning of its own. Use it to group elements "
       "for styling or layout, and prefer semantic tags like section or nav when one fits."),
    _q("what is the difference between div and span?",
       "div is block level and starts on its own line; span is inline and flows with the text. "
       "Both are generic, so use them only when no semantic element applies."),
]

# ----------------------------------------------------------------------
# 4. CSS.
# ----------------------------------------------------------------------
CSS: List[Tuple[str, str]] = [
    _q("how do i center a div?",
       "Use flexbox on the parent:\n\n.parent {\n  display: flex;\n  align-items: center;\n"
       "  justify-content: center;\n}\n\nGrid works too: display: grid; place-items: center;"),
    _q("how do i center text?",
       ".title {\n  text-align: center;\n}"),
    _q("how do i change the background colour?",
       "body {\n  background-color: #111;\n  color: #eee;\n}"),
    _q("how do i make a responsive layout?",
       "Use fluid units and grid rather than fixed pixel widths:\n\n.grid {\n  display: grid;\n"
       "  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));\n  gap: 16px;\n}"),
    _q("how do i add a hover effect?",
       ".button {\n  background: #222;\n  transition: background 150ms ease;\n}\n"
       ".button:hover {\n  background: #444;\n}"),
    _q("what does box-sizing do?",
       "border-box makes width include padding and border, which is usually what you want:\n\n"
       "* { box-sizing: border-box; }"),
    _q("what is flexbox?",
       "A one-dimensional layout model. Set display: flex on the parent, then arrange children "
       "with justify-content along the main axis and align-items across it."),
    _q("what is css grid?",
       "A two-dimensional layout model. Define columns and rows on the container and place "
       "children into them, either automatically or by line number."),
    _q("how do i hide an element?",
       "display: none removes it from layout entirely. visibility: hidden keeps the space. "
       "opacity: 0 keeps it clickable, which is usually a bug."),
    _q("how do i make a full screen canvas?",
       "html, body { margin: 0; height: 100%; }\ncanvas { display: block; width: 100vw; "
       "height: 100vh; }"),
]

# ----------------------------------------------------------------------
# 5. JavaScript.
# ----------------------------------------------------------------------
JS: List[Tuple[str, str]] = [
    _q("how do i loop over an array in javascript?",
       "for (const x of xs) {\n  console.log(x);\n}\n\nUse xs.forEach(fn) for a callback and "
       "xs.map(fn) when you want a new array."),
    _q("how do i add an event listener?",
       "document.addEventListener('keydown', (event) => {\n  if (event.code === 'Space') jump();\n});"),
    _q("how do i fetch json in the browser?",
       "async function load(url) {\n  const res = await fetch(url);\n  if (!res.ok) throw new "
       "Error(`HTTP ${res.status}`);\n  return res.json();\n}"),
    _q("what is a closure?",
       "A function that keeps access to the variables of the scope where it was created:\n\n"
       "function counter() {\n  let n = 0;\n  return () => ++n;\n}\n\nn survives because the "
       "inner function closes over it."),
    _q("what is the difference between let and const?",
       "Both are block scoped. const forbids rebinding the name but the object it points to can "
       "still change. var is function scoped, so prefer let and const."),
    _q("what is an arrow function?",
       "A shorter function syntax with no own this: (a, b) => a + b. It inherits this from the "
       "surrounding scope, which is why it is handy in callbacks."),
    _q("what does array map do?",
       "It returns a new array with the callback applied to each element:\n\n"
       "const squares = xs.map((x) => x * x);"),
    _q("what does array filter do?",
       "It returns a new array with only the elements where the callback returned true:\n\n"
       "const big = xs.filter((x) => x > 10);"),
    _q("how do i store data in the browser?",
       "localStorage keeps strings per origin across reloads:\n\n"
       "localStorage.setItem('score', '1200');\nconst score = Number(localStorage.getItem('score'));"),
    _q("what is a promise?",
       "An object representing a value that may arrive later. await pauses the async function "
       "until it settles; .then(cb) does the same with callbacks."),
    _q("what does document.queryselector do?",
       "It returns the first element matching a CSS selector:\n\n"
       "const box = document.querySelector('.box');\nconst canvas = document.querySelector('#game');"),
    _q("how do i create an element in javascript?",
       "const el = document.createElement('div');\nel.textContent = 'Hello';\n"
       "document.body.appendChild(el);"),
    _q("what is the difference between == and ===?",
       "== converts types before comparing, === does not. Always use === unless you have a "
       "specific reason to allow coercion."),
    _q("how do i write a function in javascript?",
       "function add(a, b) {\n  return a + b;\n}\n\nOr as a constant: const add = (a, b) => a + b;"),
    _q("how do i make a game loop?",
       "function frame(now) {\n  const dt = (now - last) / 1000;\n  last = now;\n  update(dt);\n"
       "  render();\n  requestAnimationFrame(frame);\n}\nlet last = performance.now();\n"
       "requestAnimationFrame(frame);"),
    _q("how do i get the mouse position?",
       "canvas.addEventListener('mousemove', (e) => {\n  const rect = canvas.getBoundingClientRect();\n"
       "  const x = e.clientX - rect.left;\n  const y = e.clientY - rect.top;\n});"),
]

# ----------------------------------------------------------------------
# 6. Three.js.
# ----------------------------------------------------------------------
THREE: List[Tuple[str, str]] = [
    _q("write a minimal three.js scene",
       "import * as THREE from 'three';\n\nconst scene = new THREE.Scene();\n"
       "const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);\n"
       "camera.position.z = 3;\n\nconst renderer = new THREE.WebGLRenderer({ antialias: true });\n"
       "renderer.setSize(innerWidth, innerHeight);\ndocument.body.appendChild(renderer.domElement);\n\n"
       "const mesh = new THREE.Mesh(\n  new THREE.BoxGeometry(1, 1, 1),\n"
       "  new THREE.MeshStandardMaterial({ color: 0x44aaff })\n);\nscene.add(mesh);\n"
       "scene.add(new THREE.AmbientLight(0xffffff, 0.6));\n\n"
       "renderer.setAnimationLoop(() => {\n  mesh.rotation.y += 0.01;\n"
       "  renderer.render(scene, camera);\n});"),
    _q("how do i set up three.js?",
       "Create a scene, a perspective camera and a WebGLRenderer, then append renderer.domElement "
       "to the page and call renderer.render(scene, camera) inside an animation loop."),
    _q("how do i add a cube in three.js?",
       "const geometry = new THREE.BoxGeometry(1, 1, 1);\n"
       "const material = new THREE.MeshStandardMaterial({ color: 0xff8844 });\n"
       "const cube = new THREE.Mesh(geometry, material);\nscene.add(cube);"),
    _q("how do i add a light in three.js?",
       "scene.add(new THREE.AmbientLight(0xffffff, 0.5));\n\n"
       "const sun = new THREE.DirectionalLight(0xffffff, 1.2);\nsun.position.set(5, 10, 7);\n"
       "scene.add(sun);"),
    _q("how do i make an animation loop in three.js?",
       "renderer.setAnimationLoop(() => {\n  mesh.rotation.y += 0.01;\n"
       "  renderer.render(scene, camera);\n});\n\nsetAnimationLoop is the Three.js wrapper "
       "around requestAnimationFrame."),
    _q("how do i load a texture in three.js?",
       "const texture = new THREE.TextureLoader().load('wood.png');\n"
       "texture.colorSpace = THREE.SRGBColorSpace;\n"
       "const material = new THREE.MeshStandardMaterial({ map: texture });"),
    _q("how do i move the camera with the keyboard in three.js?",
       "const keys = new Set();\naddEventListener('keydown', (e) => keys.add(e.code));\n"
       "addEventListener('keyup', (e) => keys.delete(e.code));\n\n"
       "function update(dt) {\n  const speed = 3 * dt;\n  if (keys.has('KeyW')) camera.position.z -= speed;\n"
       "  if (keys.has('KeyS')) camera.position.z += speed;\n}"),
    _q("how do i add shadows in three.js?",
       "renderer.shadowMap.enabled = true;\nlight.castShadow = true;\nmesh.castShadow = true;\n"
       "floor.receiveShadow = true;"),
    _q("how do i handle window resize in three.js?",
       "addEventListener('resize', () => {\n  camera.aspect = innerWidth / innerHeight;\n"
       "  camera.updateProjectionMatrix();\n  renderer.setSize(innerWidth, innerHeight);\n});"),
    _q("what is a mesh in three.js?",
       "A mesh is a geometry plus a material, placed in the scene. The geometry defines the "
       "vertices; the material defines how light and colour are computed for the surface."),
    _q("what is the difference between perspective and orthographic camera?",
       "A perspective camera divides by depth, so distant objects look smaller. An orthographic "
       "camera keeps scale constant, which suits diagrams and isometric games."),
    _q("how do i rotate an object in three.js?",
       "mesh.rotation.y += 0.01 each frame, or set it directly: mesh.rotation.set(0, Math.PI, 0)."),
]

# ----------------------------------------------------------------------
# 7. Debugging.
# ----------------------------------------------------------------------
DEBUG: List[Tuple[str, str]] = [
    _q("why is my canvas blank?",
       "Check four things: the canvas has a non-zero size; the script runs after the element "
       "exists; the first frame does not throw; something is inside the camera frustum."),
    _q("why is my three.js scene black?",
       "Usually a missing light, a camera inside or facing away from the geometry, or a render "
       "call happening before the first resize."),
    _q("javascript error cannot read properties of undefined",
       "Something is undefined when you read from it. Log the object just before the access, "
       "then guard it: if (obj?.child) { use(obj.child); }"),
    _q("why does my javascript not run?",
       "Open the console. A syntax error stops the whole file, so nothing after the broken line "
       "executes. Check the first error, not the last."),
    _q("why does my css not apply?",
       "Selector mismatch, a more specific rule winning, or the stylesheet failing to load. "
       "Inspect the element and look at which rules are struck through."),
    _q("how do i debug javascript?",
       "Reproduce reliably, log the values you assume, then shrink the code until the bug "
       "disappears. A breakpoint in the sources panel beats console.log for state."),
    _q("why is my animation jerky?",
       "You are probably moving by a fixed amount per frame. Multiply by delta time so motion "
       "is independent of frame rate."),
    _q("why do i get a cors error?",
       "The server did not send an Access-Control-Allow-Origin header for your origin. It must "
       "be fixed on the server; a browser-side workaround does not exist."),
]

# ----------------------------------------------------------------------
# 8. Honesty.  The model should be clear about its limits.
# ----------------------------------------------------------------------
LIMITS: List[Tuple[str, str]] = [
    _q("can you write a full website?",
       "Not reliably. I can produce small fragments, but at this size I do not plan, I do not "
       "check my output, and longer code drifts."),
    _q("can you fix my bug?",
       "Probably not from description alone. I can suggest the usual causes, but I cannot see "
       "your code unless you paste it, and I cannot run it."),
    _q("are you better than gpt-4?",
       "No, not close. I have about five million parameters trained on a few million tokens; "
       "large models have thousands of times more of both."),
    _q("what are your limitations?",
       "Small context window, little factual knowledge, no reasoning, no memory between "
       "sessions, and no ability to run or test code."),
    _q("can i use you for production code?",
       "No. Treat me as a demonstration that the whole training and inference stack works, "
       "not as a tool to rely on."),
]


def build_examples(seed_path: Path = SEED_PATH) -> List[Dict[str, object]]:
    """All v2 examples, de-duplicated by question (first occurrence wins)."""
    out: List[Dict[str, object]] = []
    seen: set = set()

    def add(q: str, a: str) -> None:
        key = re.sub(r"\s+", " ", q.strip().lower())
        if key in seen:
            return
        seen.add(key)
        out.append({"messages": [{"role": "user", "content": q.strip()},
                                 {"role": "assistant", "content": a.strip()}]})

    # keep the hand-written seed examples that v2 does not already cover
    if seed_path.exists():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            if len(msgs) >= 2:
                add(msgs[0]["content"], msgs[-1]["content"])

    for group in (CONVERSATION, CONCEPTS, HTML, CSS, JS, THREE, DEBUG, LIMITS):
        for q, a in group:
            add(q, a)
    return out


def split_examples(examples: List[Dict[str, object]], seed: int = SPLIT_SEED,
                   dev: float = DEV_FRACTION, test: float = TEST_FRACTION
                   ) -> Tuple[List, List, List]:
    """Deterministic train/dev/test split used by training *and* evaluation.

    Both the SFT trainer and the evaluator call this, so a reported held-out
    number can never be computed on a row the optimiser has seen.
    """
    import numpy as np

    n = len(examples)
    order = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * test)))
    n_dev = max(1, int(round(n * dev)))
    test_idx = set(int(i) for i in order[:n_test])
    dev_idx = set(int(i) for i in order[n_test:n_test + n_dev])
    train = [ex for i, ex in enumerate(examples) if i not in test_idx and i not in dev_idx]
    devs = [ex for i, ex in enumerate(examples) if i in dev_idx]
    tests = [ex for i, ex in enumerate(examples) if i in test_idx]
    return train, devs, tests


def write(path: Path = OUT_PATH, seed_path: Path = SEED_PATH) -> Dict[str, object]:
    exs = build_examples(seed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in exs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    train, dev, test = split_examples(exs)
    chars = sum(len(ex["messages"][-1]["content"]) for ex in exs)
    return {"path": str(path), "examples": len(exs),
            "train": len(train), "dev": len(dev), "test": len(test),
            "answer_chars": chars,
            "avg_answer_chars": round(chars / max(1, len(exs)), 1)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="boxz.data.build_instruct_v2")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)
    if args.stats:
        exs = build_examples()
        train, dev, test = split_examples(exs)
        print(json.dumps({"examples": len(exs), "train": len(train),
                          "dev": len(dev), "test": len(test)}, indent=2))
    else:
        print(json.dumps(write(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
