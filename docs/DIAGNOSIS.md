# Boxz diagnosis and repair report

**Scope.** The complaint was that Boxz answers, but the answers are incoherent and
its code generation is broken. This report covers the twelve requested checks, what
was actually wrong, what changed, and what Boxz can honestly do now.

**Bottom line up front.** There was **no wiring bug**. The tokenizer, the special
tokens, the chat template, the prompt path and the decode path were all correct and
all are covered by tests. Three genuine software defects existed, and all three were
in the *measurement and configuration* layer rather than the model maths — one of
them inflated every reported SFT number, one made the trainer select on a different
split than the evaluator, and one made the served decoder corrupt code it had
correctly generated. Fixing them produced a real 33.8% improvement in held-out
answer-token loss. **It did not produce a model that writes correct code.** The
honest conclusion, stated in advance and confirmed by the data, is that at 3–5 M
parameters the network has enough capacity to learn fluent English openings but not
enough to compose a syntactically valid program. That is a capability ceiling, not a
bug, and pretending otherwise would be the one thing this report must not do.

---

## 1. What was tested, and how

### 1.1 The checkpoint the app really serves (requirement 1)

Queried via `/api/status` on the running server, not by reading a config file:

| field | value |
|---|---|
| checkpoint | `checkpoints/sft/best` |
| step | 200 |
| parameters | 3,087,360 |
| context length | 128 |
| backend | NumPy 2.4.6 |

### 1.2 Tokenizer, special tokens, template, config (requirement 2)

All checks pass. Model `vocab_size` 2048 equals the tokenizer's vocabulary. All
eight special tokens round-trip:

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| token | `<pad>` | `<unk>` | `<bos>` | `<eos>` | `<\|user\|>` | `<\|assistant\|>` | `<\|system\|>` | `<\|end\|>` |

The prefix the model is *trained* on is byte-identical to the prefix it is
*conditioned* on at inference time for 40 of 40 audited examples. If this had
failed it would have explained everything; it did not fail.

### 1.3 Does the model receive the right prompt, and does the reply decode correctly? (requirement 3)

`render_prompt("who are you?")` produces

```
<|user|>who are you?<|assistant|>
```

which tokenizes to `[4, 1438, 119, 609, 333, 71, 5]`, i.e.

```
['<|user|>', 'wh', 'o', ' are', ' you', '?', '<|assistant|>']
```

and the decode path strips specials correctly. Correct prompt, correct ids, correct
text. **No bug here.**

### 1.4 Sampling settings (requirement 4)

This is where the first real bug was, and it was in the *served defaults* rather
than in the sampler code. Measured over the fixed suite with everything else held
constant:

| decoding | requirement met | token hit rate |
|---|---|---|
| temp 0.2, top-k 15, rep 1.15 (what shipped) | 0.0 % | 12.2 % |
| greedy, rep 1.0 | **15.0 %** | 9.8 % |
| temp 0.4, top-k 40, rep 1.0 | **15.0 %** | **14.6 %** |

A repetition penalty of 1.15 rewrites the logits of every recently seen token. In
prose that is fine. In code, `.`, `,`, `const`, `renderer`, `camera` and `x`
legitimately repeat, so the penalty was actively destroying the most predictable
part of the output. **Removing the penalty was worth more than the entire model
upgrade** (see §3). Defaults are now temp 0.4 / penalty 1.0 in both the server API
and the UI sliders.

### 1.5 Leaked training fragments (requirement 5)

Real, and substantial. Measured on `data/raw/code_corpus` (473 files,
2,717,338 chars):

| contamination | chars | share |
|---|---|---|
| `<meta>` / `<title>` / `og:` boilerplate | 197,183 | 7.3 % |
| lines repeated in ≥ 50 files | 316,043 | 11.6 % |

Most-repeated individual lines:

| count | line |
|---|---|
| 659 | `renderer.setSize( window.innerWidth, window.innerHeight );` |
| 393 | `<meta property="og:type" content="website">` |
| 382 | viewport meta |
| 367 | `camera.updateProjectionMatrix();` |
| 365 | `scene = new THREE.Scene();` |

That table explains the failure signature better than any theory: the reported
outputs (`og:title` headers, `renderer, renderer, renderer`, `map: map: map:`) are
the highest-frequency lines in the corpus, and a small model reproduces its
highest-frequency patterns. The prompts were never the problem. This was tested
directly by feeding the *base* model perfect corpus-style prefixes such as
`const scene = new THREE.Scene();` — it still produced `map: map: map:`, so it is
not a prompt-mismatch artefact.

### 1.6 Fixed evaluation set (requirement 6)

`boxz/eval/fixed_eval.py` scores 20 fixed prompts across five kinds (5 chat,
4 explain, 3 HTML, 4 JS, 4 Three.js). Each prompt declares required substrings,
forbidden substrings, and a structural wellformedness check for HTML/JS/Three.js.
Scoring is automatic, so it can be re-run by anyone and cannot be talked up:

```bash
python -m boxz.eval.fixed_eval --checkpoint checkpoints/sft/best --text
```

### 1.7 Where the problem actually is (requirement 7)

**Not inference, not the tokenizer.** The decisive measurement was teacher-forced
loss on a row the model was *trained on*. A model that has learned a row can
reproduce it when the correct prefix is supplied. It scored **4.1419** — it had not
memorised a single one of its own 116 training answers after 200 SFT steps. The
failure was therefore upstream of generation: insufficient capacity, insufficient
clean data, and a training loop that could not see it was failing because its
reported loss was wrong.

---

## 2. What was broken, and what was fixed (requirement 8)

Three genuine defects. All three are now covered by tests that fail if they return.

### 2.1 The reported SFT loss was not the loss being optimised — **fixed**

`evaluate()` passed the assistant-token mask into `CrossEntropy`, whose
`ignore_index` is `-100`, but the mask uses token id `0`. Every padded position was
therefore treated as a real target with weight zero, which **deflated** the mean by
the padding fraction: the number on the tin corresponded to no objective at all.
On top of that, the training log printed the *replay-weighted* composite loss, which
is dominated by pretraining batches. That is how a run printed `train loss 0.5091`
while its real answer-token NLL was above 4.5, and why every reported SFT number in
the previous milestone was optimistic.

`evaluate()` now computes plain log-softmax NLL over assistant positions only, on
the dev split the optimiser never touches.

### 2.2 Trainer and evaluator were selecting on different splits — **fixed**

`--dataset` defaulted to the old v1 file, so the trainer took its legacy
single-holdout branch (`100 train / 16 dev / 0 test`) while the evaluator used the
shared three-way split (`101 / 12 / 12`). Two different sets were being reported as
"the held-out number". Worse, the *pretraining* leakage guard still used the v1
formula: **11 of 12 evaluation rows were present in the pretraining corpus.** All
three stages now call the same `split_examples()`.

Verified: **0 of 24** dev+test rows appear in the pretraining corpus, all **101**
train rows do, and a positive control with the guard disabled shows **24 of 24** —
so the test proves the guard works rather than merely passing.

### 2.3 Context truncation could not shrink a single long message — **fixed**

`Conversation.truncated()` dropped whole messages, so one long user message was
unsinkable. Measured: a long single message produced a **407-token prompt for a
128-token model**. It now trims from the front of the message, keeping the tail
where the actual instruction lives, and reserves room for the assistant opener.
Verified at ctx 64 and 128: never exceeds the window, and the final question
survives.

### 2.4 Data rebuilt

`boxz/data/code_quality.py` strips `<meta>`/`<link>`/`<title>`/importmap noise,
drops lines repeated in ≥ 40 files, and removes near-duplicate pages by 5-token
shingle Jaccard ≥ 0.85.

| | files | chars |
|---|---|---|
| before | 473 | 2,717,338 |
| after | 461 | 1,853,278 |
| change | −12 | **−31.8 %** |

`boxz/data/build_instruct_v2.py` replaces the v1 instruction set. v1's flaw was
structural: many questions shared one of a few canned answers, so the model could
only learn a lookup keyed on the question's opening words, and held-out rows were
unanswerable *by construction*. v2 writes each answer individually:
**125 examples → 101 train / 12 dev / 12 test**, no shared answer pools.

---

## 3. Before and after, measured

### 3.1 Held-out answer-token loss (the honest number)

Unweighted NLL over assistant tokens, on rows the optimiser never saw:

| checkpoint | train | dev | test |
|---|---|---|---|
| base-5m (pretrained only) | 5.741 | 5.509 | 5.764 |
| **sft5m (after SFT)** | **1.962** | **3.648** | **4.191** |
| change on held-out | | **−33.8 %** | **−27.3 %** |

Note that train < dev < test and that train (1.96) never collapses toward zero.
That ordering is what a healthy fit looks like; the run is no longer memorising a
handful of rows.

### 3.2 Generation quality — the part that did *not* improve

Same 20-prompt suite, greedy except where noted:

| checkpoint | params | requirement met | token hit | stops cleanly | contamination |
|---|---|---|---|---|---|
| base-1m | 3.09 M | 0.0 % | 2.4 % | 0.0 % | 10.0 % |
| sft (shipped before) | 3.09 M | 10.0 % | 9.8 % | 75.0 % | 5.0 % |
| **sft, temp 0.4 / rep 1.0** | 3.09 M | **20.0 %** | **14.6 %** | 55.0 % | **0.0 %** |
| base-5m | 5.31 M | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| **sft5m, greedy** | 5.31 M | **5.0 %** | 14.6 % | 25.0 % | 0.0 % |

**The larger, better-fitting model generates worse.** Its held-out likelihood is 34%
better and its output is still not better. Reported as measured, not explained away.

This is not a rounding artefact, and it is not mysterious: **45.9% per-token top-1
accuracy** on answer tokens sounds respectable until it is compounded. A
twenty-token answer is fully correct with probability 0.459²⁰ ≈ 1 in 10 million.
Twenty tokens of drift is exactly the `renderer, renderer, renderer` and
`function resize() { const d = (d + d;` in the transcripts. Likelihood and
generation quality are different questions, and only the first one moved.

Because the older 3.09 M checkpoint actually generates better, **it remains the
serving default**. Shipping the newer model because it is newer would be the
mistake this report is supposed to avoid.

### 3.3 Representative outputs

`hi` from the shipped checkpoint (unchanged behaviour class, penalty removed):

```
Hello. I am a small language model training model is a model projected model in the
camera of the model of the model.json group of a model.
```

The opening is correct English, fluent, and unmistakably Boxz. It then walks off the
cliff into code vocabulary. That is the characteristic failure mode of a model that
has learned *style* and not *content*.

---

## 4. What Boxz can genuinely do now (requirements 10, 11)

**Yes:**
- Produce fluent, on-style English openings for greetings, identity questions and
  short conceptual questions, and stop on its own most of the time (75% before the
  decode change).
- Name correct concepts and use correct vocabulary in context: attention, causal,
  softmax, cross-entropy, tokens, Three.js class names.
- Run the full from-scratch stack end to end — byte-level BPE, NumPy autograd,
  transformer, training, checkpointing, SFT, streaming inference, web chat — on a
  CPU, with no PyTorch, no pretrained weights and no external API.

**No, and the evaluation says so:**
- It does **not** write correct HTML, JavaScript or Three.js. It produces plausible
  *fragments* of them. Structural wellformedness on JS is 0–50% depending on
  checkpoint; on Three.js it is 0%.
- It does **not** reason. Asked `can you think?` it can produce a correct denial
  from memory, but that is a memorised answer, not a demonstration of reasoning.
- It does **not** hold factual knowledge. It is not a substitute for documentation.
- It has **no memory** between sessions, and its 128–192 token window means
  conversations degrade quickly.

20% of prompts meeting their requirements is a real, measured improvement over 10%,
and it is still a failing grade for a code assistant. Both halves of that sentence
belong in the report.

---

## 5. What the next milestone must be

The single most useful result from this round is *why* things stalled:

1. **Pretraining ran out of data, not steps.** Val loss fell to **4.2157 at step
   1250**, then rose to **4.2722** and **4.4138** while train loss kept falling. The
   corpus is only **769,040 train tokens**. A 5.3 M model wants roughly 100 M tokens
   for a compute-optimal fit; it got 9.2 M. Continuing to train was pure overfitting
   and was stopped deliberately.
2. **Generation is bounded by per-token accuracy, not by loss.** Getting from 46% to
   ~80% top-1 is what turns fragments into programs. That needs data and capacity
   together, not either alone.

The next milestone should therefore be **data-first**, and should not be measured in
parameters:

- **Grow the corpus by 20–50×** (15–40 M clean tokens). The filter and the split
  machinery built here are the prerequisite, and they now exist.
- **Weight the corpus toward the target domain** — real HTML/CSS/JS/Three.js with
  the boilerplate stripped — instead of adding more prose.
- **Report top-1 token accuracy as the primary metric**, alongside held-out loss.
  Loss alone let a 34%-better model ship with worse output once already.
- **Keep generation quality as the acceptance gate.** A change that improves loss
  but not the fixed suite is not a win, and this report's own 5.3 M model is the
  proof.

---

## Reproducing everything here

```bash
python -m boxz.data.code_quality --report          # corpus contamination numbers
python -m boxz.data.code_quality --clean           # writes clean.json (-31.8%)
python -m boxz.data.build_instruct_v2 --stats      # 125 examples -> 101/12/12
python -m boxz train --config configs/boxz-5m.json # pretrain 5.3M
python -m boxz sft --checkpoint checkpoints/base-5m/best --steps 800 --lr 1e-4
python -m boxz.eval.fixed_eval \
    --checkpoint checkpoints/sft/best \
    --checkpoint checkpoints/sft5m/best --text      # the 20-prompt suite
python -m pytest tests/                             # 149 passed, 5 skipped
```

Artifacts: `reports/fixed_eval_final.json`, `checkpoints/base-5m/`,
`checkpoints/sft5m/`, `logs/train_5m.log`, `logs/sft5m_v5.log`.
