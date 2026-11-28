# Boxz Architecture

Boxz is a **decoder-only Transformer language model implemented from scratch**.
Every part of the learning system — autograd, layers, attention, optimiser,
schedules, tokenizer, data pipeline, sampler — is written in this repository on
top of NumPy. There is no PyTorch, TensorFlow, JAX, Keras, Hugging Face
Transformers, no pretrained weights, and no external AI API anywhere in the
training or inference path.

```
raw text / instruction JSONL
        │
        ▼
┌─────────────────────────┐
│ data/                   │  loading, cleaning, de-duplication, splits, batching
│  registry.py corpus.py  │  dataset provenance lives in data/sources/*.json
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ tokenizer/bpe.py        │  byte-level BPE, trained here, persisted as JSON
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ model/                  │  decoder-only Transformer
│  attention.py           │    causal multi-head self-attention (fused QKV)
│  transformer.py         │    pre-norm blocks, GELU MLP, tied embeddings
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ engine/                 │  reverse-mode autograd over NumPy
│  tensor.py ops.py       │    ~30 differentiable Functions, verified by FD tests
│  layers.py optim.py     │    Module/Parameter, AdamW, SGD, LR schedules
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ training/               │  Trainer, checkpoints, evaluation, SFT, benchmarks
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ inference/  chat/  app/ │  sampler (temp/top-k/top-p/repetition), KV cache,
│                         │  conversation format/engine, HTTP server + chat UI
└─────────────────────────┘
            ▲
            │ optional, measured kernels
        rust/boxz_rust      PyO3 extension with a NumPy reference fallback
```

## 1. Numerical engine (`boxz/engine`)

### Tensors and the store

A `Tensor` wraps a `float32` NumPy array. Array payloads live in a
process-local `TensorStore` and the tensor object only stores integer handles,
which keeps graph construction cheap (a training step builds ~200 nodes).

* **Graph nodes** (`_Node`) are created only for tensors that require grad and
  participate in an operation; constants share one sentinel tensor, and the
  *actual* constant value (`shapes`, indices, masks) is kept beside it so the
  backward pass can replay the exact call.
* **Backward** is an iterative post-order DFS (no recursion limit issues),
  with fast paths for unary and binary nodes.
* **Memory**: gradients are allocated lazily; `Tensor.free_graph()` drops saved
  activations immediately. Without it a step would retain ~200 multi-MB buffers
  until the cyclic GC runs — this was measured, fixed, and is covered by a test
  (`test_model_free_graph_and_training_step_memory`).

### Operations

`ops.py` defines the differentiable functions used by the model: add/sub/mul/
div/pow/matmul, reshape/transpose/slice/gather, sum/mean/exp/log/sqrt/clip,
`softmax`, `log_softmax`, `masked_softmax`, `gelu`, `relu`, `silu`,
`layer_norm`, `rms_norm`, and `cross_entropy_with_logits`.

Every one of them is checked against central finite differences in
`tests/test_tensor.py`, and the *whole model* is checked with a directional
derivative over all parameters in `test_full_model_directional_gradient_check`.

Two implementation rules were needed to make this both correct and fast:

1. **Saved buffers are read-only.** `ctx.save(x=...)` stores a reference to a
   live tensor's buffer; a backward pass that mutates it corrupts the caller.
2. **In-place arithmetic only on freshly allocated arrays.** Chained NumPy
   expressions like `0.5 * x * (1 + t)` fall into a slow scalar-casting path
   (measured 19.6 ms vs 2.9 ms on a 2.4 M-element tensor), so hot kernels
   allocate one temporary and use `out=`/in-place updates.

### Performance notes (measured, not guessed)

On a 4-core laptop CPU the training step is dominated by BLAS. For the
3.1 M-parameter model at `batch=24, ctx=128` a step is ~1.6 s; NumPy/OpenBLAS
delivers ~175–235 GFLOP/s on the relevant shapes, which is within ~15 % of the
AVX2 FMA ceiling. Consequently:

* Large matmuls stay in NumPy. Rewriting them in Rust would be slower.
* Wins came from *removing* pathology, not from replacing BLAS:
  `np.add.at` → `bincount`/`reduceat` (−0.13 s/step), lazy gradient allocation
  (−0.04 s), direct backward dispatch instead of `*args` unpacking (−0.3 s),
  eager graph release (memory flat instead of growing 1.4 GB/step).

## 2. Tokenizer (`boxz/tokenizer/bpe.py`)

Byte-level BPE, trained with incremental pair counting:

* ids `0..7` are special tokens (`<pad> <unk> <bos> <eos> <|user|> <|assistant|> <|system|> <|end|>`), then 256 byte tokens, then one id per merge — a stable layout that makes reloading reproduce token ids exactly.
* Any Unicode string round-trips losslessly: rare text falls back to bytes rather than `<unk>`.
* Pre-tokenisation keeps a leading space with the following word (GPT-2 convention) and keeps `snake_case` identifiers intact — important for code.
* `save`/`load` write JSON plus a human-readable vocabulary dump.

A subtle bug found by tests: a `\s+(?!\S)` alternative in the pre-tokeniser
backtracks and silently drops the last character of some runs. Every character
must be covered by exactly one alternative; `test_roundtrip_is_lossless` guards
this permanently.

## 3. Model (`boxz/model`)

Pre-norm decoder-only Transformer:

```
x = tok_emb(tokens) + pos_emb[:T]
for block in blocks:               # × n_layer
    x = x + attn(ln_attn(x))       # causal MHA, fused qkv projection
    x = x + mlp(ln_mlp(x))         # Linear -> GELU -> Linear
x = ln_final(x)
logits = x @ tok_emb.weightᵀ       # tied embeddings
loss   = cross_entropy(logits, targets)
```

Configurable: layers, heads, width, feed-forward width, context length,
dropout, LayerNorm vs RMSNorm, GELU/ReLU/SiLU, tied vs untied head, bias.
Initialisation scales residual projections by `1/sqrt(2·n_layer)`.

Attention is implemented with explicit head reshaping and a cached causal mask
(`j <= i`); masked positions receive `-30` *after* subtracting the row max,
which avoids subnormal `exp` values (subnormals are ~10× slower on x86 and were
the single largest hotspot before the fix).

## 4. Training (`boxz/training`)

* `Trainer` — deterministic init from a seed, AdamW, cosine/linear/constant
  schedule with warmup, gradient clipping, gradient accumulation, periodic
  evaluation, best-checkpoint tracking, JSONL logs, wall-clock guard.
* `checkpoint.py` — a checkpoint is a *directory*: `model.npz` (arrays),
  `config.json` (architecture + run config), `tokenizer.json`, `train_state.json`
  (step, tokens seen, history), `optimizer.npz` (Adam moments for exact resume).
  No pickle, no code execution on load.
* `evaluate.py` — validation loss, perplexity, bits-per-byte (comparable across
  tokenizers), next-token accuracy.
* `sft.py` — instruction tuning with **assistant-only loss masking**: prompt
  labels are replaced by `<pad>` and ignored by the cross entropy, so the model
  learns to answer, not to echo.
* `benchmark_suite.py` — capability evaluation (format compliance, continuation
  samples, instruction loss, throughput).

## 5. Inference (`boxz/inference`, `boxz/chat`)

* Sampling: temperature, top-k, top-p (nucleus), repetition penalty, greedy.
* **Byte-hygiene mask**: tokens that decode to control bytes and all special
  tokens are excluded from sampling. This is a decoding constraint, not a
  canned answer, and it keeps replies printable.
* **KV cache**: each generated token costs one time step instead of a full
  re-read of the sequence. Verified to match the full forward pass to 6e-8 and
  to produce identical greedy output; roughly 4–8× faster for 100+ tokens.
* `Conversation`/`ChatEngine`: one canonical chat format using special tokens
  (user text can never spoof a role boundary), history truncation by token
  budget, and stop-token handling.

## 6. Application (`boxz/app`)

A dependency-free HTTP server (standard library) serving a single-page chat UI
and a JSON API (`/api/status`, `/api/chat`, `/api/history`, `/api/reset`,
`/api/metrics`). The UI shows model metadata, generation sliders, live
tokens/second, and a loss sparkline read from `train_log.jsonl`. Every reply is
generated by the loaded checkpoint.

## 7. Rust engine (`rust/boxz_rust`)

Optional PyO3 kernels for GELU, softmax, LayerNorm and sampling, with an
automatic NumPy fallback and a parity harness. See its README for the measured
justification of what is — and is not — accelerated.
