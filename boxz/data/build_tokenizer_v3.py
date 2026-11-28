"""Train the Boxz V3 tokenizer on the V3 corpus.

Vocab 4096: the 2048-vocab proxy tokenizer measured 1.0-1.3 bytes/token on
digits, which is disastrous for a maths model (a 6-digit number costs 6 tokens).
Doubling the vocabulary gives digit pairs and short web-code tokens their own
IDs, and vocab 4096 is also required by the 61.7M parameter target.

CLI:  python -m boxz.data.build_tokenizer_v3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from boxz.tokenizer.bpe import BPETokenizer


def corpus_text() -> str:
    """Every unique training row, plus the raw prose domain it should generalise to."""
    parts = []
    for p in ("data/processed/v3/v3_train.jsonl", "data/processed/v3/v3_val.jsonl"):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            for m in obj["messages"]:
                parts.append(m["content"])
    return "\n".join(parts)


def main() -> int:
    text = corpus_text()
    print(f"tokenizer training text: {len(text):,} chars", flush=True)
    tok = BPETokenizer()
    tok.train(text, vocab_size=4096, verbose=True)
    out = Path("data/processed/v3/tokenizer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    st = tok.stats()
    print(f"saved {out}  vocab={tok.vocab_size}  merges={st.get('merges')}", flush=True)
    # byte-efficiency probes
    for probe in ["482736 + 918264", "What is 2,194,638 + 6,484,738?",
                  "document.getElementById('btn')",
                  "<div class=\"game\"></div>", "function updateScore(n) {"]:
        print(f"  {len(tok.encode(probe)):3d} tok  {probe!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
