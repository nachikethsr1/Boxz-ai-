"""Byte-level BPE tokenizer implemented from scratch (no tokenizers library).

Why byte-level?
---------------
Mapping text to UTF-8 bytes guarantees that **any** input can be encoded
losslessly (there is no such thing as an unknown character), which is a hard
requirement for a model that has to read arbitrary code and user messages.
Sub-word merges are then learned greedily on byte sequences, so common words,
identifiers and code patterns collapse into single tokens while rare text falls
back to bytes instead of producing ``<unk>``.

Layout of the vocabulary
------------------------
* ids ``0 .. S-1``              special tokens (``<pad>``, ``<unk>``, chat markers)
* ids ``S .. S+255``            the 256 base byte tokens
* ids ``S+256 .. S+256+M-1``    one id per learned merge, in merge order

This is a *stable* layout: re-loading a vocabulary reproduces token ids exactly,
which is required for checkpoint compatibility.

Training uses the classic incremental pair-counting algorithm
(Sennrich et al., 2016): only words touched by the current merge have their
pair statistics updated, which keeps training linear in corpus size.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "BPETokenizer",
    "DEFAULT_SPECIALS",
    "bytes_to_unicode",
    "unicode_to_bytes",
    "PRETOKEN_PATTERN",
]

# Special tokens: the chat format reserves its own markers so that user text can
# never be confused with a role boundary.
DEFAULT_SPECIALS: Tuple[str, ...] = (
    "<pad>",        # 0 - padding / ignore index
    "<unk>",        # 1 - never produced by byte-level encoding; API safety net
    "<bos>",        # 2 - begin of sequence
    "<eos>",        # 3 - end of sequence
    "<|user|>",     # 4 - chat: user turn
    "<|assistant|>",  # 5 - chat: assistant turn
    "<|system|>",   # 6 - chat: system prompt
    "<|end|>",      # 7 - chat: end of assistant turn
)

# Pre-tokenisation.  ``\w`` in a str pattern is Unicode-aware, so ``[^\W\d_]``
# means "letters".  A leading optional space is kept attached to the following
# word ("foo bar" -> ["foo", " bar"]), which is the GPT-2 convention.
#
# NOTE: there is deliberately no ``\s+(?!\S)`` alternative here.  A failing
# lookahead makes the regex engine backtrack into the preceding alternative, and
# because the leading " ?" is optional the match silently shortens ("^_" would
# tokenise as "^" while "_" vanished).  Every character must be covered by
# exactly one alternative; ``test_roundtrip_is_lossless`` guards this.
PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"          # contractions
    r"| ?[^\W\d]+"                   # optional space + letters/digits/underscore
    r"| ?\d+"                        # optional space + digits
    r"| ?[^\s\w]+"                   # optional space + punctuation/symbols
    r"|\s+"                          # whitespace runs
)


def bytes_to_unicode() -> Dict[int, str]:
    """Map the 256 byte values to printable unicode chars (GPT-2 convention)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def unicode_to_bytes(s: str) -> bytes:
    """Inverse of :func:`bytes_to_unicode` for a token string."""
    return bytes(_UNICODE_TO_BYTE[ch] for ch in s)


@dataclass
class BPEConfig:
    vocab_size: int = 4096
    specials: Tuple[str, ...] = DEFAULT_SPECIALS
    min_frequency: int = 2
    verbose: bool = False


class BPETokenizer:
    """A byte-level byte-pair-encoding tokenizer."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    SYSTEM = "<|system|>"
    END = "<|end|>"

    def __init__(self, specials: Sequence[str] = DEFAULT_SPECIALS) -> None:
        self.specials: List[str] = list(specials)
        self.special_to_id: Dict[str, int] = {s: i for i, s in enumerate(self.specials)}
        # token id -> token string  (base bytes first, then merges)
        self.id_to_token: List[str] = []
        self.merges: List[Tuple[int, int]] = []
        self.ranks: Dict[Tuple[int, int], int] = {}
        self._byte_ids: List[int] = []
        self._cache: Dict[str, List[int]] = {}
        self._special_re: Optional[re.Pattern] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @property
    def n_specials(self) -> int:
        return len(self.specials)

    @property
    def byte_offset(self) -> int:
        return self.n_specials

    @property
    def vocab_size(self) -> int:
        return self.n_specials + 256 + len(self.merges)

    def _init_base_vocab(self) -> None:
        self.id_to_token = []
        self._byte_ids = []
        for i in range(256):
            tok = _BYTE_TO_UNICODE[i]
            self.id_to_token.append(tok)
            self._byte_ids.append(self.byte_offset + i)
        self.merges = []
        self.ranks = {}
        self._cache = {}
        self._build_special_re()

    def _build_special_re(self) -> None:
        # Longest-first so that <|assistant|> wins over short prefixes.
        # The wrapping group is required: re.split only keeps captured separators.
        specials = sorted(self.specials, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")

    # ------------------------------------------------------------------
    # ids of special tokens
    # ------------------------------------------------------------------
    def special_id(self, token: str) -> int:
        return self.special_to_id[token]

    @property
    def pad_id(self) -> int:
        return self.special_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.special_to_id[self.UNK]

    @property
    def eos_id(self) -> int:
        return self.special_to_id[self.EOS]

    @property
    def bos_id(self) -> int:
        return self.special_to_id[self.BOS]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(
        self,
        texts: Union[str, Iterable[str]],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn merges until ``vocab_size`` tokens exist.

        ``texts`` may be a single string or an iterable of strings.
        """
        if isinstance(texts, str):
            corpus: Iterable[str] = [texts]
        else:
            corpus = texts
        if vocab_size <= self.n_specials + 256:
            raise ValueError(f"vocab_size must exceed {self.n_specials + 256}")

        self._init_base_vocab()
        t0 = time.time()

        # 1. pre-tokenise and count word frequencies ---------------------
        word_counts: Counter[str] = Counter()
        total_words = 0
        for text in corpus:
            for m in PRETOKEN_PATTERN.finditer(text):
                w = m.group(0)
                word_counts[w] += 1
                total_words += 1

        # 2. words as lists of byte ids ---------------------------------
        words: List[List[int]] = []
        counts: List[int] = []
        for w, c in word_counts.items():
            ids = [self.byte_offset + b for b in w.encode("utf-8")]
            words.append(ids)
            counts.append(c)

        # 3. pair statistics with an inverted index pair -> word ids -----
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_words: Dict[Tuple[int, int], set] = defaultdict(set)
        for wi, ids in enumerate(words):
            c = counts[wi]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += c
                pair_words[(a, b)].add(wi)

        num_merges = vocab_size - (self.n_specials + 256)
        for step in range(num_merges):
            if not pair_counts:
                break
            # most frequent pair (ties broken deterministically by pair value)
            best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            (a, b), freq = best
            if freq < min_frequency:
                break
            new_id = self.n_specials + 256 + len(self.merges)
            self.merges.append((a, b))
            self.ranks[(a, b)] = len(self.ranks)
            self.id_to_token.append(self.id_to_token[a - self.byte_offset] + self.id_to_token[b - self.byte_offset])

            affected = list(pair_words.pop((a, b), set()))
            pair_counts.pop((a, b), None)

            for wi in affected:
                ids = words[wi]
                c = counts[wi]
                out: List[int] = []
                i = 0
                n = len(ids)
                touched: List[Tuple[int, int]] = []
                while i < n:
                    if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                        touched.append((i, new_id))
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                if len(out) == len(ids):
                    continue
                # recompute pair stats for this word from scratch (words are short)
                old_pairs = Counter(zip(ids, ids[1:]))
                new_pairs = Counter(zip(out, out[1:]))
                for p, k in old_pairs.items():
                    pair_counts[p] -= c * k
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                for p, k in new_pairs.items():
                    pair_counts[p] += c * k
                    pair_words[p].add(wi)
                words[wi] = out

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{num_merges}  pair=({a},{b}) freq={freq} "
                      f"tok={self.id_to_token[-1]!r} vocab={self.vocab_size} "
                      f"({time.time() - t0:.1f}s)")

        self._cache = {}
        if verbose:
            print(f"trained BPE: vocab={self.vocab_size} merges={len(self.merges)} "
                  f"from {total_words} pretokens in {time.time() - t0:.1f}s")
        return self

    # ------------------------------------------------------------------
    # encoding / decoding
    # ------------------------------------------------------------------
    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [self.byte_offset + b for b in word.encode("utf-8")]
        if not self.ranks:
            out = ids
        else:
            while len(ids) >= 2:
                best_rank = None
                best_pos = -1
                for i in range(len(ids) - 1):
                    r = self.ranks.get((ids[i], ids[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank = r
                        best_pos = i
                if best_rank is None:
                    break
                a, b = ids[best_pos], ids[best_pos + 1]
                new_id = self.n_specials + 256 + best_rank
                ids[best_pos:best_pos + 2] = [new_id]
            out = ids
        if len(self._cache) < 200_000:
            self._cache[word] = out
        return out

    def encode(self, text: str, add_special_tokens: bool = False,
               allowed_special: bool = True) -> List[int]:
        """Encode ``text`` into token ids.

        Special-token strings inside ``text`` (``<|user|>`` etc.) are recognised
        as atomic tokens when ``allowed_special`` is true; otherwise they are
        encoded as literal text.  This is what makes prompt formatting safe.
        """
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        if allowed_special and self._special_re is not None:
            parts = self._special_re.split(text)
            specials = set(self.specials)
            for part in parts:
                if part in specials:
                    ids.append(self.special_to_id[part])
                elif part:
                    for m in PRETOKEN_PATTERN.finditer(part):
                        ids.extend(self._encode_word(m.group(0)))
        else:
            for m in PRETOKEN_PATTERN.finditer(text):
                ids.extend(self._encode_word(m.group(0)))
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back into text (lossless for byte-level tokens)."""
        chunks: List[bytes] = []
        n_specials = self.n_specials
        for i in ids:
            i = int(i)
            if i < 0:
                continue
            if i < n_specials:
                if not skip_special_tokens:
                    chunks.append(self.specials[i].encode("utf-8"))
                continue
            if i >= self.vocab_size:
                chunks.append(self.UNK.encode())
                continue
            tok = self.id_to_token[i - n_specials]
            chunks.append(unicode_to_bytes(tok))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def decode_with_offsets(self, ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
        """Decode and return byte spans per token (used by the web UI)."""
        out = bytearray()
        spans: List[Tuple[int, int]] = []
        for i in ids:
            i = int(i)
            start = len(out)
            if 0 <= i < self.n_specials:
                out.extend(self.specials[i].encode())
            elif i < self.vocab_size:
                out.extend(unicode_to_bytes(self.id_to_token[i - self.n_specials]))
            spans.append((start, len(out)))
        return out.decode("utf-8", errors="replace"), spans

    # convenience ---------------------------------------------------------
    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "boxz-bpe-v1",
            "specials": list(self.specials),
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        # A plain-text view helps humans inspect what the model learned.
        txt = path.with_suffix(".vocab.txt")
        with txt.open("w", encoding="utf-8") as f:
            for i, tok in enumerate(self.id_to_token):
                f.write(f"{i + self.n_specials}\t{tok!r}\n")

    @classmethod
    def from_dict(cls, obj: dict) -> "BPETokenizer":
        tok = cls(specials=tuple(obj.get("specials", DEFAULT_SPECIALS)))
        tok._init_base_vocab()
        for a, b in obj.get("merges", []):
            a = int(a)
            b = int(b)
            tok.merges.append((a, b))
            tok.ranks[(a, b)] = len(tok.ranks)
            tok.id_to_token.append(tok.id_to_token[a - tok.byte_offset] + tok.id_to_token[b - tok.byte_offset])
        tok._cache = {}
        return tok

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def token_bytes(self, idx: int) -> bytes:
        if idx < self.n_specials:
            return self.specials[idx].encode()
        return unicode_to_bytes(self.id_to_token[idx - self.n_specials])

    def token_str(self, idx: int) -> str:
        return self.decode([idx], skip_special_tokens=False)

    def stats(self) -> dict:
        lens = [len(self.token_bytes(i)) for i in range(self.n_specials, self.vocab_size)]
        return {
            "vocab_size": self.vocab_size,
            "specials": self.n_specials,
            "merges": len(self.merges),
            "mean_token_bytes": float(sum(lens) / max(1, len(lens))),
            "max_token_bytes": max(lens) if lens else 0,
        }
