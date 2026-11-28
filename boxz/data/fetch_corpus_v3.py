"""Fetch real JavaScript / CSS / HTML source from the npm registry.

Why: the instruction rows total only ~265k tokens.  A 61.7M-parameter model
needs orders of magnitude more text than that to learn syntax at all.  The npm
registry is a licensed, no-auth source of genuine web source code, which is
exactly Boxz's target domain.

Only ``.js``/``.mjs``/``.cjs``/``.css``/``.html`` files inside each tarball are
kept.  Files are filtered for size (skip minified bundles and >200 kB giants),
deduplicated by content hash, and written as plain text under ``data/raw/npm``.

CLI::

    python -m boxz.data.fetch_corpus_v3 --target-mb 25
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

__all__ = ["PACKAGES", "fetch_packages", "clean_text"]

REGISTRY = "https://registry.npmjs.org"

#: Real libraries a web-game developer would actually use.  Mixing a few large
#: utility libraries with tiny focused ones keeps the style distribution broad.
PACKAGES = [
    # animation / 3D
    "three", "gsap", "tween.js", "@tweenjs/tween.js", "matter-js", "pixi.js",
    "phaser", "kaboom", "melonjs", "box2d-wasm", "ammo.js",
    # utilities commonly seen in game code
    "lodash", "underscore", "ramda", "date-fns", "nanoid", "uuid", "qs",
    "axios", "chalk", "debug", "ms", "semver", "validator", "slugify",
    # state / events
    "eventemitter3", "mitt", "nanostores", "zustand-lite", "tiny-invariant",
    # dom helpers
    "dompurify", "hyperscript", "htm", "lit-html", "preact", "solid-js",
    # css / bundling ecosystem sources
    "normalize.css", "animate.css", "bulma", "tailwindcss", "postcss",
    # audio
    "howler", "tone",
]

KEEP = {".js", ".mjs", ".cjs", ".css", ".html", ".htm"}
MIN_BYTES = 400
MAX_BYTES = 200_000
SKIP = re.compile(r"(\.min\.|\.map$|package-lock|\.d\.ts$|/dist/.*bundle|node_modules/)")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "boxz-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def clean_text(s: str) -> str:
    """Strip licence boilerplate that would otherwise dominate the corpus."""
    lines = []
    in_block = False
    for ln in s.splitlines():
        st = ln.strip()
        low = st.lower()
        if st.startswith("/*") and not st.startswith("/**"):
            in_block = True
        if in_block:
            if "*/" in st:
                in_block = False
            continue
        if re.match(r"^\s*(//|/\*|\*)\s*(copyright|\(c\)|licen[cs]e|@license|"
                    r"@preserve|mit license|the mit)", st, re.I):
            continue
        if len(ln) > 400:                    # minified line: skip the file later
            return ""
        lines.append(ln)
    out = "\n".join(lines)
    return out if len(out) and "\x00" not in out else ""


def fetch_packages(target_bytes: int, out_dir: Path, packages: list[str] | None = None,
                   verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    total = 0
    stats = {"packages_tried": 0, "packages_ok": 0, "files": 0, "bytes": 0,
             "skipped_big": 0, "skipped_dup": 0, "failed_packages": []}
    buf: list[str] = []
    chunk_no = 0

    def flush() -> None:
        nonlocal buf, chunk_no, total
        if not buf:
            return
        text = "\n\n".join(buf)
        p = out_dir / f"npm_{chunk_no:03d}.txt"
        p.write_text(text, encoding="utf-8")
        total += len(text.encode())
        stats["bytes"] = total
        chunk_no += 1
        buf = []

    for name in packages or PACKAGES:
        if total >= target_bytes:
            break
        stats["packages_tried"] += 1
        try:
            meta = json.loads(_get(f"{REGISTRY}/{name.replace('/', '%2f')}"))
            latest = meta["dist-tags"]["latest"]
            tarball = meta["versions"][latest]["dist"]["tarball"]
            raw = _get(tarball, timeout=90)
        except Exception as e:                       # noqa: BLE001 - network variance
            stats["failed_packages"].append(f"{name}: {type(e).__name__}")
            if verbose:
                print(f"  skip {name}: {type(e).__name__}", file=sys.stderr, flush=True)
            continue
        stats["packages_ok"] += 1
        got = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if Path(member.name).suffix.lower() not in KEEP:
                        continue
                    if SKIP.search(member.name):
                        continue
                    if member.size < MIN_BYTES:
                        continue
                    if member.size > MAX_BYTES:
                        stats["skipped_big"] += 1
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    try:
                        txt = fh.read().decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        continue
                    clean = clean_text(txt)
                    if len(clean) < MIN_BYTES:
                        continue
                    h = hashlib.sha256(clean.encode()).hexdigest()
                    if h in seen_hashes:
                        stats["skipped_dup"] += 1
                        continue
                    seen_hashes.add(h)
                    buf.append(f"// {name}/{Path(member.name).name}\n{clean}")
                    stats["files"] += 1
                    got += 1
                    if sum(len(b) for b in buf) > 4_000_000:
                        flush()
        except tarfile.TarError:
            stats["failed_packages"].append(f"{name}: bad tarball")
            continue
        if verbose:
            print(f"  {name:28s} {got:4d} files   total {stats['bytes']/1e6:6.2f} MB",
                  file=sys.stderr, flush=True)
    flush()
    stats["chunks"] = chunk_no
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=Path("data/raw/npm"))
    args = ap.parse_args(argv)
    st = fetch_packages(int(args.target_mb * 1_000_000), args.out)
    print(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
