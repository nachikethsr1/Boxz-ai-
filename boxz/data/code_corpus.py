"""Build a licensing-audited code corpus from MIT-licensed GitHub projects.

Why this exists
---------------
A 3 M-parameter model cannot write HTML/CSS/JavaScript/Three.js unless it has
*seen* those languages.  Tiny Shakespeare contains none of them, and 36
hand-written instruction pairs are far too few to learn a language from.

This fetcher pulls a bounded, reproducible slice of real web-game code from
projects whose licences explicitly permit redistribution:

===================  ======================================================
project              licence (checked at fetch time via the GitHub API)
===================  ======================================================
mrdoob/three.js      MIT -- 620 `examples/*.html` pages: WebGL, shaders,
                     animation, physics, games
jquery/jquery        MIT -- the library source itself (large, idiomatic JS)
processing/p5.js     LGPL-2.1 -- *not* redistributed; listed only to document
                     why it is excluded (copyleft file-level, still fine, but
                     we keep the corpus permissive to be safe)
===================  ======================================================

Policy
------
* Every file's project licence is verified through ``api.github.com`` before a
  single byte is written; a project whose declared licence does not match the
  allow-list aborts the fetch.
* Results are cached in ``data/raw/code_corpus/`` and the manifest records the
  exact commit SHA, per-file path, byte count and licence, so the corpus is
  reproducible and auditable.
* Nothing is committed to git: only the manifest and the fetcher.
* HTML files are stripped of their vendored CDN <script> imports (we keep the
  page's own inline code) to avoid embedding third-party bundles.

Usage
-----
    python -m boxz.data.code_corpus fetch --max-files 400
    python -m boxz.data.code_corpus stats
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["CodeProject", "PROJECTS", "fetch_corpus", "corpus_stats", "main"]

_API = "https://api.github.com"
_TIMEOUT = 60

#: Projects we are willing to redistribute excerpts from, with the licence we
#: require the API to confirm.  Anything else is refused.
PROJECTS: List["CodeProject"] = []


@dataclass
class CodeProject:
    repo: str                 # "owner/name"
    licence: str              # SPDX id we require, e.g. "MIT"
    paths: List[str]          # directories to harvest
    glob: str                 # file extension filter, e.g. ".html"
    purpose: str

    def to_dict(self) -> dict:
        return {"repo": self.repo, "licence": self.licence,
                "paths": self.paths, "glob": self.glob, "purpose": self.purpose}


PROJECTS.extend([
    CodeProject("mrdoob/three.js", "MIT", ["examples"], ".html",
                "Three.js example pages: WebGL, shaders, animation, games"),
    CodeProject("jquery/jquery", "MIT", ["src"], ".js",
                "jQuery source: battle-tested JavaScript, DOM and events"),
])


# ----------------------------------------------------------------------
# HTTP helpers (stdlib only, no dependency on the fetch registry)
# ----------------------------------------------------------------------
def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "boxz-corpus-builder/1.0",
    })
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_file_text(repo: str, path: str, ref: str) -> Optional[str]:
    """Fetch one file via the contents API (base64 payload) as text."""
    data = _get_json(f"{_API}/repos/{repo}/contents/{path}?ref={ref}")
    if not isinstance(data, dict):
        return None
    if data.get("encoding") == "base64" and data.get("content"):
        raw = base64.b64decode(data["content"])
        return raw.decode("utf-8", errors="replace")
    if data.get("download_url"):
        req = urllib.request.Request(data["download_url"],
                                     headers={"User-Agent": "boxz-corpus-builder/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.read().decode("utf-8", errors="replace")
    return None


def _safe_get(repo: str, path: str, ref: str) -> Optional[str]:
    """Fetch one file, returning None on any failure (missing, binary, rate limit)."""
    try:
        return _get_file_text(repo, path, ref)
    except Exception:                          # noqa: BLE001
        return None


def verify_licence(project: CodeProject) -> Dict[str, str]:
    """Confirm the project's licence through the GitHub API before fetching."""
    meta = _get_json(f"{_API}/repos/{project.repo}")
    assert isinstance(meta, dict)
    spdx = (meta.get("license") or {}).get("spdx_id")
    if spdx != project.licence:
        raise RuntimeError(
            f"licence mismatch for {project.repo}: expected {project.licence}, "
            f"GitHub reports {spdx!r}. Refusing to add this project to the corpus."
        )
    return {"spdx": spdx, "default_branch": meta["default_branch"],
            "commit": _default_commit(project.repo, meta["default_branch"]),
            "stars": meta.get("stargazers_count", 0)}


def _default_commit(repo: str, branch: str) -> str:
    data = _get_json(f"{_API}/repos/{repo}/commits/{branch}")
    assert isinstance(data, dict)
    return str(data["sha"])


def _list_files(project: CodeProject, ref: str) -> List[str]:
    """Recursively list matching files under the project's paths."""
    out: List[str] = []
    for base in project.paths:
        stack = [base]
        while stack:
            path = stack.pop()
            try:
                entries = _get_json(f"{_API}/repos/{project.repo}/contents/{path}?ref={ref}")
            except Exception:                      # noqa: BLE001 - network shape varies
                continue
            if not isinstance(entries, list):
                continue
            for e in entries:
                if e.get("type") == "dir":
                    stack.append(e["path"])
                elif e.get("type") == "file" and e["name"].endswith(project.glob):
                    out.append(e["path"])
    return sorted(set(out))


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
_SCRIPT_SRC_RE = re.compile(r"<script[^>]+src=[\"'][^\"']*[\"'][^>]*>\s*</script>", re.I)
_STYLE_LINK_RE = re.compile(r"<link[^>]+rel=[\"']stylesheet[\"'][^>]*>", re.I)
_IMPORT_MAP_RE = re.compile(r"<script[^>]+type=[\"']importmap[\"'][^>]*>.*?</script>", re.I | re.S)


def clean_code(text: str) -> str:
    """Drop vendored/CDN includes but keep the page's own markup and code."""
    text = _IMPORT_MAP_RE.sub("", text)
    text = _SCRIPT_SRC_RE.sub("", text)
    text = _STYLE_LINK_RE.sub("", text)
    # collapse runs of blank lines so token budget is spent on code, not air
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------
def fetch_corpus(out_dir: str = "data/raw/code_corpus", max_files: int = 400,
                 per_file_limit: int = 12000, verbose: bool = True) -> dict:
    """Fetch, clean and cache the corpus; return the manifest."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    docs: List[dict] = []
    project_meta: Dict[str, dict] = {}

    for project in PROJECTS:
        meta = verify_licence(project)
        project_meta[project.repo] = {**project.to_dict(), **meta}
        if verbose:
            print(f"[code] {project.repo}: licence {meta['spdx']} confirmed @ "
                  f"{meta['commit'][:8]} ({project.glob})", flush=True)
        files = _list_files(project, meta["commit"])
        if verbose:
            print(f"[code]   {len(files)} candidate files under {project.paths}", flush=True)
        # Fetch concurrently: the contents API is one request per file and a
        # sequential crawl of 500 files takes over ten minutes.
        todo = files[:max_files * 2]
        with ThreadPoolExecutor(max_workers=12) as pool:
            fetched = list(pool.map(
                lambda path: (path, _safe_get(project.repo, path, meta["commit"])),
                todo,
            ))
        taken = 0
        for path, raw in fetched:
            if taken >= max_files:
                break
            if not raw:
                continue
            code = clean_code(raw)[:per_file_limit]
            if len(code) < 400:                # skip stubs
                continue
            docs.append({"repo": project.repo, "path": path, "licence": meta["spdx"],
                         "commit": meta["commit"], "chars": len(code), "code": code})
            taken += 1
        if verbose:
            print(f"[code]   kept {taken} files ({project.glob})")

    manifest = {
        "licences": project_meta,
        "files": [{k: v for k, v in d.items() if k != "code"} for d in docs],
        "total_files": len(docs),
        "total_chars": sum(d["chars"] for d in docs),
    }
    (root / "corpus.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(f"[code] corpus: {manifest['total_files']} files, "
              f"{manifest['total_chars']:,} chars -> {root / 'corpus.json'}")
    return manifest


def corpus_stats(corpus_path: str = "data/raw/code_corpus/corpus.json") -> dict:
    p = Path(corpus_path)
    if not p.exists():
        raise FileNotFoundError(f"no corpus at {p}; run: python -m boxz.data.code_corpus fetch")
    docs = json.loads(p.read_text(encoding="utf-8"))
    by_repo: Dict[str, int] = {}
    for d in docs:
        by_repo[d["repo"]] = by_repo.get(d["repo"], 0) + 1
    return {"files": len(docs), "chars": sum(d["chars"] for d in docs),
            "by_repo": by_repo, "bytes": p.stat().st_size}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="boxz.data.code_corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and cache the code corpus")
    f.add_argument("--out", default="data/raw/code_corpus")
    f.add_argument("--max-files", type=int, default=400)
    f.add_argument("--per-file-limit", type=int, default=12000)
    s = sub.add_parser("stats", help="report corpus size")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fetch_corpus(args.out, args.max_files, args.per_file_limit)
    else:
        print(json.dumps(corpus_stats(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
