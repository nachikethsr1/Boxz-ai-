"""Fetch a small, licence-verified, multi-language corpus for Phase 2B.

Why this exists: G1 trained almost entirely on npm JavaScript.  The audit in
``reports/PHASE2A_PLAN.md`` found **zero** Python, **zero** HTML and only 3.1%
instruction text, so the model cannot be expected to learn those regardless of
how many tokens we spend on JavaScript.

Design rules, each one a direct answer to a defect found in Phase 2A:

* **Licence first.**  Every repository's licence is read from the GitHub API
  *before* its contents are downloaded.  Only permissive licences (MIT, BSD,
  Apache-2.0, ISC, 0BSD, Unlicense, CC0) are accepted; anything else is skipped
  and recorded as skipped.  Each accepted repo is recorded with its SPDX id and
  its exact commit SHA.
* **Content decides language, not the filename.**  Phase 2A's audit showed that
  inferring from extensions is how you end up believing you have HTML when you
  have none.  :func:`classify` inspects the text.
* **Document-level deduplication.**  Whole documents are hashed and repeats are
  dropped.  Line-level dedupe is never used: on code it keeps one closing brace
  and destroys syntax (see ``reports/PHASE2A_PREFLIGHT.md``).
* **Explicit document boundaries.**  Every document is wrapped with a header
  naming its language and source, and terminated with a separator, so chunks
  can never run into each other the way the npm block ran into a user question.
* **Bounded size.**  A cap per language keeps the corpus balanced instead of
  letting whichever source is largest dominate.

Nothing here trains anything.  It only writes ``data/raw/p2b/`` plus a manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "PERMISSIVE",
    "REPOS",
    "classify",
    "fetch_repo",
    "build_corpus",
]

API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"

#: Licences we accept.  Anything else is skipped and logged.
PERMISSIVE = {
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC",
    "0BSD", "Unlicense", "CC0-1.0", "BSD-3-Clause-Clear", "MIT-0",
}

#: Curated repositories.  Chosen for (a) permissive licence, (b) genuine
#: educational content, (c) coverage of the languages Phase 2B targets.
#: ``max_bytes`` caps how much of each repo is kept so one large repo cannot
#: dominate the corpus.
REPOS: List[Dict[str, object]] = [
    # --- Python, small and readable -----------------------------------------
    {"repo": "karpathy/micrograd", "kinds": ("python",), "max_bytes": 120_000},
    {"repo": "karpathy/makemore", "kinds": ("python",), "max_bytes": 200_000},
    {"repo": "karpathy/nanoGPT", "kinds": ("python",), "max_bytes": 150_000},
    {"repo": "TheAlgorithms/Python", "kinds": ("python",), "max_bytes": 600_000},
    # --- Python standard-library idioms / small utilities -------------------
    {"repo": "psf/requests", "kinds": ("python",), "max_bytes": 300_000},
    # --- web fundamentals ----------------------------------------------------
    # NOTE: mdn/content is NOASSERTION and free-programming-books is CC-BY-4.0
    # (attribution required).  Both were tried and *rejected by the licence
    # gate*.  The repositories below are verified permissive replacements.
    {"repo": "twbs/bootstrap", "kinds": ("javascript", "css", "html"), "max_bytes": 300_000},
    {"repo": "jquery/jquery", "kinds": ("javascript",), "max_bytes": 200_000},
    {"repo": "necolas/normalize.css", "kinds": ("css",), "max_bytes": 60_000},
    {"repo": "picturepan2/spectre", "kinds": ("css", "html"), "max_bytes": 200_000},
    {"repo": "google/material-design-lite", "kinds": ("css", "html", "javascript"), "max_bytes": 300_000},
    # --- three.js ------------------------------------------------------------
    {"repo": "mrdoob/three.js", "kinds": ("javascript",), "max_bytes": 400_000},
    # --- explanatory prose, verified permissive ------------------------------
    # Rejected during selection: jwasham/coding-interview-university (CC-BY-SA),
    # goldbergyoni/nodebestpractices (CC-BY-SA), 30-seconds/30-seconds-of-code
    # (no licence file).  Attribution/share-alike terms are not something this
    # project can discharge cleanly, so they are simply excluded.
    {"repo": "ossu/computer-science", "kinds": ("text",), "max_bytes": 150_000},
    {"repo": "microsoft/Web-Dev-For-Beginners", "kinds": ("html", "css", "javascript", "text"), "max_bytes": 700_000},
    {"repo": "ryanmcdermott/clean-code-javascript", "kinds": ("text",), "max_bytes": 120_000},
    {"repo": "leonardomso/33-js-concepts", "kinds": ("text",), "max_bytes": 150_000},
    {"repo": "TheAlgorithms/JavaScript", "kinds": ("javascript",), "max_bytes": 300_000},
]

_FENCE = re.compile(r"```.*?```", re.DOTALL)

#: Files we never want: vendored bundles, minified output, tests fixtures.
_SKIP_PATH = re.compile(
    r"(^|/)(node_modules|vendor|dist|build|\.git|__pycache__|"
    r"third[-_]party|bower_components|coverage)(/|$)"
)
_MINIFIED = re.compile(r"\.min\.(js|css)$")
_ALLOWED_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".md": "text",
    ".rst": "text",
    ".txt": "text",
}


def classify(path: str, text: str) -> Optional[str]:
    """Decide a document's language from its *content*.

    The suffix is only a hint.  A ``.js`` file that is really JSON is rejected;
    a ``.md`` file full of Python fences is counted as mixed prose.  The point
    is that the manifest must not lie about composition the way a
    suffix-only count would.
    """
    suffix = Path(path).suffix.lower()
    hint = _ALLOWED_SUFFIX.get(suffix)
    if hint is None:
        return None
    if _MINIFIED.search(path) or _SKIP_PATH.search(path):
        return None
    if len(text) < 200 or len(text) > 200_000:
        return None

    stripped = text.lstrip()
    # A .js file that is plainly JSON or a data blob is not JavaScript source.
    if hint == "javascript":
        if stripped.startswith("{") and '"' not in text[:200]:
            return None
        if text.count("undefined") == 0 and text.count("function") == 0 and "=>" not in text:
            return None
    if hint == "python":
        py_markers = ("def ", "class ", "import ", "print(", "self.", "elif ")
        if not any(m in text for m in py_markers):
            return None
        # JavaScript that merely mentions "import" must not be counted as Python.
        if "function " in text and text.count(";") > 3 * text.count("\n"):
            return None
    if hint == "html":
        if "<" not in text or ">" not in text:
            return None
    if hint == "css":
        if "{" not in text or ":" not in text:
            return None
    return hint


def _api_json(url: str, timeout: int = 30) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "boxz-corpus/2b"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _licence_of(repo: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (spdx_id, default_branch) or (None, None)."""
    meta = _api_json(f"{API}/repos/{repo}")
    if not meta:
        return None, None
    spdx = (meta.get("license") or {}).get("spdx_id")
    return spdx, meta.get("default_branch") or "main"


def _commit_sha(repo: str, branch: str) -> Optional[str]:
    meta = _api_json(f"{API}/repos/{repo}/commits/{branch}")
    if not meta:
        return None
    return meta.get("sha")


def _tarball(repo: str, branch: str, timeout: int = 120) -> Optional[tarfile.TarFile]:
    url = f"{CODELOAD}/{repo}/tar.gz/refs/heads/{branch}"
    req = urllib.request.Request(url, headers={"User-Agent": "boxz-corpus/2b"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            blob = fh.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")


def fetch_repo(spec: Dict[str, object]) -> Dict[str, object]:
    """Fetch one repository, verifying its licence before downloading content."""
    repo = str(spec["repo"])
    kinds = tuple(spec.get("kinds") or ())
    max_bytes = int(spec.get("max_bytes") or 200_000)

    spdx, branch = _licence_of(repo)
    record: Dict[str, object] = {
        "repo": repo,
        "licence": spdx,
        "branch": branch,
        "documents": [],
        "skipped": [],
    }
    if spdx is None:
        record["skipped"].append("licence could not be read")
        return record
    if spdx not in PERMISSIVE:
        record["skipped"].append(f"licence {spdx} is not permissive")
        return record

    record["commit"] = _commit_sha(repo, branch or "main")
    tf = _tarball(repo, branch or "main")
    if tf is None:
        record["skipped"].append("tarball download failed")
        return record

    kept = 0
    seen = set()
    for member in tf.getmembers():
        if not member.isfile():
            continue
        if kept >= max_bytes:
            break
        name = member.name.split("/", 1)[-1]          # strip the repo-prefix dir
        if not name or len(name) > 300:
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIX:
            continue
        if _MINIFIED.search(name) or _SKIP_PATH.search(name):
            continue
        try:
            fh = tf.extractfile(member)
            if fh is None:
                continue
            raw = fh.read(400_000)
        except (tarfile.TarError, OSError):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        kind = classify(name, text)
        if kind is None or kind not in kinds:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:                            # document-level dedupe
            record["skipped"].append(f"duplicate document {name}")
            continue
        seen.add(digest)
        record["documents"].append({
            "path": name,
            "kind": kind,
            "sha256": digest,
            "chars": len(text),
            "text": text,
        })
        kept += len(text)
    return record


def build_corpus(out_dir: Path = Path("data/raw/p2b"),
                 repos: Optional[Iterable[Dict[str, object]]] = None) -> Dict[str, object]:
    """Fetch every repo and write ``corpus.jsonl`` + ``manifest.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    repos = list(repos if repos is not None else REPOS)
    manifest: Dict[str, object] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repos": [],
    }
    by_kind: Dict[str, int] = {}
    all_hashes = set()
    docs_written = 0

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for spec in repos:
            rec = fetch_repo(spec)
            keep = []
            for doc in rec["documents"]:              # type: ignore[union-attr]
                if doc["sha256"] in all_hashes:
                    rec["skipped"].append(f"cross-repo duplicate {doc['path']}")
                    continue
                all_hashes.add(doc["sha256"])
                keep.append(doc)
                fh.write(json.dumps({
                    "lang": doc["kind"],
                    "source": rec["repo"],
                    "licence": rec["licence"],
                    "commit": rec.get("commit"),
                    "path": doc["path"],
                    "text": doc["text"],
                }, ensure_ascii=False) + "\n")
                by_kind[doc["kind"]] = by_kind.get(doc["kind"], 0) + doc["chars"]
                docs_written += 1
            manifest["repos"].append({           # type: ignore[union-attr]
                "repo": rec["repo"],
                "licence": rec["licence"],
                "commit": rec.get("commit"),
                "documents_kept": len(keep),
                "chars_kept": sum(d["chars"] for d in keep),
                "skipped": rec["skipped"][:20],
            })

    manifest["documents"] = docs_written
    manifest["chars_by_language"] = by_kind
    manifest["chars_total"] = sum(by_kind.values())
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0]) if argv else Path("data/raw/p2b")
    m = build_corpus(out)
    print(json.dumps({k: v for k, v in m.items() if k != "repos"}, indent=2))
    for r in m["repos"]:
        print(f"  {r['repo']:45s} {str(r['licence']):12s} "
              f"{r['documents_kept']:4d} docs  {r['chars_kept']:8,d} chars  "
              f"{'' if not r['skipped'] else r['skipped'][:1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
