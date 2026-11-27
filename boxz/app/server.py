"""Boxz web application server.

A dependency-free HTTP server (standard library only) that exposes the real
inference engine over a small JSON API and serves a single-page chat UI.

    python -m boxz.app.server --checkpoint checkpoints/base-1m/best --port 8000

Endpoints
---------
GET  /                     the chat UI
GET  /api/status           model info, backend, training telemetry
GET  /api/history          current conversation
POST /api/chat             {"message": "...", "max_new_tokens": 200, ...}
POST /api/reset            clear the conversation
GET  /api/metrics          training log tail (train_log.jsonl)

There is no simulated model anywhere in this file: every reply comes from
``ChatEngine.reply`` -> ``generate_with_kv_cache`` -> the trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..config import GenerationConfig, load_json
from ..backends import get_backend
from ..chat.engine import ChatEngine
from ..training.checkpoint import load_checkpoint

__all__ = ["BoxzApp", "serve"]

STATIC = Path(__file__).resolve().parent / "static"


class BoxzApp:
    def __init__(self, checkpoint: str, system_prompt: Optional[str] = None,
                 max_context_tokens: Optional[int] = None) -> None:
        t0 = time.time()
        self.ckpt = load_checkpoint(checkpoint)
        self.engine = ChatEngine(self.ckpt.model, self.ckpt.tokenizer,
                                 system_prompt=system_prompt,
                                 max_context_tokens=max_context_tokens)
        self.load_seconds = time.time() - t0
        self.checkpoint = checkpoint
        self.lock = threading.Lock()
        self.start_time = time.time()

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        cfg = self.ckpt.model.cfg
        backend = get_backend("auto")
        return {
            "model": {
                "parameters": self.ckpt.model.num_parameters(),
                "layers": cfg.n_layer,
                "heads": cfg.n_head,
                "d_model": cfg.d_model,
                "context_length": cfg.context_length,
                "vocab_size": cfg.vocab_size,
                "positional": cfg.pos_embeddings,
                "norm": cfg.norm,
                "activation": cfg.activation,
            },
            "checkpoint": self.checkpoint,
            "train_state": {
                "step": self.ckpt.train_state.get("step"),
                "best_val_loss": self.ckpt.train_state.get("best_val_loss"),
                "tokens_seen": self.ckpt.train_state.get("tokens_seen"),
                "stage": self.ckpt.train_state.get("stage", "pretrain"),
            },
            "backend": {"name": backend.name, "version": backend.version},
            "load_seconds": round(self.load_seconds, 2),
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = str(payload.get("message", "")).strip()
        if not message:
            raise ValueError("message must not be empty")
        cfg = GenerationConfig(
            max_new_tokens=int(payload.get("max_new_tokens", 160)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 40)) or None,
            top_p=float(payload.get("top_p", 0.95)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
        )
        with self.lock:
            reply = self.engine.reply(message, cfg)
        return reply.to_dict()

    def history(self) -> List[dict]:
        return self.engine.history()

    def reset(self) -> None:
        with self.lock:
            self.engine.reset()

    def metrics(self, tail: int = 200) -> Dict[str, Any]:
        """Return the tail of the training log next to the checkpoint, if present."""
        log = Path(self.checkpoint).parent / "train_log.jsonl"
        if not log.exists():
            log = Path("checkpoints/base-1m/train_log.jsonl")
        if not log.exists():
            return {"available": False}
        rows: List[dict] = []
        with log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return {"available": True, "path": str(log), "rows": rows[-tail:]}


def _make_handler(app: BoxzApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BoxzApp/1.0"

        # -- helpers ---------------------------------------------------
        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            body = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if path.suffix == ".js":
                ctype = "text/javascript"
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            return

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                if route in ("/", "/index.html"):
                    self._file(STATIC / "index.html")
                elif route == "/api/status":
                    self._json(app.status())
                elif route == "/api/history":
                    self._json({"messages": app.history()})
                elif route == "/api/metrics":
                    self._json(app.metrics())
                elif route.startswith("/static/"):
                    self._file(STATIC / route[len("/static/"):])
                elif route == "/favicon.ico":
                    self._json({}, 204)
                else:
                    self._json({"error": f"unknown route {route}"}, 404)
            except Exception as exc:  # pragma: no cover - defensive
                self._json({"error": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "invalid JSON"}, 400)
                return
            try:
                if route == "/api/chat":
                    self._json(app.chat(payload))
                elif route == "/api/reset":
                    app.reset()
                    self._json({"ok": True})
                else:
                    self._json({"error": f"unknown route {route}"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - defensive
                self._json({"error": str(exc)}, 500)

    return Handler


def make_server(app: BoxzApp, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    return server


def serve(checkpoint: str, host: str = "0.0.0.0", port: int = 8000,
          system_prompt: Optional[str] = None, max_context_tokens: Optional[int] = None) -> None:
    app = BoxzApp(checkpoint, system_prompt=system_prompt, max_context_tokens=max_context_tokens)
    st = app.status()
    print(f"[app] loaded {checkpoint} ({st['model']['parameters']:,} params) "
          f"in {app.load_seconds:.2f}s, backend={st['backend']['name']}")
    server = make_server(app, host, port)
    print(f"[app] serving on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] shutting down")
    finally:
        server.server_close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Boxz chat web application")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--system", default=None)
    p.add_argument("--max-context-tokens", type=int, default=None)
    args = p.parse_args(argv)
    serve(args.checkpoint, args.host, args.port, args.system, args.max_context_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
