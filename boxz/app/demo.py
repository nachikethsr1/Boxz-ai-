"""A dependency-free demo server for Boxz V3.

`boxz.app.server` is the production-ish server (binds a port, loads a
checkpoint).  This module is the "prove it works" surface: one page, three
panels, and every number it shows comes from a real call into the Math Engine,
the Coding Engine, or the model — nothing is hardcoded.

Endpoints
---------
``GET  /``              the demo page
``GET  /api/health``    model size, ctx, engine availability
``POST /api/chat``      {"message": ...} -> routed reply + tool trace
``GET  /api/math?q=``   Math Engine only
``GET  /api/check?q=``  Coding Engine only

Run::

    python -m boxz.app.demo --checkpoint checkpoints/boxz-26m/boxz-26m/best --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

__all__ = ["make_server", "main"]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Boxz V3</title>
<style>
:root{--bg:#0b0f14;--card:#131a22;--line:#1f2a36;--fg:#e6edf3;--dim:#8b9bab;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
h1{font-size:17px;margin:0;letter-spacing:.5px}
.badge{font-size:11px;padding:2px 8px;border:1px solid var(--line);border-radius:99px;color:var(--dim)}
main{max-width:940px;margin:0 auto;padding:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
h2{font-size:13px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.9px}
input,button,textarea{font:inherit;background:#0e151c;color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:9px 11px}
input{width:100%}
textarea{width:100%;min-height:74px;resize:vertical}
.row{display:flex;gap:8px}
button{background:var(--acc);color:#04101f;border:0;font-weight:700;cursor:pointer;white-space:nowrap}
button:disabled{opacity:.5;cursor:default}
.out{margin-top:12px;white-space:pre-wrap;word-break:break-word}
.mono{font-size:13px}
.hit{color:var(--ok)} .miss{color:var(--bad)} .dim{color:var(--dim)}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:4px 6px;border-top:1px solid var(--line);vertical-align:top}
td:first-child{color:var(--dim);width:150px}
</style></head><body>
<header>
  <h1>BOXZ V3</h1>
  <span class="badge" id="health">loading…</span>
</header>
<main>
  <div class="card">
    <h2>chat — routed automatically</h2>
    <div class="row"><input id="msg" placeholder="482736 + 918264  ·  or: write javascript for a button click" autofocus>
    <button id="send">send</button></div>
    <div class="out mono" id="reply"></div>
    <div id="trace"></div>
  </div>

  <div class="card">
    <h2>math engine — exact, model not involved</h2>
    <div class="row"><input id="mq" placeholder="What is 17% of 2,480?">
    <button id="mbtn">compute</button></div>
    <div class="out mono" id="mout"></div>
  </div>

  <div class="card">
    <h2>coding engine — syntax check, never executes</h2>
    <textarea id="code">function go( { return 1; }</textarea>
    <div class="row" style="margin-top:8px"><button id="cbtn">check syntax</button></div>
    <div class="out mono" id="cout"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }
(async () => {
  const h = await j('/api/health');
  $('#health').textContent = `${h.params.toLocaleString()} params · ctx ${h.context_length} · vocab ${h.vocab_size} · node ${h.node ? 'yes' : 'no'}`;
})();
$('#send').onclick = async () => {
  const message = $('#msg').value.trim(); if(!message) return;
  $('#send').disabled = true; $('#reply').textContent = '…';
  try {
    const r = await j('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
                                   body: JSON.stringify({message})});
    $('#reply').textContent = r.text;
    let rows = `<table><tr><td>routed to</td><td>${r.tools.used_math ? 'MATH ENGINE (model generated 0 tokens)' : (r.tools.used_coding ? 'model + CODING ENGINE' : 'model')}</td></tr>`;
    if(r.tools.math) rows += `<tr><td>exact value</td><td class="hit">${r.tools.math.value}</td></tr><tr><td>expression</td><td>${r.tools.math.expression || '—'}</td></tr>`;
    if(r.tools.code_blocks && r.tools.code_blocks.length)
      rows += r.tools.code_blocks.map(b => `<tr><td>${b.language}</td><td class="${b.status==='passed'?'hit':'miss'}">${b.status}${b.errors&&b.errors.length?': '+b.errors[0].message:''}</td></tr>`).join('');
    rows += `<tr><td>tokens generated</td><td>${r.tokens_generated}</td></tr></table>`;
    $('#trace').innerHTML = rows;
  } finally { $('#send').disabled = false; }
};
$('#msg').onkeydown = e => { if(e.key === 'Enter') $('#send').click(); };
$('#mbtn').onclick = async () => {
  const q = $('#mq').value.trim(); if(!q) return;
  const r = await j('/api/math?q=' + encodeURIComponent(q));
  $('#mout').innerHTML = r.ok ? `<span class="hit">${r.value}</span>  <span class="dim">${r.expression} → ${r.kind}</span>` : `<span class="miss">${r.error}</span>`;
};
$('#mq').onkeydown = e => { if(e.key === 'Enter') $('#mbtn').click(); };
$('#cbtn').onclick = async () => {
  const r = await j('/api/check?q=' + encodeURIComponent($('#code').value));
  $('#cout').innerHTML = r.blocks.map(b =>
    `<div class="${b.status==='passed'?'hit':'miss'}">${b.language}: ${b.status}${b.errors&&b.errors.length?' — '+b.errors[0].message:''}</div>`
  ).join('') || '<span class="dim">no fenced blocks</span>';
};
</script></body></html>"""


class _State:
    engine: Any = None
    tokenizer: Any = None
    model: Any = None
    lock = threading.Lock()
    info: Dict[str, Any] = {}


def _json(handler: BaseHTTPRequestHandler, obj: Any, status: int = 200) -> None:
    body = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # keep the console quiet
            pass

        def _html(self, text: str) -> None:
            b = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:                 # noqa: N802
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._html(PAGE)
            if u.path == "/api/health":
                return _json(self, _State.info)
            if u.path == "/api/math":
                from ..math.engine import solve

                a = solve((qs.get("q") or [""])[0])
                return _json(self, {"ok": a.ok, "value": a.value,
                                    "expression": a.expression, "kind": a.kind,
                                    "error": a.error})
            if u.path == "/api/check":
                from ..coding.engine import validate_reply

                blocks = validate_reply((qs.get("q") or [""])[0])
                return _json(self, {"blocks": [
                    {"language": b.language, "tag": b.tag,
                     "status": b.validation.status,
                     "errors": b.validation.errors,
                     "note": b.validation.note}
                    for b in blocks if b.validation is not None]})
            return _json(self, {"error": "not found"}, 404)

        def do_POST(self) -> None:                # noqa: N802
            u = urlparse(self.path)
            if u.path != "/api/chat":
                return _json(self, {"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return _json(self, {"error": "bad json"}, 400)
            message = str(payload.get("message", "")).strip()
            if not message:
                return _json(self, {"error": "empty message"}, 400)
            from ..config import GenerationConfig

            cfg = GenerationConfig(max_new_tokens=int(payload.get("max_new_tokens", 120)),
                                   temperature=float(payload.get("temperature", 0.4)),
                                   top_k=20, top_p=0.95)
            try:
                with _State.lock:
                    reply = _State.engine.reply(message, cfg)
            except Exception as exc:              # noqa: BLE001
                return _json(self, {"error": f"{type(exc).__name__}: {exc}"}, 500)
            d = reply.to_dict()
            d["text"] = reply.text
            return _json(self, d)

    return Handler


def make_server(checkpoint: Optional[str], port: int, host: str = "0.0.0.0"):
    """Load ``checkpoint`` (if any) and return an unstarted HTTP server."""
    info: Dict[str, Any] = {"params": 0, "context_length": 0, "vocab_size": 0,
                            "node": False, "checkpoint": checkpoint}
    try:
        from ..coding.engine import node_available

        info["node"] = node_available()
    except Exception:                              # noqa: BLE001
        pass

    if checkpoint and Path(checkpoint).exists():
        from ..chat.engine import ChatEngine
        from ..training.checkpoint import load_checkpoint

        ck = load_checkpoint(checkpoint)
        _State.model = ck.model
        _State.tokenizer = ck.tokenizer
        _State.engine = ChatEngine(ck.model, ck.tokenizer, use_tools=True)
        info.update(params=sum(p.data.size for p in ck.model.parameters()),
                    context_length=int(ck.model_config.context_length),
                    vocab_size=int(ck.tokenizer.vocab_size))
    _State.info = info
    return ThreadingHTTPServer((host, port), make_handler())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/boxz-26m/boxz-26m/best")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args(argv)
    srv = make_server(a.checkpoint, a.port, a.host)
    print(f"Boxz V3 demo on http://{a.host}:{a.port}  {_State.info}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
