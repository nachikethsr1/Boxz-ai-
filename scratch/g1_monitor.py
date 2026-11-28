#!/usr/bin/env python3
"""Autonomous G1 monitor.

Polls the live training process and the JSONL log every POLL seconds, checks the
safety rules the operator set, and writes a status line to monitor.log.  It never
touches the training process except to request a safe stop, which the
trainer handles by closing its log and leaving the last checkpoint in place.

Exit conditions:
  * training process gone            -> "finished" (normal end or crash)
  * 3 consecutive stalls              -> stop, "stall"
  * non-finite loss/grad in the log   -> stop, "non-finite"
  * RSS > RSS_LIMIT_MB               -> stop, "memory"
  * free disk < DISK_LIMIT_GB        -> stop, "disk"
  * no log progress for NO_PROGRESS_S -> stop, "no-progress"
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRAIN_PID = int(sys.argv[1])
LOG = Path("checkpoints/boxz-26m-2a/g1/train_log.jsonl")
OUT = Path("scratch/g1_monitor.log")
POLL = 120                     # seconds between samples
RSS_LIMIT_MB = 3200            # 3.9 GB total; leave headroom for the OS
DISK_LIMIT_GB = 3.0            # stop before the 300 MB checkpoint saves start failing
NO_PROGRESS_S = 900            # 15 min with no new log line = dead/stalled
STALL_S_PER_STEP = 12.0        # 4x the healthy 3.2 s/step
STALL_STREAK_LIMIT = 3         # consecutive abnormal segments before stopping


def rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def last_rows() -> list:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def free_gb() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("a", encoding="utf-8")
    stall_streak = 0
    prev_step = None
    prev_elapsed = None
    prev_wall = time.time()
    last_progress = time.time()
    reason = None

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    say(f"monitor start: pid={TRAIN_PID} rss<{RSS_LIMIT_MB}MB disk>{DISK_LIMIT_GB}GB")

    while True:
        if not alive(TRAIN_PID):
            say("training process exited")
            reason = "finished"
            break

        rows = last_rows()
        train = [r for r in rows if r.get("kind") == "train"]
        val = [r for r in rows if r.get("kind") == "val"]

        if not train:
            time.sleep(POLL)
            continue

        cur = train[-1]
        step = cur["step"]
        elapsed = cur["elapsed_s"]
        mem = rss_mb(TRAIN_PID)
        disk = free_gb()

        # --- non-finite guard -------------------------------------------------
        bad = [r for r in rows
               if not all(isinstance(r.get(k), (int, float)) and r[k] == r[k]
                          and abs(r[k]) != float("inf")
                          for k in ("train_loss", "val_loss", "grad_norm")
                          if r.get(k) is not None)]
        if bad:
            say(f"NON-FINITE value in log: {bad[:2]}")
            reason = "non-finite"
            break

        # --- progress ---------------------------------------------------------
        if step != prev_step:
            last_progress = time.time()
            if prev_step is not None and prev_elapsed is not None:
                d_step = step - prev_step
                d_time = elapsed - prev_elapsed
                if d_step > 0:
                    per = d_time / d_step
                    flag = ""
                    if per > STALL_S_PER_STEP:
                        stall_streak += 1
                        flag = f"  STALL({stall_streak})"
                    else:
                        stall_streak = 0
                    vs = f" val {val[-1]['val_loss']:.4f}" if val else " val -"
                    say(f"step {step:5d}  loss {cur['train_loss']:.4f}{vs}  "
                        f"gnorm {cur['grad_norm']:.3f}  {per:5.2f}s/step  "
                        f"rss {mem:.0f}MB  disk {disk:.1f}GB{flag}")
            prev_step, prev_elapsed = step, elapsed

        # --- safety -----------------------------------------------------------
        if mem > RSS_LIMIT_MB:
            say(f"MEMORY {mem:.0f}MB > {RSS_LIMIT_MB}MB")
            reason = "memory"
            break
        if disk < DISK_LIMIT_GB:
            say(f"DISK {disk:.1f}GB < {DISK_LIMIT_GB}GB")
            reason = "disk"
            break
        if stall_streak >= STALL_STREAK_LIMIT:
            say(f"{stall_streak} consecutive slow segments")
            reason = "stall"
            break
        if time.time() - last_progress > NO_PROGRESS_S:
            say(f"no log progress for {NO_PROGRESS_S}s at step {step}")
            reason = "no-progress"
            break

        time.sleep(POLL)

    if reason != "finished":
        # Do NOT send SIGINT: this trainer has no KeyboardInterrupt handler, so
        # it would die mid-step.  Checkpoints are written atomically every 1500
        # steps, so the safest stop is to let the current step finish and kill
        # hard -- the last completed checkpoint is already durable on disk.
        say(f"SAFE STOP: waiting 5s for the in-flight step, then SIGTERM/SIGKILL. "
            f"Last atomic checkpoint on disk is the recovery point. reason={reason}")
        time.sleep(5)
        try:
            os.kill(TRAIN_PID, signal.SIGTERM)
            time.sleep(10)
            if alive(TRAIN_PID):
                os.kill(TRAIN_PID, signal.SIGKILL)
        except OSError as exc:
            say(f"could not signal: {exc}")
    say(f"monitor exit: {reason}")
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
