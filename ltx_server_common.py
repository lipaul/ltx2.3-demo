"""Shared FastAPI application for LTX-2.3 text-to-video via subprocess worker.

Each generation job spawns run_t2v_xpu.py (or run_t2v_xpu_perf.py) as a
subprocess, avoiding OOM from in-process model lifecycle accumulation.
"""

import asyncio
import json
import logging
import os
import queue
import re
import secrets
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("ltx_server")

LTX23_RUN_DIR = Path(__file__).resolve().parent
LTX23_ENV_PYTHON = os.environ.get("LTX_PYTHON", str(LTX23_RUN_DIR / ".venv" / "bin" / "python"))
GENERATION_SCRIPT = str(LTX23_RUN_DIR / "run_t2v_xpu_perf.py")
MULTI_SCRIPT = str(LTX23_RUN_DIR / "run_multi_xpu.py")
MAX_LOG_LINES = 100

# Gemma text-encoder placement for the shared pre-encode step. Defaults to
# block-streaming on a spare XPU (encoding runs before generation, so every
# device is free); set LTX_GEMMA_DEVICE=cpu to fall back to the CPU path.
GEMMA_DEVICE = os.environ.get("LTX_GEMMA_DEVICE", "xpu:0")
GEMMA_OFFLOAD = os.environ.get("LTX_GEMMA_OFFLOAD", "cpu")

# Persistent encoder service (T3): a long-lived process that keeps the Gemma
# pinned weight source warm across jobs, so the per-job encode skips the
# subprocess import and the pinned-source rebuild. Off by default; enable with
# LTX_ENCODER_SERVICE=1. Falls back to the encode_prompts.py subprocess.
ENCODER_SERVICE = os.environ.get("LTX_ENCODER_SERVICE", "0") == "1"
ENCODER_FP8 = os.environ.get("LTX_ENCODER_FP8", "1") == "1"
ENCODER_SOCK = os.environ.get("LTX_ENCODER_SOCK", str(Path(tempfile.gettempdir()) / "ltx_encoder.sock"))
_ENCODER_PROC: subprocess.Popen | None = None
_ENCODER_LOCK = threading.Lock()


def _gemma_device_is_xpu(spec: str) -> bool:
    return (spec or "").strip().lower().startswith("xpu")


def _encoder_ping() -> bool:
    import socket as _socket

    if not os.path.exists(ENCODER_SOCK):
        return False
    try:
        c = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        c.settimeout(3)
        c.connect(ENCODER_SOCK)
        c.sendall(b'{"ping": true}\n')
        c.recv(4096)
        c.close()
        return True
    except OSError:
        return False


def _ensure_encoder_service() -> bool:
    global _ENCODER_PROC
    if _encoder_ping():
        return True
    Path(ENCODER_SOCK).unlink(missing_ok=True)
    cmd = [LTX23_ENV_PYTHON, "-u", str(LTX23_RUN_DIR / "encode_service.py"),
           "--sock", ENCODER_SOCK, "--device", GEMMA_DEVICE]
    if ENCODER_FP8:
        cmd.append("--fp8")
    logger.info("starting encoder service: %s", " ".join(cmd))
    _ENCODER_PROC = subprocess.Popen(cmd, cwd=str(LTX23_RUN_DIR))
    for _ in range(240):
        if _encoder_ping():
            return True
        if _ENCODER_PROC.poll() is not None:
            logger.error("encoder service exited rc=%s", _ENCODER_PROC.returncode)
            return False
        time.sleep(0.5)
    return False


def _encode_via_service(prompts: list[str], out_dir: str) -> str:
    import socket as _socket

    req = json.dumps({"prompts": prompts, "out_dir": out_dir}) + "\n"
    c = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    c.settimeout(900)
    c.connect(ENCODER_SOCK)
    c.sendall(req.encode())
    data = b""
    while not data.endswith(b"\n"):
        chunk = c.recv(1 << 20)
        if not chunk:
            break
        data += chunk
    c.close()
    resp = json.loads(data.decode())
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "encoder service error"))
    return resp["out_dir"]


@dataclass(frozen=True)
class ModelProfile:
    display_name: str
    default_width: int
    default_height: int
    default_frames: int
    multi_mode: int = 8  # videos per job: 8/16 for LTX-2.3, 1 for single-path 2.5
    # Runner selection. LTX-2.3 pre-encodes prompts (shared Gemma) and pairs the
    # 32 XPUs; single-path LTX-2.5 encodes internally (streamed Gemma-4) on one XPU.
    generation_script: str = GENERATION_SCRIPT
    pre_encode: bool = True
    device_pairs: bool = True
    # Retry a job this many times if a worker dies (e.g. a transient XPU driver
    # segfault during model load). 0 disables retries (LTX-2.3 default).
    retries: int = 0


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    api_token: str
    queue_size: int
    output_dir: Path
    database_path: Path

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        return cls(
            host=os.environ.get("LTX_HOST", "127.0.0.1"),
            port=int(os.environ.get("LTX_PORT", "8001")),
            api_token=os.environ.get("LTX_API_TOKEN", ""),
            queue_size=int(os.environ.get("LTX_QUEUE_SIZE", "2")),
            output_dir=Path(os.environ.get("LTX_OUTPUT_DIR", "outputs/ltx-server")).resolve(),
            database_path=Path(
                os.environ.get("LTX_DB", "outputs/ltx-server/jobs.sqlite3")
            ).resolve(),
        )


@dataclass(frozen=True)
class ServerApplication:
    app: FastAPI
    settings: ServerSettings
    profile: ModelProfile


def is_loopback(host: str) -> bool:
    return host.lower() in {"127.0.0.1", "localhost", "::1"}


def generate_seed() -> int:
    return (time.time_ns() ^ secrets.randbits(63) ^ uuid4().int) & (2**63 - 1)


# ---------------------------------------------------------------------------
# JobStore  (SQLite)
# ---------------------------------------------------------------------------

class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    prompts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    output_dir TEXT
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    def create_multi(self, job_id: str, prompts: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO multi_jobs (id, status, prompts, created_at) VALUES (?, ?, ?, ?)",
                (job_id, "queued", json.dumps(prompts), now),
            )
            conn.commit()

    def get_multi(self, job_id: str) -> dict | None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM multi_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            r = dict(row)
            r["prompts"] = json.loads(r["prompts"])
            return r

    def list_multi(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM multi_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["prompts"] = json.loads(r["prompts"])
                results.append(r)
            return results

    def mark_multi_running(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (now, job_id),
            )
            conn.commit()

    def mark_multi_succeeded(self, job_id: str, output_dir: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'succeeded', completed_at = ?, output_dir = ? WHERE id = ?",
                (now, output_dir, job_id),
            )
            conn.commit()

    def mark_multi_failed(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE multi_jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                (now, error, job_id),
            )
            conn.commit()




# ---------------------------------------------------------------------------
# ServerState  (thread-safe shared state, broadcast via SSE)
# ---------------------------------------------------------------------------

class ServerState:
    """Holds all active job state server-side. Updated by worker threads,
    read by the SSE endpoint. Clients are pure viewers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._multi_log: deque[str] = deque(maxlen=MAX_LOG_LINES)

        # active multi job
        self.active_multi_id: str | None = None
        self.active_multi_status: str = "idle"
        self.active_multi_prompts: list[str] = []
        self.active_multi_workers: list[dict] = []  # [{idx, status, last_log}]
        self.active_multi_videos: list[str] = []
        self.active_multi_error: str = ""

        # cached history (refreshed from DB periodically)
        self.multi_history: list[dict] = []

    # -- multi job state --

    def multi_start(self, job_id: str, prompts: list[str]) -> None:
        with self._lock:
            self.active_multi_id = job_id
            self.active_multi_status = "running"
            self.active_multi_prompts = list(prompts)
            self.active_multi_workers = [
                {"idx": i, "status": "queued", "last_log": ""} for i in range(len(prompts))
            ]
            self.active_multi_videos = []
            self.active_multi_error = ""
            self._multi_log.clear()

    def multi_append_log(self, line: str) -> None:
        with self._lock:
            self._multi_log.append(str(line))

    def multi_update_worker(self, idx: int, status: str, last_log: str = "") -> None:
        with self._lock:
            if 0 <= idx < len(self.active_multi_workers):
                self.active_multi_workers[idx]["status"] = status
                if last_log:
                    self.active_multi_workers[idx]["last_log"] = last_log

    def multi_succeeded(self, videos: list[str]) -> None:
        with self._lock:
            self.active_multi_status = "succeeded"
            self.active_multi_videos = list(videos)

    def multi_failed(self, error: str) -> None:
        with self._lock:
            self.active_multi_status = "failed"
            self.active_multi_error = error

    def multi_clear(self) -> None:
        with self._lock:
            self.active_multi_id = None
            self.active_multi_status = "idle"
            self.active_multi_prompts = []
            self.active_multi_workers = []
            self.active_multi_videos = []
            self.active_multi_error = ""
            self._multi_log.clear()

    # -- snapshot for SSE broadcast --

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "multi": {
                    "id": self.active_multi_id,
                    "status": self.active_multi_status,
                    "prompts": list(self.active_multi_prompts),
                    "log": list(self._multi_log),
                    "workers": list(self.active_multi_workers),
                    "videos": list(self.active_multi_videos),
                    "error": self.active_multi_error,
                },
                "history": {
                    "multi": self.multi_history,
                },
            }


# ---------------------------------------------------------------------------
# MultiLtxWorker  (single background thread, subprocess-based)
# ---------------------------------------------------------------------------

class MultiLtxWorker:
    """Background worker that directly spawns encode_prompts.py + N generation workers."""

    def __init__(self, store: JobStore, output_dir: Path, state: ServerState,
                 max_workers: int = 8, *,
                 generation_script: str = GENERATION_SCRIPT,
                 pre_encode: bool = True,
                 device_pairs: bool = True,
                 retries: int = 0) -> None:
        self._store = store
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state = state
        self._max_workers = max_workers
        self._generation_script = generation_script
        self._pre_encode = pre_encode
        self._device_pairs = device_pairs
        self._retries = retries
        self._stagger = 1
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        self._ready.set()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ltx-multi-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

    def submit(self, task: dict) -> None:
        self._queue.put_nowait(task)

    def _spawn_and_wait(self, job_dir: str, prompts: list[str], n: int,
                        embeddings_dir: str | None) -> list[dict]:
        """Spawn the N generation workers and wait, returning per-worker results."""
        # Step 2: spawn generation workers with stagger
        self._state.multi_append_log(f"[step 2] Spawning {n} workers...")
        logger.info("Step 2/%d: spawning %d workers (staggered)", n + 1, n)
        processes: list[dict] = []
        for i in range(n):
            # device assignment: pairs (0,1), (2,3), … (30,31); single
            # runners use one XPU (the worker sets its own LTX_TDEV).
            tdev = i * 2
            cdev = i * 2 + 1
            dev_label = f"xpu:({tdev},{cdev})" if self._device_pairs else "xpu:0"
            output_path = os.path.join(job_dir, f"video_{i}.mp4")
            log_path = os.path.join(job_dir, f"video_{i}.log")

            env = os.environ.copy()
            if self._device_pairs:
                env.update({
                    "LTX_TDEV": str(tdev),
                    "LTX_CDEV": str(cdev),
                    "LTX_PROMPT": prompts[i],
                    "LTX_OUTPUT_PATH": output_path,
                    "LTX_EMBEDDINGS_PATH": os.path.join(embeddings_dir, f"embeddings_{i}.pt"),
                    "LTX_GEMMA_DEVICE": "cpu",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                })
            else:
                env.update({
                    "LTX_PROMPT": prompts[i],
                    "LTX_OUTPUT_PATH": output_path,
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                })
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                [LTX23_ENV_PYTHON, "-u", self._generation_script],
                cwd=str(LTX23_RUN_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append({
                "idx": i, "proc": proc, "log_file": log_file,
                "output_path": output_path, "log_path": log_path,
            })
            self._state.multi_update_worker(i, "spawned", f"pid={proc.pid} {dev_label}")
            self._state.multi_append_log(f"  worker {i+1}/{n}  pid={proc.pid}  {dev_label}")
            logger.info("  worker %d/%d  pid=%d  %s", i + 1, n, proc.pid, dev_label)

            # stagger between spawns to avoid XPU driver race
            if i < n - 1:
                time.sleep(self._stagger)

        # Step 3: wait for all workers with periodic log tailing
        self._state.multi_append_log("[step 3] Waiting for all workers...")
        logger.info("Waiting for %d workers...", n)
        remaining = list(processes)
        results = []
        while remaining:
            for info in list(remaining):
                rc = info["proc"].poll()
                if rc is not None:
                    remaining.remove(info)
                    info["log_file"].close()
                    exists = os.path.isfile(info["output_path"])
                    size = os.path.getsize(info["output_path"]) if exists else 0
                    status = "OK" if rc == 0 and exists and size > 0 else "FAIL"
                    results.append({
                        "idx": info["idx"], "rc": rc, "exists": exists,
                        "size": size, "status": status,
                        "output_path": info["output_path"],
                    })
                    self._state.multi_update_worker(info["idx"], "done" if status == "OK" else "failed")
                    self._state.multi_append_log(
                        f"  worker {info['idx']+1}/{n} done  {'OK' if status == 'OK' else f'rc={rc}'}  "
                        f"{info['output_path']} ({size/1024**2:.1f} MB)" if exists else "no file"
                    )
                    logger.info("  worker %d/%d done  rc=%d  %s  %s  (%s)",
                                info["idx"] + 1, n, rc,
                                "OK" if status == "OK" else f"rc={rc}",
                                info["output_path"],
                                f"{size / 1024**2:.1f} MB" if exists else "no file")
            # read tail of each running worker's log
            if remaining:
                for info in remaining:
                    try:
                        with open(info["log_path"]) as lf:
                            lines = lf.read().strip().splitlines()
                            last = lines[-1] if lines else ""
                            if last:
                                self._state.multi_update_worker(info["idx"], "running", last[-120:])
                    except OSError:
                        pass
                time.sleep(2)
        return results

    def _run(self) -> None:
        logger.info("Multi-worker thread started")
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            job_id = task["id"]
            prompts = task["prompts"]
            n = len(prompts)
            logger.info("Processing multi-job %s (%d prompts)", job_id, n)
            self._store.mark_multi_running(job_id)
            self._state.multi_start(job_id, prompts)
            self._state.multi_append_log(f"[multi] {n} prompts, job {job_id[:8]}...")

            job_dir = str(self._output_dir / job_id)
            os.makedirs(job_dir, exist_ok=True)

            prompts_file = os.path.join(job_dir, "prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(prompts, f)

            try:
                # Step 1: encode prompts (shared Gemma). Skipped for single-path
                # runners (LTX-2.5) that build/stream their own text encoder.
                embeddings_dir = None
                if self._pre_encode:
                    self._state.multi_append_log(
                        f"[step 1] Encoding prompts via Gemma ({GEMMA_DEVICE}, offload={GEMMA_OFFLOAD})..."
                    )
                    encode_t0 = time.perf_counter()
                    if ENCODER_SERVICE and _gemma_device_is_xpu(GEMMA_DEVICE):
                        logger.info("Step 1/%d: encoding %d prompts via encoder service", n + 1, n)
                        try:
                            if _ensure_encoder_service():
                                embeddings_dir = _encode_via_service(
                                    prompts, os.path.join(job_dir, "embeddings")
                                )
                                self._state.multi_append_log(
                                    f"[step 1] Done (service), embeddings in {embeddings_dir}"
                                )
                            else:
                                logger.warning("encoder service unavailable; falling back to subprocess")
                        except Exception:
                            logger.exception("encoder service failed; falling back to subprocess")
                            embeddings_dir = None
                    if embeddings_dir is None:
                        logger.info("Step 1/%d: encoding %d prompts via encode_prompts.py", n + 1, n)
                        env = os.environ.copy()
                        env.update({
                            "LTX_PROMPTS_FILE": prompts_file,
                            "LTX_GEMMA_DEVICE": GEMMA_DEVICE,
                            "LTX_GEMMA_OFFLOAD": GEMMA_OFFLOAD,
                            "HF_HUB_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        })
                        enc_result = subprocess.run(
                            [LTX23_ENV_PYTHON, "-u", str(LTX23_RUN_DIR / "encode_prompts.py")],
                            capture_output=True, text=True, timeout=600,
                            env=env, cwd=str(LTX23_RUN_DIR),
                        )
                        if enc_result.returncode != 0:
                            raise RuntimeError(
                                f"encode_prompts.py failed: {(enc_result.stderr or '')[-500:]}"
                            )
                        embeddings_dir = (enc_result.stdout or "").strip().splitlines()[-1]
                        self._state.multi_append_log(f"[step 1] Done, embeddings in {embeddings_dir}")
                    logger.info("Embeddings dir: %s", embeddings_dir)
                    logger.info("Step 1 done in %.1f s", time.perf_counter() - encode_t0)

                # Step 2+3: spawn + wait, retrying on transient worker crashes
                # (e.g. an XPU driver segfault during model load).
                attempts = self._retries + 1
                results: list[dict] = []
                ok = 0
                for attempt in range(attempts):
                    if attempt:
                        self._state.multi_append_log(
                            f"[retry] {ok}/{n} ok; retrying (attempt {attempt + 1}/{attempts})"
                        )
                        logger.warning("job %s: retry %d/%d", job_id, attempt + 1, attempts)
                    results = self._spawn_and_wait(job_dir, prompts, n, embeddings_dir)
                    ok = sum(1 for r in results if r["status"] == "OK")
                    if ok == n:
                        break

                with open(os.path.join(job_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)

                if ok == n:
                    self._store.mark_multi_succeeded(job_id, job_dir)
                    videos = [f"/api/multi-jobs/{job_id}/videos/{i}" for i in range(n)]
                    self._state.multi_succeeded(videos)
                    self._state.multi_append_log(f"[done] {ok}/{n} workers succeeded")
                    logger.info("Multi-job %s: %d/%d succeeded", job_id, ok, n)
                else:
                    rc_list = [r["rc"] for r in results if r["status"] != "OK"]
                    raise RuntimeError(f"{ok}/{n} workers succeeded (worker rc: {rc_list})")
            except Exception as e:
                logger.exception("Multi-job %s failed", job_id)
                self._store.mark_multi_failed(job_id, str(e))
                self._state.multi_failed(str(e))
                self._state.multi_append_log(f"[error] {str(e)[:200]}")


# Multi-job public response helpers

def public_multi_job(job: dict, output_dir: str | None = None) -> dict:
    result = dict(job)
    result.pop("output_dir", None)
    if job["status"] == "succeeded":
        result["videos"] = [
            f"/api/multi-jobs/{job['id']}/videos/{i}" for i in range(len(job["prompts"]))
        ]
    else:
        result["videos"] = []
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__MODEL_DISPLAY_NAME__</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0c111b; color: #e8edf5; }
    main { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
    h1 { margin: 0 0 8px; } .muted { color: #94a3b8; }
    .card { background: #151d2b; border: 1px solid #263349; border-radius: 14px; padding: 18px; margin-top: 18px; }
    label { display: block; margin: 10px 0 5px; color: #b9c6d8; }
    textarea, input, select { box-sizing: border-box; width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #34445f; background: #0e1623; color: white; }
    textarea { min-height: 100px; resize: vertical; font-family: inherit; }
    button { margin-top: 16px; border: 0; border-radius: 9px; padding: 11px 18px; background: #4f7cff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .job { padding: 10px 0; border-bottom: 1px solid #29364a; }
    .error { color: #ff8e8e; white-space: pre-wrap; }
    .note { font-size: 13px; color: #94a3b8; margin-top: 4px; }
    .multi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .multi-prompts > div { margin-bottom: 8px; }
    .multi-prompts textarea { min-height: 60px; }
    .multi-gallery { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .multi-video-card { min-width: 0; padding: 8px; background: #0e1623; border: 1px solid #29364a; border-radius: 8px; }
    .multi-video-card video { display: block; width: 100%; border-radius: 6px; }
    .multi-video-card .idx { color: #94a3b8; font-size: 11px; margin-bottom: 4px; }
    .log-box { background: #0a0f18; border: 1px solid #29364a; border-radius: 8px; padding: 10px; margin-top: 8px; max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; line-height: 1.5; }
    .log-box .line { color: #94a3b8; }
    .log-box .line.info { color: #b9d0ff; }
    .log-box .line.error { color: #ff8e8e; }
    .log-box .line.done { color: #6fcf97; }
    .multi-worker-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
    .multi-worker-item { background: #0e1623; border: 1px solid #29364a; border-radius: 6px; padding: 6px 10px; font-size: 12px; }
    .multi-worker-item .w-name { color: #94a3b8; }
    .multi-worker-item .w-status { font-weight: 700; }
    .multi-worker-item .w-status.running { color: #f1c40f; }
    .multi-worker-item .w-status.done { color: #6fcf97; }
    .multi-worker-item .w-status.failed { color: #ff8e8e; }
    .multi-worker-item .w-status.spawned { color: #b9d0ff; }
    .multi-worker-item .w-status.queued { color: #94a3b8; }
    .multi-worker-item .w-log { color: #6c7a91; font-family: monospace; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; }
    .status-badge.running { background: #f1c40f22; color: #f1c40f; border: 1px solid #f1c40f44; }
    .status-badge.succeeded { background: #6fcf9722; color: #6fcf97; border: 1px solid #6fcf9744; }
    .status-badge.failed { background: #ff8e8e22; color: #ff8e8e; border: 1px solid #ff8e8e44; }
    .status-badge.queued { background: #94a3b822; color: #94a3b8; border: 1px solid #94a3b844; }
    .status-badge.idle { background: #4f7cff22; color: #4f7cff; border: 1px solid #4f7cff44; }
    .progress-bar { background: #29364a; border-radius: 6px; height: 6px; margin-top: 8px; overflow: hidden; }
    .progress-bar .fill { height: 100%; border-radius: 6px; background: #4f7cff; transition: width 0.5s; }
    @media (max-width: 700px) { .multi-gallery { grid-template-columns: 1fr 1fr; } .multi-grid { grid-template-columns: 1fr; } .multi-worker-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body><main>
  <h1>__MODEL_DISPLAY_NAME__</h1>
  <div class="muted">All state is server-side. Every connected client sees the same view. Multi mode: __MULTI_MODE__×.</div>

  <section class="card">
    <label>API Token</label><input id="token" type="password" autocomplete="off" placeholder="Required for submitting jobs">
  </section>

  <h2>Multi Video Generation (__MULTI_MODE__×)</h2>
  <section class="card">
    <div class="note" style="margin-bottom:10px">__MULTI_MODE__ concurrent 1024×1024, 121-frame videos.</div>
    __MULTI_PROMPTS_HTML__
    <button id="multiSubmit">Generate __MULTI_MODE__ Videos</button>
    <div id="multiStatus" class="note" style="margin-top:12px"></div>
    <div id="multiWorkerPanel" class="multi-worker-grid" style="margin-top:8px"></div>
    <div id="multiLog" class="log-box" style="display:none"></div>
    <div id="multiError" class="error"></div>
    <div id="multiGallery" class="multi-gallery"></div>
  </section>

  <section class="card">
    <h2 style="margin:0 0 8px">Server Log <span class="note">(shared, last __MAX_LOG_LINES__ lines)</span></h2>
    <div id="serverLog" class="log-box"></div>
  </section>

  <section class="card"><h2>Recent Multi-Jobs</h2><div id="multiHistory"></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
const STATE_LABELS = {idle:'Idle',queued:'Queued',running:'Running',succeeded:'Completed',failed:'Failed'};
const DEFAULT_PROMPTS = [
  'A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, photorealistic, 4k.',
  'A majestic eagle soaring over a deep canyon at golden hour, warm sunlight, cinematic, 8k.',
  'An underwater scene with a sea turtle swimming through a coral reef, volumetric lighting, 4k.',
  'A cyberpunk city at night with neon signs reflecting on wet streets, blade runner aesthetic, 8k.',
  'A serene mountain lake at sunrise with mist rising from the water, photorealistic, warm golden light.',
  'A macro shot of a dragonfly perched on a dewy leaf, morning light, shallow depth of field, 4k.',
  'A medieval castle on a stormy cliff edge, lightning flashing, dramatic clouds, epic scale.',
  'A futuristic greenhouse on Mars under a transparent dome, lush exotic plants, sci-fi, 8k.',
  'A steaming cup of coffee on a wooden table at sunrise, cinematic, warm tones, photorealistic, 4k.',
  'A wizard casting a spell in an ancient library, floating books, mystical blue light, cinematic.',
  'A neon-lit sushi bar in Tokyo at midnight, rain on window, reflections, cyberpunk aesthetic.',
  'A polar bear on a melting ice floe at sunset, dramatic sky, climate change, photorealistic.',
  'A race car speeding through a futuristic city tunnel, motion blur, neon reflections, 8k.',
  'A ballerina performing on an empty stage, spotlight, dust particles, emotional, cinematic.',
  'A supercell thunderstorm over a prairie at twilight, lightning, rotating clouds, epic scale.',
  'An astronaut floating in space overlooking Earth, stars, cosmic rays, photorealistic, 8k.'
];
const ST = sessionStorage;
$('token').value = ST.getItem('ltx_token') || '';
function headers(json) {
  const t = $('token').value.trim(); ST.setItem('ltx_token', t);
  const h = {}; if (t) h.Authorization = 'Bearer ' + t; if (json) h['Content-Type'] = 'application/json'; return h;
}
async function api(path, opts) {
  const r = await fetch(path, opts || {});
  if (!r.ok) { let d; try { d = (await r.json()).detail; } catch { d = r.statusText; } throw new Error(typeof d === 'string' ? d : JSON.stringify(d)); }
  return r;
}

// fill default prompts
document.querySelectorAll('.mp').forEach(ta => { ta.value = DEFAULT_PROMPTS[+ta.dataset.idx]; });

// ---- SSE: all state comes from server ----
let evtSource = null;
function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/events');
  evtSource.onmessage = function(event) {
    try { renderAll(JSON.parse(event.data)); } catch(e) { console.error(e); }
  };
  evtSource.onerror = function() {
    evtSource.close();
    setTimeout(connectSSE, 3000);
  };
}

function renderAll(state) {
  renderMulti(state.multi);
  renderHistory(state.history);
  renderServerLog(state);
}

// ---- multi job ----
function renderMulti(m) {
  const statusEl = $('multiStatus');
  const logEl = $('multiLog');
  const workerPanel = $('multiWorkerPanel');
  const gallery = $('multiGallery');
  const errorEl = $('multiError');

  if (m.status === 'idle') {
    statusEl.textContent = '';
    logEl.style.display = 'none';
    workerPanel.replaceChildren();
    gallery.replaceChildren();
    errorEl.textContent = '';
    return;
  }

  let html = `<span class="status-badge ${m.status}">${STATE_LABELS[m.status] || m.status}</span>`;
  if (m.prompts) html += ` <span style="color:#94a3b8;font-size:13px">${m.prompts.length} prompts</span>`;
  if (m.videos && m.videos.length) {
    html += ' ';
    m.videos.forEach((url, i) => { html += `<a href="${url}" target="_blank" style="color:#4f7cff;margin-right:4px">[vid${i}]</a>`; });
  }
  statusEl.innerHTML = html;

  if (m.error) errorEl.textContent = m.error; else errorEl.textContent = '';

  // worker grid - only update text/classes, not DOM (avoid flicker)
  if (m.workers && m.workers.length) {
    while (workerPanel.children.length < m.workers.length) {
      const div = document.createElement('div'); div.className = 'multi-worker-item';
      div.innerHTML = '<span class="w-name"></span> <span class="w-status"></span><div class="w-log"></div>';
      workerPanel.appendChild(div);
    }
    while (workerPanel.children.length > m.workers.length) {
      workerPanel.lastChild.remove();
    }
    m.workers.forEach((w, i) => {
      const div = workerPanel.children[i];
      div.querySelector('.w-name').textContent = 'Worker ' + (w.idx+1);
      const ws = div.querySelector('.w-status');
      ws.textContent = w.status;
      ws.className = 'w-status ' + w.status;
      div.querySelector('.w-log').textContent = w.last_log || '';
    });
  } else {
    workerPanel.replaceChildren();
  }

  // log
  if (m.log && m.log.length) {
    logEl.style.display = '';
    logEl.replaceChildren();
    m.log.forEach(line => {
      const d = document.createElement('div'); d.className = 'line';
      if (line.startsWith('[error]')) d.classList.add('error');
      else if (line.startsWith('[done]')) d.classList.add('done');
      else if (line.startsWith('[')) d.classList.add('info');
      d.textContent = line;
      logEl.appendChild(d);
    });
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    logEl.style.display = 'none';
  }

  // gallery - only rebuild when video URLs change
  const newVideos = m.status === 'succeeded' && m.videos ? m.videos.join(',') : '';
  if (newVideos && newVideos !== gallery.dataset.videosKey) {
    gallery.replaceChildren();
    gallery.dataset.videosKey = newVideos;
    (m.prompts || []).forEach((p, i) => {
      const card = document.createElement('div'); card.className = 'multi-video-card';
      const label = document.createElement('div'); label.className = 'idx';
      label.textContent = 'Video ' + (i+1) + ': ' + p.slice(0, 60);
      card.appendChild(label);
      const video = document.createElement('video');
      video.src = m.videos[i]; video.controls = true; video.preload = 'metadata';
      video.style.maxHeight = '240px'; video.style.width = '100%';
      card.appendChild(video);
      gallery.appendChild(card);
    });
  } else if (!newVideos) {
    gallery.replaceChildren();
    delete gallery.dataset.videosKey;
  }
}

// ---- history ----
function renderHistory(h) {
  // multi history
  const multiRoot = $('multiHistory');
  multiRoot.replaceChildren();
  if (h && h.multi) {
    h.multi.forEach(job => {
      const row = document.createElement('div'); row.className = 'job';
      const first = (job.prompts || [])[0] || '';
      const label = document.createElement('span');
      label.textContent = (job.created_at || '') + ' \u00b7 ' + STATE_LABELS[job.status] + ' \u00b7 ' + first.slice(0, 100);
      row.appendChild(label);
      if (job.status === 'succeeded' && job.videos) {
        job.videos.forEach((url, i) => {
          const a = document.createElement('a'); a.href = url; a.target = '_blank';
          a.textContent = ' [vid' + i + ']'; a.style.color = '#4f7cff'; a.style.marginRight = '4px';
          row.appendChild(a);
        });
      }
      multiRoot.appendChild(row);
    });
  }
}

// ---- server log (combines single + multi logs) ----
function renderServerLog(state) {
  const el = $('serverLog');
  const lines = [];
  if (state.multi && state.multi.log) state.multi.log.forEach(l => lines.push(l));
  if (!lines.length) { el.style.display = 'none'; return; }
  el.style.display = '';
  el.replaceChildren();
  lines.slice(-40).forEach(line => {
    const d = document.createElement('div'); d.className = 'line';
    if (line.includes('error') || line.includes('fail')) d.classList.add('error');
    d.textContent = line;
    el.appendChild(d);
  });
  el.scrollTop = el.scrollHeight;
}

async function submitMulti() {
  $('multiSubmit').disabled = true;
  const prompts = [];
  document.querySelectorAll('.mp').forEach(ta => { const v = ta.value.trim(); prompts[+ta.dataset.idx] = v || DEFAULT_PROMPTS[+ta.dataset.idx]; });
  try {
    await api('/api/multi-jobs', { method: 'POST', headers: headers(true), body: JSON.stringify({prompts: prompts}) });
  } catch(e) { $('multiError').textContent = e.message; }
  finally { $('multiSubmit').disabled = false; }
}

$('multiSubmit').onclick = submitMulti;

connectSSE();
</script></body></html>"""



def _generate_prompts_html(n: int) -> str:
    """Generate the prompt textarea grid for *n* prompts (1, 8 or 16)."""
    cols = 1 if n <= 1 else (2 if n <= 8 else 4)
    rows_per = -(-n // cols)  # ceil
    parts = ['<div class="multi-grid">']
    for c in range(cols):
        parts.append('<div class="multi-prompts">')
        for r in range(rows_per):
            i = c * rows_per + r
            if i >= n:
                break
            parts.append(
                f'<div><label>Prompt {i+1}</label>'
                f'<textarea class="mp" data-idx="{i}"></textarea></div>'
            )
        parts.append('</div>')
    parts.append('</div>')
    return "\n".join(parts)


def render_html(profile: ModelProfile) -> str:
    prompts_html = _generate_prompts_html(profile.multi_mode)
    return (
        HTML_TEMPLATE.replace("__MODEL_DISPLAY_NAME__", profile.display_name)
        .replace("__DEFAULT_WIDTH__", str(profile.default_width))
        .replace("__DEFAULT_HEIGHT__", str(profile.default_height))
        .replace("__DEFAULT_FRAMES__", str(profile.default_frames))
        .replace("__MAX_LOG_LINES__", str(MAX_LOG_LINES))
        .replace("__MULTI_PROMPTS_HTML__", prompts_html)
        .replace("__MULTI_MODE__", str(profile.multi_mode))
    )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(profile: ModelProfile) -> ServerApplication:
    settings = ServerSettings.from_environment()
    store = JobStore(settings.database_path)
    state = ServerState()
    multi_worker: MultiLtxWorker | None = None
    _history_timer: threading.Event | None = None

    def refresh_history() -> None:
        """Load recent history from DB into ServerState (runs periodically)."""
        try:
            state.multi_history = [
                public_multi_job(j) for j in store.list_multi(limit=10, offset=0)
            ]
        except Exception as e:
            logger.warning("refresh_history error: %s", e)

    class MultiJobRequest(BaseModel):
        prompts: list[str] = Field(min_length=1, max_length=profile.multi_mode)

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not settings.api_token:
            return
        scheme, _, value = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, settings.api_token):
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal multi_worker, _history_timer
        if not is_loopback(settings.host) and not settings.api_token:
            raise RuntimeError(
                "LTX_API_TOKEN is required when LTX_HOST is not a loopback address."
            )
        store.initialize()
        refresh_history()
        multi_worker = MultiLtxWorker(
            store=store,
            output_dir=settings.output_dir,
            state=state,
            max_workers=profile.multi_mode,
            generation_script=profile.generation_script,
            pre_encode=profile.pre_encode,
            device_pairs=profile.device_pairs,
            retries=profile.retries,
        )
        multi_worker.start()
        app.state.multi_worker = multi_worker
        # background history refresher every 5s
        _stop = threading.Event()
        _history_timer = _stop

        def _history_loop():
            while not _stop.wait(5):
                refresh_history()

        threading.Thread(target=_history_loop, daemon=True).start()
        try:
            yield
        finally:
            _stop.set()
            multi_worker.stop()

    app = FastAPI(
        title=f"{profile.display_name} Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.profile = profile
    app.state.settings = settings
    app.state.store = store
    html_page = render_html(profile)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html_page

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/events")
    async def event_stream():
        """SSE endpoint — broadcasts server state to all connected clients."""
        async def _generate():
            last_json = ""
            while True:
                snap = state.snapshot()
                cur = json.dumps(snap, default=str)
                # only send if state changed
                if cur != last_json:
                    yield f"data: {cur}\n\n"
                    last_json = cur
                await asyncio.sleep(0.8)
        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.post("/api/multi-jobs", status_code=202, dependencies=[Depends(require_token)])
    def create_multi_job(request: MultiJobRequest):
        mw = app.state.multi_worker
        if not mw.ready:
            raise HTTPException(status_code=503, detail="Multi-worker is not ready.")
        prompts = [p.strip() for p in request.prompts if p.strip()]
        if not prompts:
            raise HTTPException(status_code=422, detail="At least one non-blank prompt required.")
        if len(prompts) > profile.multi_mode:
            raise HTTPException(status_code=422, detail=f"Maximum {profile.multi_mode} prompts.")
        job_id = uuid4().hex
        store.create_multi(job_id, prompts)
        try:
            mw.submit({"id": job_id, "prompts": prompts})
        except queue.Full:
            store.mark_multi_failed(job_id, "Multi-job queue is full")
            raise HTTPException(status_code=429, detail="Multi-job queue is full.")
        return public_multi_job(store.get_multi(job_id))

    @app.get("/api/multi-jobs", dependencies=[Depends(require_token)])
    def list_multi_jobs():
        return [public_multi_job(j) for j in store.list_multi(limit=20, offset=0)]

    @app.get("/api/multi-jobs/{job_id}", dependencies=[Depends(require_token)])
    def get_multi_job(job_id: str):
        job = store.get_multi(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Multi-job not found")
        return public_multi_job(job)

    @app.get("/api/multi-jobs/{job_id}/videos/{index:int}")
    def get_multi_video(job_id: str, index: int):
        job = store.get_multi(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Multi-job not found")
        if job["status"] != "succeeded" or not job["output_dir"]:
            raise HTTPException(status_code=409, detail="Videos not available")
        if index < 0 or index >= len(job["prompts"]):
            raise HTTPException(status_code=422, detail=f"Index out of range: 0..{len(job['prompts'])-1}")
        video_path = Path(job["output_dir"]) / f"video_{index}.mp4"
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(video_path, media_type="video/mp4",
                            headers={"Content-Disposition": "inline"})

    return ServerApplication(app=app, settings=settings, profile=profile)


def run_server(server: ServerApplication) -> None:
    import uvicorn
    uvicorn.run(server.app, host=server.settings.host, port=server.settings.port, workers=1)
