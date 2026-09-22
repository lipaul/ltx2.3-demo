#!/usr/bin/env python3
"""FastAPI entry point for single-path LTX-2.5 text-to-video generation.

Same design as ltx_server.py (LTX-2.3) -- FastAPI + SSE + SQLite history + an
HTML UI -- but the job runs one video on a single XPU via run_t2v_25_xpu.py.
The runner builds/streams its own Gemma-4 text encoder, so there is no shared
pre-encode step (pre_encode=False) and no device pairing (device_pairs=False).
"""

import os
from pathlib import Path

from ltx_server_common import ModelProfile, create_server, run_server

HERE = Path(__file__).resolve().parent

PROFILE = ModelProfile(
    display_name="LTX-2.5 Video Generator (1x)",
    default_width=1024,
    default_height=1024,
    default_frames=121,
    multi_mode=1,
    generation_script=str(HERE / "run_t2v_25_xpu.py"),
    pre_encode=False,
    device_pairs=False,
)

server = create_server(PROFILE)
app = server.app

if __name__ == "__main__":
    run_server(server)
