"""Measure XPU memory bandwidth (HBM read/write/copy + host->device) on the B70.

Run:
    .venv/bin/python bench_xpu_bw.py [--size-gib 1.0] [--iters 30]

Why this exists: during the LTX profiling session, ``intel_gpu_top`` could not
read the B70 (its PMU is i915-only; the B70 uses the ``xe`` driver) and
``xpu-smi`` only enumerated the integrated GPU, not the discrete B70. This
script measures the same thing directly with torch, and is the working method
on this host.

Interpretation: the read-only ``reduce`` number is the cleanest HBM read
estimate and sits near the B70's 608 GB/s spec; ``D2D copy`` / ``mul`` / ``add``
count read+write traffic and can exceed the DRAM spec thanks to caching. The
``H2D copy`` number is the effective host->device link (the sysfs link-width
reading on this box is misleading).
"""

import argparse
import time

import torch


def measure(bytes_moved: float, fn, iters: int) -> float:
    fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return bytes_moved / dt / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-gib", type=float, default=1.0, help="per-tensor size (bf16)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("xpu", args.device)
    n = int(args.size_gib * 1024**3 // 2)  # bf16 = 2 bytes
    x = torch.empty(n, device=dev, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    b = n * 2
    print(f"B70 [{torch.xpu.get_device_name(args.device)}] tensor {b / 1024**3:.2f} GiB, iters={args.iters}")

    print(f"  D2D copy  (R1+W1): {measure(2 * b, lambda: y.copy_(x), args.iters):8.1f} GB/s")
    print(f"  mul out   (R1+W1): {measure(2 * b, lambda: torch.mul(x, 2.0, out=y), args.iters):8.1f} GB/s")
    print(f"  add out   (R2+W1): {measure(3 * b, lambda: torch.add(x, y, out=z), args.iters):8.1f} GB/s")
    print(f"  reduce    (R1)   : {measure(1 * b, lambda: x.sum(), args.iters):8.1f} GB/s")
    try:
        host = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        x2 = torch.empty_like(x)
        print(f"  H2D copy  (W1)   : {measure(1 * b, lambda: x2.copy_(host, non_blocking=True), args.iters):8.1f} GB/s")
    except Exception as e:  # noqa: BLE001
        print(f"  H2D copy: unavailable ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
