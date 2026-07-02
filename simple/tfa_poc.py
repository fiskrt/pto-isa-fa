#!/usr/bin/env python3
"""PoC + benchmark: drive the manual Flash-Attention kernel from torch_npu tensors.

Creates Q/K/V as torch tensors directly on the NPU, allocates the kernel workspace,
and hands device pointers to the C-ABI launcher (libtfa_torch.so via tfa_kernel.TfaKernel).
First checks correctness against a plain torch reference attention, then times the kernel
with utils/bench.py:do_bench. Nothing on disk is needed (no .bin golden files).

S0/S1 are runtime kernel args now, so any shape with S0 a multiple of the reported s0_multiple
and S1 a multiple of s1_multiple works against a single build; override with --s0/--s1.

Run:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    /home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py [--s0 128] [--s1 1024]
"""
import argparse
import math
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the 'npu' backend)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfa_kernel import TfaKernel
from utils.bench import do_bench

ATOL = 1e-3  # same bar the kernel's own golden test uses (ResultCmp in main.cpp)
# Causal needs a looser bar: the kernel stores the softmax P and V in fp16, so each row carries
# ~per-key fp16 error that averages out over many keys (~1/sqrt(#keys)). Non-causal rows sum all
# S1 keys so it stays ~2e-4; causal's near-diagonal rows sum only a handful (row 0 is exact, row i
# sums i+1 keys), so the residual peaks ~4e-3 there and decays as rows attend more keys.
CAUSAL_ATOL = 6e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s0", type=int, default=128, help="query rows (multiple of s0_multiple)")
    p.add_argument("--s1", type=int, default=1024, help="key/value rows (multiple of s1_multiple)")
    p.add_argument("--causal", action="store_true", help="apply lower-triangular causal mask (attend to key j <= query i)")
    return p.parse_args()


def main():
    args = parse_args()

    kernel = TfaKernel()
    HEAD, s0_mult, s1_mult = kernel.config
    S0, S1 = args.s0, args.s1
    kernel.validate_shape(S0, S1)
    causal = args.causal
    print(f"[poc] kernel shape: S0={S0} HEAD={HEAD} S1={S1}  (multiples: S0%{s0_mult}, S1%{s1_mult})  causal={causal}")

    torch.npu.set_device(0)
    dev = "npu:0"
    scale = 1.0 / math.sqrt(HEAD)

    torch.manual_seed(0)
    # Kernel layouts: q=[S0,HEAD], k transposed=[HEAD,S1], v=[S1,HEAD]; output o=[S0,HEAD] fp32.
    q = (torch.randn(S0, HEAD, device=dev) * 0.1).to(torch.float16)
    k = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)  # logical K as [S1,HEAD]
    v = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)
    kt = k.t().contiguous()  # [HEAD, S1] as the kernel expects
    o = torch.zeros(S0, HEAD, dtype=torch.float32, device=dev)

    # Kernel scratch: allocate one workspace block here and hand it to the kernel.
    # Kept alive for every launch below (the torch-stream path is async).
    ws_bytes = kernel.workspace_size(S0, S1)
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    print(f"[poc] workspace = {ws_bytes // 2 ** 20} MiB")

    def run():
        # Launch on torch's own current stream, so it is ordered after the tensor
        # creation and timed correctly by do_bench's events. run() only enqueues
        # here (no internal sync) — the caller owns synchronization.
        stream = torch.npu.current_stream().npu_stream
        return kernel.run(q.data_ptr(), kt.data_ptr(), v.data_ptr(), o.data_ptr(),
                          workspace.data_ptr(), stream, S0, S1, causal)

    # --- Correctness vs a torch reference: softmax(Q Kt * scale) @ V, all in fp32 ---
    run()
    torch.npu.synchronize()
    scores = (q.float() @ kt.float()) * scale  # [S0, S1]
    if causal:
        # Match the kernel: query row i attends to key j only when j <= i (absolute indices).
        mask = torch.ones(S0, S1, device=dev, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ v.float()
    max_abs = (o - ref).abs().max().item()
    atol = CAUSAL_ATOL if causal else ATOL
    print(f"[poc] o    [0,:5] = {o[0, :5].tolist()}")
    print(f"[poc] ref  [0,:5] = {ref[0, :5].tolist()}")
    print(f"[poc] max abs diff = {max_abs:.6e}  (atol={atol:.0e})")
    passed = max_abs < atol
    print("[poc] PASS" if passed else "[poc] FAIL")

    # --- Benchmark ---
    t_us = do_bench(run, warmup_iters=10, benchmark_iters=50, unit="us")
    flops = 4.0 * S0 * S1 * HEAD  # QK: 2*S0*S1*HEAD, PV: 2*S0*S1*HEAD
    tflops = flops / (t_us * 1e-6) / 1e12
    print(f"[poc] latency = {t_us:.3f} us  (~{tflops:.2f} TFLOP/s, {flops/1e6:.1f} MFLOP)")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
