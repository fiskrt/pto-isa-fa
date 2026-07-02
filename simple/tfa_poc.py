#!/usr/bin/env python3
"""PoC + benchmark: drive the manual Flash-Attention kernel from torch_npu tensors.

Creates Q/K/V as torch tensors directly on the NPU, allocates the kernel workspace,
and hands device pointers to the C-ABI launcher (libtfa_torch.so -> tfa_run). First
checks correctness against a plain torch reference attention, then times the kernel
with utils/bench.py:do_bench. Nothing on disk is needed (no .bin golden files).

Run:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    /home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py
"""
import ctypes
import math
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers the 'npu' backend)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.bench import do_bench

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "build", "lib", "libtfa_torch.so")

ATOL = 1e-3  # same bar the kernel's own golden test uses (ResultCmp in main.cpp)


def main():
    if not os.path.exists(LIB):
        sys.exit(f"missing {LIB}; build first: bash build.sh")

    lib = ctypes.CDLL(LIB)
    lib.tfa_run.restype = ctypes.c_int
    lib.tfa_run.argtypes = [ctypes.c_void_p] * 6
    lib.tfa_shape.restype = None
    lib.tfa_shape.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
    lib.tfa_workspace_size.restype = ctypes.c_size_t
    lib.tfa_workspace_size.argtypes = []

    s0 = ctypes.c_int(0)
    head = ctypes.c_int(0)
    s1 = ctypes.c_int(0)
    lib.tfa_shape(ctypes.byref(s0), ctypes.byref(head), ctypes.byref(s1))
    S0, HEAD, S1 = s0.value, head.value, s1.value
    print(f"[poc] kernel shape: S0={S0} HEAD={HEAD} S1={S1}")

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
    ws_bytes = lib.tfa_workspace_size()
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    print(f"[poc] workspace = {ws_bytes} bytes")

    q_ptr = ctypes.c_void_p(q.data_ptr())
    kt_ptr = ctypes.c_void_p(kt.data_ptr())
    v_ptr = ctypes.c_void_p(v.data_ptr())
    o_ptr = ctypes.c_void_p(o.data_ptr())
    ws_ptr = ctypes.c_void_p(workspace.data_ptr())

    def run():
        # Launch on torch's own current stream, so it is ordered after the tensor
        # creation and timed correctly by do_bench's events. tfa_run only enqueues
        # here (no internal sync) — the caller owns synchronization.
        stream = torch.npu.current_stream().npu_stream
        return lib.tfa_run(q_ptr, kt_ptr, v_ptr, o_ptr, ws_ptr, ctypes.c_void_p(stream))

    # --- Correctness vs a torch reference: softmax(Q Kt * scale) @ V, all in fp32 ---
    run()
    torch.npu.synchronize()
    ref = torch.softmax((q.float() @ kt.float()) * scale, dim=-1) @ v.float()
    max_abs = (o - ref).abs().max().item()
    print(f"[poc] o    [0,:5] = {o[0, :5].tolist()}")
    print(f"[poc] ref  [0,:5] = {ref[0, :5].tolist()}")
    print(f"[poc] max abs diff = {max_abs:.6e}  (atol={ATOL:.0e})")
    passed = max_abs < ATOL
    print("[poc] PASS" if passed else "[poc] FAIL")

    # --- Benchmark ---
    t_us = do_bench(run, warmup_iters=10, benchmark_iters=50, unit="us")
    flops = 4.0 * S0 * S1 * HEAD  # QK: 2*S0*S1*HEAD, PV: 2*S0*S1*HEAD
    tflops = flops / (t_us * 1e-6) / 1e12
    print(f"[poc] latency = {t_us:.3f} us  (~{tflops:.2f} TFLOP/s, {flops/1e6:.1f} MFLOP)")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
