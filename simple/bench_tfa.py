#!/usr/bin/env python3
"""Benchmark the manual Flash-Attention kernel via utils/bench.py:do_bench.

Reuses the tfa_run C-ABI launcher from tfa_poc.py. Tensors are created once;
do_bench warms up, flushes L2, and times the kernel over several iterations.
Also verifies correctness once vs a torch reference before timing.
"""
import ctypes
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.bench import do_bench

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "build", "lib", "libtfa_torch.so")


def main():
    lib = ctypes.CDLL(LIB)
    lib.tfa_run.restype = ctypes.c_int
    lib.tfa_run.argtypes = [ctypes.c_void_p] * 5
    lib.tfa_shape.restype = None
    lib.tfa_shape.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3

    s0 = ctypes.c_int(0); head = ctypes.c_int(0); s1 = ctypes.c_int(0)
    lib.tfa_shape(ctypes.byref(s0), ctypes.byref(head), ctypes.byref(s1))
    S0, HEAD, S1 = s0.value, head.value, s1.value
    print(f"[bench] shape S0={S0} HEAD={HEAD} S1={S1}")

    torch.npu.set_device(0)
    dev = "npu:0"
    scale = 1.0 / math.sqrt(HEAD)

    torch.manual_seed(0)
    q = (torch.randn(S0, HEAD, device=dev) * 0.1).to(torch.float16)
    k = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)
    v = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)
    kt = k.t().contiguous()
    o = torch.zeros(S0, HEAD, dtype=torch.float32, device=dev)

    q_ptr = ctypes.c_void_p(q.data_ptr())
    kt_ptr = ctypes.c_void_p(kt.data_ptr())
    v_ptr = ctypes.c_void_p(v.data_ptr())
    o_ptr = ctypes.c_void_p(o.data_ptr())

    def run():
        stream = torch.npu.current_stream().npu_stream
        lib.tfa_run(q_ptr, kt_ptr, v_ptr, o_ptr, ctypes.c_void_p(stream))

    # Correctness sanity check first.
    run()
    torch.npu.synchronize()
    ref = torch.softmax((q.float() @ kt.float()) * scale, dim=-1) @ v.float()
    max_abs = (o - ref).abs().max().item()
    print(f"[bench] correctness max_abs={max_abs:.3e} ({'PASS' if max_abs < 1e-3 else 'FAIL'})")

    t_us = do_bench(run, warmup_iters=10, benchmark_iters=50, unit="us", flush_cache=True)
    print(f"[bench] kernel mean latency = {t_us:.3f} us")

    # FA flops: 2 * S0 * S1 * HEAD (QK) + 2 * S0 * S1 * HEAD (PV)
    flops = 4.0 * S0 * S1 * HEAD
    tflops = flops / (t_us * 1e-6) / 1e12
    print(f"[bench] ~{tflops:.2f} TFLOP/s (FA fwd, {flops/1e6:.1f} MFLOP)")


if __name__ == "__main__":
    main()
