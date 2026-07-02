#!/usr/bin/env python3
"""
PoC: drive the manual Flash-Attention kernel from torch_npu tensors (no .bin golden files).

We create Q/K/V as torch tensors directly on the NPU, hand their device pointers to the
C-ABI launcher (libtfa_torch.so -> tfa_run), and read the O tensor back. Correctness is
checked against a plain torch reference attention, so nothing on disk is needed.

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

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "build", "lib", "libtfa_torch.so")


def main():
    if not os.path.exists(LIB):
        sys.exit(f"missing {LIB}; build first: bash run.sh -r npu -v Ascend910B1 -n 0 -c case_float_H_128_S0_128_S1_1024")

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
    ws_bytes = lib.tfa_workspace_size()
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    print(f"[poc] workspace = {ws_bytes} bytes")

    # Launch on torch's own current stream, so it is ordered after the Q/K/V tensor creation.
    stream = torch.npu.current_stream().npu_stream
    rc = lib.tfa_run(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(kt.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(stream),
    )
    torch.npu.synchronize()
    if rc != 0:
        sys.exit(f"[poc] tfa_run returned {rc}")

    # torch reference: softmax(Q Kt * scale) @ V, all in fp32
    ref = torch.softmax((q.float() @ kt.float()) * scale, dim=-1) @ v.float()

    # Same bar the kernel's own golden test uses: absolute tolerance 1e-3 (ResultCmp in main.cpp).
    ATOL = 1e-3
    diff = (o - ref).abs()
    max_abs = diff.max().item()
    print(f"[poc] o    [0,:5] = {o[0, :5].tolist()}")
    print(f"[poc] ref  [0,:5] = {ref[0, :5].tolist()}")
    print(f"[poc] max abs diff = {max_abs:.6e}  (atol={ATOL:.0e})")
    print("[poc] PASS" if max_abs < ATOL else "[poc] FAIL")


if __name__ == "__main__":
    main()
