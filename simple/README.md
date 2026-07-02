# simple/ — torch → FA kernel PoC

Self-contained: drive the manual Flash-Attention kernel directly from `torch_npu` tensors.
No golden `.bin` files, no case sweep, no cost model — just the kernel and a torch caller.

## Files
- `fa_performance_kernel.cpp/.h` — the FA kernel + `LaunchTFA` host launcher (device code).
- `pto_macro_*.hpp` — kernel sub-stage macros (matmul / softmax / gu).
- `include/` — vendored `pto/` framework headers (so this folder needs nothing from pto-isa).
- `tfa_torch_launch.cpp` — C-ABI `tfa_run(q,k,v,o,workspace,stream)` + `tfa_workspace_size()`;
  slices the kernel's scratch out of a caller-provided workspace, calls `LaunchTFA<...>`.
- `tfa_poc.py` — creates Q/K/V/O + workspace as NPU tensors, launches via `ctypes`, checks vs a
  torch reference, then benchmarks the kernel with `utils/bench.py:do_bench`.
- `utils/bench.py` — `do_bench` timing helper (warmup, L2 flush, torch-event timing).
- `CMakeLists.txt`, `build.sh`.

## Shape
Fixed at `S0=128, HEAD=128, S1=1024` in two places that must agree:
the `INSTANTIATE_TFA(...)` line in `fa_performance_kernel.cpp` and the `LaunchTFA<...>` call in
`tfa_torch_launch.cpp`. (The template is defined in the `.cpp`, so the shape must be explicitly
instantiated into the `.so` for the host wrapper to link.)

## Build & run
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
/home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py
```
Expect `[poc] PASS` (max abs diff ~2e-4 vs torch reference, atol 1e-3) followed by a
`[poc] latency = ... us` line from the benchmark.

Arch is `dav-c220` (Ascend 910B1/B2/B3, A2). For Ascend910_9599 (A3) set `dav-c310` in `CMakeLists.txt`.
