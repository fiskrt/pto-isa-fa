# simple/ — torch → FA kernel PoC

Self-contained: drive the manual Flash-Attention kernel directly from `torch_npu` tensors.
No golden `.bin` files, no case sweep, no cost model — just the kernel and a torch caller.

## Files
- `fa_performance_kernel.cpp/.h` — the FA kernel + `LaunchTFA` host launcher (device code).
- `pto_macro_*.hpp` — kernel sub-stage macros (matmul / softmax / gu).
- `include/` — vendored `pto/` framework headers (so this folder needs nothing from pto-isa).
- `tfa_torch_launch.cpp` — C-ABI `tfa_run(q,k,v,o,workspace,stream,s0,s1)` + `tfa_workspace_size(s0,s1)`
  + `tfa_config(head,s0_mult,s1_mult)`; slices the kernel's scratch out of a caller-provided
  workspace, calls `LaunchTFA<...>`.
- `tfa_kernel.py` — `TfaKernel`, a thin `ctypes` wrapper over `libtfa_torch.so` (all FFI lives here).
- `tfa_poc.py` — creates Q/K/V/O + workspace as NPU tensors, launches via `TfaKernel`, checks vs a
  torch reference, then benchmarks the kernel with `utils/bench.py:do_bench`.
- `utils/bench.py` — `do_bench` timing helper (warmup, L2 flush, torch-event timing).
- `CMakeLists.txt`, `build.sh`.

## Shape
`S0` (rows) and `S1` (cols) are **runtime** kernel args, so one build serves any shape with
`S0 % CUBE_S0 == 0` and `S1 % TILE_S1 == 0` (query the multiples via `tfa_config`). Only the tiling
(`HEAD=128, CUBE_S0=128, CUBE_S1=128, TILE_S1=256, QK_PRELOAD=4`) is fixed at compile time, and must
agree between the `INSTANTIATE_TFA(...)` line in `fa_performance_kernel.cpp` and the `LaunchTFA<...>`
call in `tfa_torch_launch.cpp`. Pick a shape at runtime with `tfa_poc.py --s0 <rows> --s1 <cols>`.

## Build & run
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
/home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py            # default S0=128 S1=1024
/home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py --s0 256 --s1 2048
/home/fskogh/famy/.fa_env/bin/python3 tfa_poc.py --s0 2048 --s1 2048 --causal
```
`--causal` applies a lower-triangular mask (query row `i` attends to key `j <= i`). Both mask
variants are compiled in (`INSTANTIATE_TFA(... , false/true)`); `tfa_run` picks one at runtime.
Causal uses a looser bar (`6e-3` vs `1e-3`): the kernel keeps the softmax `P` and `V` in fp16, so
each row carries ~per-key fp16 error that averages out over many keys but not over causal's
near-diagonal rows (row 0 is exact, row `i` sums only `i+1` keys), where the residual peaks ~4e-3.
Expect `[poc] PASS` (max abs diff ~2e-4 vs torch reference, atol 1e-3) followed by a
`[poc] latency = ... us` line from the benchmark.

Arch is `dav-c220` (Ascend 910B1/B2/B3, A2). For Ascend910_9599 (A3) set `dav-c310` in `CMakeLists.txt`.
