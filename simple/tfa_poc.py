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
import csv
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

# The correctness check builds an fp32 [S0,S1] score matrix + mask (O(S0*S1)); at 64k that alone is
# ~16 GiB and OOMs. Above this per-dim size we skip the reference and only benchmark (no diff/PASS).
REF_MAX_S = 32768

# Square shapes swept by the CSV mode: S0 == S1 == 1k, 2k, ..., 32k.
SWEEP_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 64*1024, 128*1024]
CSV_FIELDS = ["S0", "S1", "causal", "impl", "tflops", "ms"]

HERE = os.path.dirname(os.path.abspath(__file__))
LIB256 = os.path.join(HERE, "build", "lib", "libtfa_torch.so")            # TILE_S1=256 (build.sh)
LIB512 = os.path.join(HERE, "build_tile512", "lib", "libtfa_torch_tile512.so")  # TILE_S1=512

# The one place to edit what the sweep/plot compares: label -> (libtfa_torch.so, qk_preload).
# TILE_S1 is baked into the .so at build time; qk_preload is a runtime knob (0 = build default),
# so sweeping preload needs no rebuild. Missing .so files are skipped. Valid qk_preload per build
# is TfaKernel.qk_preload_range (currently 2..8).
VARIANTS = {
    "t256_p2": (LIB256, 2),
    "t256_p4": (LIB256, 4),
    "t256_p8": (LIB256, 8),
    "t512_p2": (LIB512, 2),
    "t512_p4": (LIB512, 4),
    "t512_p8": (LIB512, 8),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s0", type=int, default=128, help="query rows (multiple of s0_multiple)")
    p.add_argument("--s1", type=int, default=1024, help="key/value rows (multiple of s1_multiple)")
    p.add_argument("--causal", action="store_true", help="apply lower-triangular causal mask (attend to key j <= query i)")
    p.add_argument("--csv", metavar="PATH", help="sweep SWEEP_SIZES x {causal,non-causal} and write a CSV here")
    p.add_argument("--plot", metavar="CSV_PATH", help="read a CSV written by --csv and plot the speedup grid")
    p.add_argument("--out", metavar="PATH", default="tfa_speedup.png", help="output image path for --plot")
    return p.parse_args()


def _bench_fn(fn, S0, S1, HEAD, causal):
    """Time `fn` (returns nothing meaningful) and convert to (ms, tflops)."""
    flops = 4.0 * S0 * S1 * HEAD  # QK: 2*S0*S1*HEAD, PV: 2*S0*S1*HEAD
    if causal:
        flops *= 0.5  # only the lower triangle is computed (standard causal convention)
    t_us = do_bench(fn, warmup_iters=10, benchmark_iters=50, unit="us")
    return t_us / 1e3, flops / (t_us * 1e-6) / 1e12


def benchmark_one(variants, S0, S1, causal, verbose=True):
    """Run + time every `ours` variant and the torch_npu baseline for one shape.

    `variants` maps label -> (TfaKernel, qk_preload). Returns {label: {"ms", "tflops", "passed"}}
    with the torch_npu baseline included under "torch_npu". All single-shape logic lives here so
    both the CLI and the CSV sweep share it.
    """
    HEAD = next(iter(variants.values()))[0].config[0]
    dev = "npu:0"
    scale = 1.0 / math.sqrt(HEAD)

    torch.manual_seed(0)
    # Kernel layouts: q=[S0,HEAD], k transposed=[HEAD,S1], v=[S1,HEAD]; output o=[S0,HEAD] fp32.
    q = (torch.randn(S0, HEAD, device=dev) * 0.1).to(torch.float16)
    k = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)  # logical K as [S1,HEAD]
    v = (torch.randn(S1, HEAD, device=dev) * 0.1).to(torch.float16)
    kt = k.t().contiguous()  # [HEAD, S1] as the kernel expects

    # --- torch reference (fp32) for correctness: softmax(Q Kt * scale) @ V ---
    # The [S0,S1] score matrix is O(S0*S1); skip it above REF_MAX_S to avoid OOM (benchmark only).
    check = S0 <= REF_MAX_S and S1 <= REF_MAX_S
    ref = None
    if check:
        scores = (q.float() @ kt.float()) * scale  # [S0, S1]
        if causal:
            mask = torch.ones(S0, S1, device=dev, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        ref = torch.softmax(scores, dim=-1) @ v.float()
    else:
        print(f"[poc] S={S0}>{REF_MAX_S}: skipping fp32 reference (would OOM); benchmark only, no correctness check")
    atol = CAUSAL_ATOL if causal else ATOL

    results = {}

    # --- Our variants (each is a kernel .so + a runtime qk_preload) ---
    for label, (kernel, qk_preload) in variants.items():
        kernel.validate_shape(S0, S1)
        o = torch.zeros(S0, HEAD, dtype=torch.float32, device=dev)
        workspace = torch.empty(kernel.workspace_size(S0, S1), dtype=torch.uint8, device=dev)

        def run(kernel=kernel, o=o, workspace=workspace, qk_preload=qk_preload):
            stream = torch.npu.current_stream().npu_stream
            return kernel.run(q.data_ptr(), kt.data_ptr(), v.data_ptr(), o.data_ptr(),
                              workspace.data_ptr(), stream, S0, S1, causal, qk_preload)

        run()
        torch.npu.synchronize()
        max_abs = (o - ref).abs().max().item() if check else None
        passed = (max_abs < atol) if check else None  # None = not checked (ref skipped)
        ms, tflops = _bench_fn(run, S0, S1, HEAD, causal)
        results[label] = {"ms": ms, "tflops": tflops, "passed": passed, "max_abs": max_abs}
        # Free this variant's workspace before the next impl so resident buffers don't force the
        # next kernel (esp. torch_npu's internal alloc) to malloc inside its timed loop.
        del run, o, workspace  # run holds o/workspace via default args; drop all three
        torch.npu.empty_cache()

    # --- Baseline: torch_npu's fused attention op (npu_fused_infer_attention_score) ---
    qb = q.view(1, 1, S0, HEAD)
    kb = k.view(1, 1, S1, HEAD)
    vb = v.view(1, 1, S1, HEAD)
    # sparse_mode=3 uses a compressed upper-triangular mask (True == masked out), bottom-right
    # aligned; our kernel/ref use top-left aligned causal. They coincide only when S0==S1.
    atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=dev), diagonal=1) if causal else None

    def run_baseline():
        return torch_npu.npu_fused_infer_attention_score(
            qb, kb, vb, atten_mask=atten_mask, num_heads=1, scale=scale,
            input_layout="BNSD", num_key_value_heads=1,
            pre_tokens=65535, next_tokens=65535, sparse_mode=3 if causal else 0,
        )[0]

    if check:
        ob = run_baseline().view(S0, HEAD).float()
        torch.npu.synchronize()
        base_max = (ob - ref).abs().max().item()
    else:
        base_max = None
    base_passed = (base_max < atol) if check else None
    ms, tflops = _bench_fn(run_baseline, S0, S1, HEAD, causal)
    results["torch_npu"] = {"ms": ms, "tflops": tflops, "passed": base_passed, "max_abs": base_max}

    if verbose:
        print(f"[poc] S0={S0} S1={S1} HEAD={HEAD} causal={causal}")
        for label, r in results.items():
            status = "SKIP" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
            diff = "  n/a   " if r["max_abs"] is None else f"{r['max_abs']:.2e}"
            print(f"[{label:>12}] {r['tflops']:7.2f} TFLOP/s  {r['ms']*1e3:9.3f} us  diff={diff}  {status}")
    return results


def generate_csv(variants, path, sizes=SWEEP_SIZES):
    """Sweep sizes x {causal, non-causal}, one tidy row per (shape, causal, impl)."""
    rows = []
    for S in sizes:
        for causal in (False, True):
            print(f"\n=== S0=S1={S} causal={causal} ===")
            res = benchmark_one(variants, S, S, causal)
            for label, r in res.items():
                rows.append({"S0": S, "S1": S, "causal": causal, "impl": label,
                             "tflops": r["tflops"], "ms": r["ms"]})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] wrote {len(rows)} rows to {path}")


def plot_csv(csv_path, out_path):
    """Plot a grid colored by <variant>/torch_npu TFLOP/s (green = ours faster, red = slower).

    One row per (ours variant, causal); one column per size. Every impl in the CSV except
    torch_npu is treated as an ours-variant, so adding tile sizes just adds rows automatically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # data[(causal, S)] = {impl: tflops}
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["causal"] == "True", int(row["S0"]))
            data.setdefault(key, {})[row["impl"]] = float(row["tflops"])

    sizes = sorted({S for _, S in data})
    variants = sorted({impl for d in data.values() for impl in d} - {"torch_npu"})
    # Rows = every (variant, causal) pair; each colored by that variant's speedup vs torch_npu.
    rows = [(v, c) for v in variants for c in (False, True)]
    ratio = [[data[(c, S)][v] / data[(c, S)]["torch_npu"] for S in sizes] for v, c in rows]

    fig, ax = plt.subplots(figsize=(1.5 * len(sizes) + 1.5, 0.9 * len(rows) + 1))
    lo = min(min(r) for r in ratio)
    hi = max(max(r) for r in ratio)
    norm = TwoSlopeNorm(vmin=min(lo, 0.99), vcenter=1.0, vmax=max(hi, 1.01))
    ax.imshow(ratio, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{S // 1024}k" for S in sizes])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{v}\ncausal={c}" for v, c in rows])
    ax.set_xlabel("S0 = S1")

    for i, (v, c) in enumerate(rows):
        for j, S in enumerate(sizes):
            d = data[(c, S)]
            ax.text(j, i, f"{ratio[i][j]:.2f}x\n{d[v]:.0f} vs {d['torch_npu']:.0f}\nTFLOP/s",
                    ha="center", va="center", fontsize=8)

    ax.set_title("Speedup ours / torch_npu  (green = ours faster)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")


def main():
    args = parse_args()

    if args.plot:
        plot_csv(args.plot, args.out)
        return

    torch.npu.set_device(0)
    # Build the variant table: load one TfaKernel per unique .so (cached), skipping missing builds
    # so a single-tile build still runs. Each variant is (kernel, qk_preload).
    cache = {}
    variants = {}
    for label, (path, preload) in VARIANTS.items():
        if not os.path.exists(path):
            continue
        if path not in cache:
            cache[path] = TfaKernel(path)
        variants[label] = (cache[path], preload)
    if not variants:
        raise FileNotFoundError(f"no kernel .so found; build first (bash build.sh). VARIANTS: {VARIANTS}")

    if args.csv:
        generate_csv(variants, args.csv)
        plot_csv(args.csv, args.out)  # auto-plot the sweep we just wrote
        return

    res = benchmark_one(variants, args.s0, args.s1, args.causal)
    # None == correctness skipped (large S); only fail on an actual False.
    sys.exit(0 if all(r["passed"] is not False for r in res.values()) else 1)


if __name__ == "__main__":
    main()
