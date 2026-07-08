#!/usr/bin/env python3
"""PoC + benchmark: drive the manual Flash-Attention kernel from torch_npu tensors.

Creates Q/K/V as torch tensors directly on the NPU, allocates the kernel workspace,
and hands device pointers to the C-ABI launcher (libtfa_torch.so via tfa_kernel.TfaKernel).
First checks correctness against the torch_npu fused-attention op (used as the reference, since it
is O(S) memory and validates any shape), then times the kernel with utils/bench.py:do_bench.
Nothing on disk is needed (no .bin golden files).

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
CAUSAL_ATOL = 9e-3

# Square shapes swept by the CSV mode: S0 == S1 == 1k, 2k, ..., 32k.
SWEEP_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, ]#64*1024, 128*1024]
# (batch, num_q_heads, num_kv_heads) triples swept by the CSV mode; num_kv_heads must divide
# num_q_heads (GQA). The full sweep is this list x SWEEP_SIZES x {causal, non-causal}.
HEAD_CONFIGS = [
    (1, 1, 1),
    (1, 8, 8),
    (1, 8, 1),
    (16, 8, 1),
    (1, 16, 16),
    (1, 32, 8),
    (1, 32, 16),
    (1, 32, 32),
]
CSV_FIELDS = ["batch", "nq", "nkv", "S0", "S1", "causal", "impl", "tflops", "ms"]

HERE = os.path.dirname(os.path.abspath(__file__))
LIB256 = os.path.join(HERE, "build", "lib", "libtfa_torch.so")            # TILE_S1=256 (build.sh)
LIB512 = os.path.join(HERE, "build_tile512", "lib", "libtfa_torch_tile512.so")     # TILE_S1=512
LIB1024 = os.path.join(HERE, "build_tile1024", "lib", "libtfa_torch_tile1024.so")  # TILE_S1=1024

# The one place to edit what the sweep/plot compares: label -> (libtfa_torch.so, qk_preload).
# TILE_S1 is baked into the .so at build time; qk_preload is a runtime knob (0 = build default),
# so sweeping preload needs no rebuild. Missing .so files are skipped. Valid qk_preload per build
# is TfaKernel.qk_preload_range (currently 2..8).
VARIANTS = {
    # "t256_p2": (LIB256, 2),
    # "t256_p4": (LIB256, 4),
    # "t256_p8": (LIB256, 8),
    # "t512_p2": (LIB512, 2),
    "t512_p4": (LIB512, 4),
    # "t512_p8": (LIB512, 8),
    # "t1024_p2": (LIB1024, 2),
    # "t1024_p4": (LIB1024, 4),
    # "t1024_p8": (LIB1024, 8),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s0", type=int, default=128, help="query rows (multiple of s0_multiple)")
    p.add_argument("--s1", type=int, default=1024, help="key/value rows (multiple of s1_multiple)")
    p.add_argument("--batch", type=int, default=1, help="batch size B")
    p.add_argument("--nq", type=int, default=1, help="number of query heads Nq")
    p.add_argument("--nkv", type=int, default=1, help="number of kv heads Nkv (must divide Nq; GQA)")
    p.add_argument("--causal", action="store_true", help="apply lower-triangular causal mask (attend to key j <= query i)")
    p.add_argument("--profile", action="store_true", help="launch the kernel exactly once for the given shape (no correctness/bench), for use under an external profiler (msprof)")
    p.add_argument("--csv", metavar="PATH", help="sweep SWEEP_SIZES x {causal,non-causal} and write a CSV here")
    p.add_argument("--plot", metavar="CSV_PATH", help="read a CSV written by --csv and plot the speedup grid")
    p.add_argument("--out", metavar="PATH", default="tfa_speedup.png", help="output image path for --plot")
    return p.parse_args()


def _bench_fn(fn, S0, S1, HEAD, causal, n_problems=1):
    """Time `fn` (returns nothing meaningful) and convert to (ms, tflops).

    n_problems = batch * num_q_heads independent attention problems are computed per call.
    """
    flops = 4.0 * n_problems * S0 * S1 * HEAD  # QK: 2*S0*S1*HEAD, PV: 2*S0*S1*HEAD, per problem
    if causal:
        flops *= 0.5  # only the lower triangle is computed (standard causal convention)
    t_us = do_bench(fn, warmup_iters=10, benchmark_iters=50, unit="us")
    return t_us / 1e3, flops / (t_us * 1e-6) / 1e12


def benchmark_one(variants, S0, S1, causal, batch=1, num_q_heads=1, num_kv_heads=1, verbose=True):
    """Run + time every `ours` variant and the torch_npu baseline for one shape.

    Tensors are BNSD: Q/O are [B, Nq, S0, HEAD], K/V are [B, Nkv, S1, HEAD]. Nkv must divide Nq
    (GQA); query head h uses kv head h // (Nq / Nkv). `variants` maps label -> (TfaKernel,
    qk_preload). Returns {label: {"ms", "tflops", "passed"}} with the torch_npu baseline included
    under "torch_npu". All single-shape logic lives here so both the CLI and the CSV sweep share it.
    """
    HEAD = next(iter(variants.values()))[0].config[0]
    dev = "npu:0"
    scale = 1.0 / math.sqrt(HEAD)
    if num_q_heads % num_kv_heads:
        raise ValueError(f"num_kv_heads={num_kv_heads} must divide num_q_heads={num_q_heads}")
    n_problems = batch * num_q_heads

    torch.manual_seed(0)
    # BNSD logical tensors: Q/O [B, Nq, S0, HEAD]; K/V [B, Nkv, S1, HEAD].
    q = (torch.randn(batch, num_q_heads, S0, HEAD, device=dev) * 0.1).to(torch.float16)
    k = (torch.randn(batch, num_kv_heads, S1, HEAD, device=dev) * 0.1).to(torch.float16)
    v = (torch.randn(batch, num_kv_heads, S1, HEAD, device=dev) * 0.1).to(torch.float16)
    # The kernel reads K transposed (DN layout), i.e. each head as [HEAD, S1]. Feed it that way:
    # [B, Nkv, HEAD, S1] contiguous. Per head this is exactly the old kt = K.t(); heads are stacked
    # with stride HEAD*S1 == S1*HEAD, which is what the launcher's kv_head_stride expects.
    k_kernel = k.transpose(2, 3).contiguous()

    atol = CAUSAL_ATOL if causal else ATOL

    # --- Reference: torch_npu's fused attention op (npu_fused_infer_attention_score), BNSD/GQA. ---
    # We verify OUR kernel against this rather than a naive fp32 softmax: the naive path materializes
    # an [B,Nq,S0,S1] score matrix (O(n_problems*S0*S1)) that OOMs at large S, while the fused op is
    # O(S) memory and validates ANY shape. Caveat: for causal, sparse_mode=3's mask is bottom-right
    # aligned, matching our top-left-aligned kernel only when S0 == S1 (always true in the sweep).
    atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=dev), diagonal=1) if causal else None

    def run_baseline():
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v, atten_mask=atten_mask, num_heads=num_q_heads, scale=scale,
            input_layout="BNSD", num_key_value_heads=num_kv_heads,
            pre_tokens=65535, next_tokens=65535, sparse_mode=3 if causal else 0,
        )[0]

    ref = run_baseline().float()  # [B, Nq, S0, HEAD]
    torch.npu.synchronize()

    results = {}

    # --- Our variants (each is a kernel .so + a runtime qk_preload) ---
    for label, (kernel, qk_preload) in variants.items():
        kernel.validate_shape(S0, S1)
        o = torch.zeros(batch, num_q_heads, S0, HEAD, dtype=torch.float32, device=dev)
        workspace = torch.empty(kernel.workspace_size(S0, S1, batch, num_q_heads), dtype=torch.uint8, device=dev)

        def run(kernel=kernel, o=o, workspace=workspace, qk_preload=qk_preload):
            stream = torch.npu.current_stream().npu_stream
            return kernel.run(q.data_ptr(), k_kernel.data_ptr(), v.data_ptr(), o.data_ptr(),
                              workspace.data_ptr(), stream, S0, S1, batch, num_q_heads, num_kv_heads,
                              causal, qk_preload)

        run()
        torch.npu.synchronize()
        max_abs = (o - ref).abs().max().item()
        passed = max_abs < atol
        ms, tflops = _bench_fn(run, S0, S1, HEAD, causal, n_problems)
        results[label] = {"ms": ms, "tflops": tflops, "passed": passed, "max_abs": max_abs}
        # Free this variant's workspace before the next impl so resident buffers don't force the
        # next kernel (esp. torch_npu's internal alloc) to malloc inside its timed loop.
        del run, o, workspace  # run holds o/workspace via default args; drop all three
        torch.npu.empty_cache()

    # torch_npu is the reference itself, so it passes by construction (diff 0); just time it.
    ms, tflops = _bench_fn(run_baseline, S0, S1, HEAD, causal, n_problems)
    results["torch_npu"] = {"ms": ms, "tflops": tflops, "passed": True, "max_abs": 0.0}

    if verbose:
        print(f"[poc] B={batch} Nq={num_q_heads} Nkv={num_kv_heads} S0={S0} S1={S1} HEAD={HEAD} causal={causal}")
        for label, r in results.items():
            status = "SKIP" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
            diff = "  n/a   " if r["max_abs"] is None else f"{r['max_abs']:.2e}"
            print(f"[{label:>12}] {r['tflops']:7.2f} TFLOP/s  {r['ms']*1e3:9.3f} us  diff={diff}  {status}")
    return results


def generate_csv(variants, path, configs=HEAD_CONFIGS, sizes=SWEEP_SIZES):
    """Sweep configs x sizes x {causal, non-causal}, one row per (config, shape, causal, impl).

    `configs` is a list of (batch, Nq, Nkv) triples (Nkv must divide Nq; GQA). The batch/head
    config is recorded in every row so the CSV (and the plot it feeds) is self-describing rather
    than implicitly assuming B=Nq=Nkv=1.
    """
    rows = []
    for batch, num_q_heads, num_kv_heads in configs:
        for S in sizes:
            for causal in (False, True):
                print(f"\n=== B={batch} Nq={num_q_heads} Nkv={num_kv_heads} S0=S1={S} causal={causal} ===")
                res = benchmark_one(variants, S, S, causal,
                                    batch=batch, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
                for label, r in res.items():
                    rows.append({"batch": batch, "nq": num_q_heads, "nkv": num_kv_heads,
                                 "S0": S, "S1": S, "causal": causal, "impl": label,
                                 "tflops": r["tflops"], "ms": r["ms"]})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] wrote {len(rows)} rows to {path}")


def plot_csv(csv_path, out_path):
    """Plot a grid colored by <variant>/torch_npu TFLOP/s (green = ours faster, red = slower).

    One row per (config, ours variant, causal); one column per size, where config is the
    (B, Nq, Nkv) triple. Every impl in the CSV except torch_npu is treated as an ours-variant,
    so adding tile sizes or head configs just adds rows automatically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # data[(cfg, causal, S)] = {impl: tflops}; cfg = (batch, nq, nkv) as strings, .get keeps
    # old configless CSVs working.
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cfg = (row.get("batch", "1"), row.get("nq", "1"), row.get("nkv", "1"))
            key = (cfg, row["causal"] == "True", int(row["S0"]))
            data.setdefault(key, {})[row["impl"]] = float(row["tflops"])

    sizes = sorted({S for _, _, S in data})
    cfgs = sorted({cfg for cfg, _, _ in data})
    variants = sorted({impl for d in data.values() for impl in d} - {"torch_npu"})
    # Rows = every (config, variant, causal) triple; each colored by that variant's speedup.
    rows = [(cfg, v, c) for cfg in cfgs for v in variants for c in (False, True)]
    ratio = [[data[(cfg, c, S)][v] / data[(cfg, c, S)]["torch_npu"] for S in sizes]
             for cfg, v, c in rows]

    fig, ax = plt.subplots(figsize=(1.5 * len(sizes) + 2.0, 0.9 * len(rows) + 1))
    lo = 0.8 #min(min(r) for r in ratio)
    hi = 1.1 # max(max(r) for r in ratio)
    norm = TwoSlopeNorm(vmin=min(lo, 0.99), vcenter=1.0, vmax=max(hi, 1.01))
    ax.imshow(ratio, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{S // 1024}k" for S in sizes])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"B{cfg[0]} Nq{cfg[1]} Nkv{cfg[2]}\n{v} causal={c}" for cfg, v, c in rows])
    ax.set_xlabel("S0 = S1")

    for i, (cfg, v, c) in enumerate(rows):
        for j, S in enumerate(sizes):
            d = data[(cfg, c, S)]
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

    if args.profile:
        # Single kernel launch (no correctness check, no timing loop) so an external profiler
        # (msprof) captures exactly one FA invocation for the requested shape/params.
        label, (kernel, qk_preload) = next(iter(variants.items()))
        S0, S1, HEAD = args.s0, args.s1, kernel.config[0]
        dev = "npu:0"
        kernel.validate_shape(S0, S1)
        torch.manual_seed(0)
        q = (torch.randn(args.batch, args.nq, S0, HEAD, device=dev) * 0.1).to(torch.float16)
        k = (torch.randn(args.batch, args.nkv, S1, HEAD, device=dev) * 0.1).to(torch.float16)
        v = (torch.randn(args.batch, args.nkv, S1, HEAD, device=dev) * 0.1).to(torch.float16)
        k_kernel = k.transpose(2, 3).contiguous()  # DN layout the kernel reads (see benchmark_one)
        o = torch.zeros(args.batch, args.nq, S0, HEAD, dtype=torch.float32, device=dev)
        workspace = torch.empty(kernel.workspace_size(S0, S1, args.batch, args.nq), dtype=torch.uint8, device=dev)
        kernel.run(q.data_ptr(), k_kernel.data_ptr(), v.data_ptr(), o.data_ptr(), workspace.data_ptr(),
                   torch.npu.current_stream().npu_stream, S0, S1, args.batch, args.nq, args.nkv,
                   args.causal, qk_preload)
        torch.npu.synchronize()
        print(f"[profile] ran {label} once: B={args.batch} Nq={args.nq} Nkv={args.nkv} "
              f"S0={S0} S1={S1} HEAD={HEAD} causal={args.causal}")
        return

    if args.csv:
        generate_csv(variants, args.csv)  # sweeps HEAD_CONFIGS defined at the top
        plot_csv(args.csv, args.out)  # auto-plot the sweep we just wrote
        return

    res = benchmark_one(variants, args.s0, args.s1, args.causal,
                        batch=args.batch, num_q_heads=args.nq, num_kv_heads=args.nkv)
    # None == correctness skipped (large S); only fail on an actual False.
    sys.exit(0 if all(r["passed"] is not False for r in res.values()) else 1)


if __name__ == "__main__":
    main()
