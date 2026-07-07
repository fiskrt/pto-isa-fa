import argparse
import os
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

import torch
import torch_npu

from flash_atten.jit_util import jit_compile_flash
from utils.bench import do_bench


torch.set_default_device("npu")
torch.manual_seed(0)


@dataclass
class BenchResult:
    name: str
    b: int
    q: int
    s: int
    us: float
    flops: float

    @property
    def ms(self):
        return self.us / 1000

    @property
    def tflops(self):
        return self.flops / self.us / 1e6


# TODO: only good for square matrices
def attn_flops(b, q, s, h, d, causal):
    pairs = q * s / (2 if causal else 1)
    return 4 * b * h * pairs * d


def make_attention_keep_mask(q_len: int, kv_len: int, causal: bool, device: torch.device) -> torch.Tensor:
    if not causal:
        return torch.ones((q_len, kv_len), dtype=torch.bool, device=device)

    q_pos = torch.arange(q_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return kv_pos <= q_pos + (kv_len - q_len)


def ref_native_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    if k.shape[1] != q.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )

    # this will OOM since its naive
    # q_float = q.float()
    # k_float = k.float()
    # v_float = v.float()
    # scale = (1.0 / q.shape[-1]) ** 0.5
    # scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    # keep_mask = make_attention_keep_mask(q.shape[-2], k.shape[-2], causal, q.device)
    # keep_mask = keep_mask[None, None, :, :]
    # scores = scores.masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
    # probs = torch.softmax(scores, dim=-1).masked_fill(~keep_mask, 0.0)
    # return torch.matmul(probs, v_float)


def make_qkv(pattern, b, q_heads, kv_heads, q_len, s_len, d, dtype=torch.float16):
    """Build (q, k, v) for a diagnostic pattern.

    Each pattern suppresses one error mechanism; the fp32-SDPA reference is the
    exact ground truth for all of them (inputs are fp16-exact by construction):

      random  - full error (baseline)
      constV  - V is a constant => O == const regardless of weights. Removes weight
                correctness; a fail = numerator(fp16-P)/denominator(fp32-L) inconsistency
                or accumulation.
      uniform - Q=0 => all scores equal => unnormalized P == 1.0 (EXACT in fp16, no
                rounding). O == mean(V). A fail = fp32 reduction / streaming-sum / L bug,
                NOT fp16-P.
      onehot  - one key dominates (huge score) => O == V_j. Removes accumulation of many
                small weights; a fail = softmax/exp/masking/indexing logic bug.
    """
    rand = lambda heads, length: torch.randn(b, heads, length, d, dtype=dtype)
    if pattern == "random":
        return 16*rand(q_heads, q_len), rand(kv_heads, s_len), rand(kv_heads, s_len)
    if pattern == "constV":
        q = rand(q_heads, q_len)
        k = rand(kv_heads, s_len)
        v = 10*torch.full((b, kv_heads, s_len, d), 0.5, dtype=dtype)  # 0.5 exact in fp16
        return q, k, v
    if pattern == "constKV":
        q = rand(q_heads, q_len)
        k = 1*torch.full((b, kv_heads, s_len, d), 0.5, dtype=dtype)  # 0.5 exact in fp16
        v = 10*torch.full((b, kv_heads, s_len, d), 0.5, dtype=dtype)  # 0.5 exact in fp16
        return q, k, v
    if pattern == "constK":
        q = rand(q_heads, q_len)
        k = 1*torch.full((b, kv_heads, s_len, d), 0.5, dtype=dtype)  # 0.5 exact in fp16
        v = rand(kv_heads, s_len)
        return q, k, v
    if pattern == "constQ":
        q = 1*torch.full((b, q_heads, s_len, d), 0.5, dtype=dtype)  # 0.5 exact in fp16
        k = rand(kv_heads, s_len)
        v = rand(kv_heads, s_len)
        return q, k, v
    if pattern == "uniform":
        q = torch.zeros(b, q_heads, q_len, d, dtype=dtype)  # all scores -> 0 -> P==1.0 exact
        k = rand(kv_heads, s_len)
        v = rand(kv_heads, s_len)
        return q, k, v
    if pattern == "onehot":
        # Key 0 dominates: score gap must beat ln(S) after scale (scale=1/sqrt(d)).
        # raw score = beta; scaled = beta/sqrt(d) must be >> ln(s_len). beta=50000 (<65504
        # fp16 max) gives scaled ~4400 >> ln(49152)=10.8, so softmax is one-hot at key 0.
        # Key 0 is always causal-valid.
        q = torch.zeros(b, q_heads, q_len, d, dtype=dtype)
        q[:, :, :, 0] = 1.0
        k = torch.zeros(b, kv_heads, s_len, d, dtype=dtype)
        k[:, :, 0, 0] = 50000.0
        v = rand(kv_heads, s_len)
        return q, k, v
    raise ValueError(f"unknown pattern: {pattern}")


def bench_callable(fn, args):
    torch.npu.synchronize()
    us = float(
        do_bench(
            fn,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            flush_cache=not args.no_flush_cache,
        )
    )
    torch.npu.synchronize()
    return us


def make_ptoisa_runner(flash, q, k, v, q_heads, kv_heads):
    # Single launch handles all B*Hq heads (GQA/MQA via kv-head index map inside the kernel).
    def run_once():
        return flash(q, k, v)

    return run_once


def report_check(tag, ref_output, output, rtol=1e-3, atol=1e-3):
    """Report the ratio metric without aborting, so a whole grid can run."""
    abs_diff = (ref_output - output).abs()
    tol = atol + rtol * ref_output.abs()
    ratio = abs_diff / tol  # >1 anywhere => assert_close would fail
    argmax = ratio.argmax()
    n_fail = (ratio > 1).sum().item()
    max_ratio = ratio.flatten()[argmax].item()
    verdict = "PASS" if max_ratio <= 1 else "FAIL"
    print(f'  [{tag}] {verdict}  max|diff|={abs_diff.max().item():.6g}  '
          f'worst: |diff|={abs_diff.flatten()[argmax].item():.6g} '
          f'|ref|={ref_output.abs().flatten()[argmax].item():.6g} '
          f'ratio={max_ratio:.4g}  #fail={n_fail}/{ratio.numel()}')
    return max_ratio <= 1


def bench_ptoisa(flash, q, k, v, args, q_len, s_len, q_heads, kv_heads):
    run_once = make_ptoisa_runner(flash, q, k, v, q_heads, kv_heads)
    us = bench_callable(run_once, args)
    flops = attn_flops(args.B, q_len, s_len, q_heads, args.D, args.causal)

    if args.check:
        output = run_once()
        torch.npu.synchronize()
        ref_output = ref_native_attention(q, k, v, args.causal)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-3, atol=1e-3)
        print(f"PTOISA Check Passed! B={args.B} Q={q_len} S={s_len}")
        #report_check("PTOISA", ref_output, output)

    return BenchResult("PTOISA", args.B, q_len, s_len, us, flops)


def make_ascendc_runner(q, k, v, q_heads, kv_heads, head_size, causal):
    sm_scale = (1.0 / head_size) ** 0.5
    atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1) if causal else None

    def run_once():
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            atten_mask=atten_mask,
            num_heads=q_heads,
            scale=sm_scale,
            input_layout="BNSD",
            num_key_value_heads=kv_heads,
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=3 if causal else 0,
        )[0]

    return run_once


def bench_ascendc(q, k, v, args, q_len, s_len, q_heads, kv_heads):
    run_once = make_ascendc_runner(q, k, v, q_heads, kv_heads, args.D, args.causal)
    us = bench_callable(run_once, args)
    flops = attn_flops(args.B, q_len, s_len, q_heads, args.D, args.causal)

    # if args.check:
    #     output = run_once()
    #     torch.npu.synchronize()
    #     ref_output = ref_native_attention(q, k, v, args.causal)
    #     torch.npu.synchronize()
    #     report_check("AscendC", ref_output, output)

    return BenchResult("AscendC npu_fused_infer_attention_score", args.B, q_len, s_len, us, flops)


def title_args(args):
    q_heads = args.q_heads or args.H
    kv_heads = args.kv_heads or args.H
    return f"B={args.B}, Q={args.Q or 'S'}, H={args.H}, QH={q_heads}, KVH={kv_heads}, D={args.D}, causal={args.causal}"


def annotate_ratio(ax, ptoisa_by_s, ascendc_by_s, value_fn):
    for s_len in sorted(ptoisa_by_s.keys() & ascendc_by_s.keys()):
        ptoisa_value = value_fn(ptoisa_by_s[s_len])
        ascendc_value = value_fn(ascendc_by_s[s_len])
        if ascendc_value == 0:
            continue
        ratio = ptoisa_value / ascendc_value
        ax.annotate(
            f"{ratio:.2f}x",
            xy=(s_len, max(ptoisa_value, ascendc_value)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2F2F2F",
        )


def plot_results(results, path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "PTOISA": "#0072B2",
        "AscendC npu_fused_infer_attention_score": "#D55E00",
    }
    markers = {
        "PTOISA": "o",
        "AscendC npu_fused_infer_attention_score": "s",
    }
    names = ["PTOISA", "AscendC npu_fused_infer_attention_score"]
    xs = sorted({row.s for row in results})
    series_by_name = {name: sorted((row for row in results if row.name == name), key=lambda row: row.s) for name in names}

    fig, (ax_tflops, ax_ms) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    for name in names:
        series = series_by_name[name]
        if not series:
            continue
        ax_tflops.plot(
            [row.s for row in series],
            [row.tflops for row in series],
            marker=markers[name],
            color=colors[name],
            linewidth=2.2,
            label=name,
        )
        ax_ms.plot(
            [row.s for row in series],
            [row.ms for row in series],
            marker=markers[name],
            color=colors[name],
            linewidth=2.2,
            label=name,
        )

    ptoisa_by_s = {row.s: row for row in series_by_name["PTOISA"]}
    ascendc_by_s = {row.s: row for row in series_by_name["AscendC npu_fused_infer_attention_score"]}
    annotate_ratio(ax_tflops, ptoisa_by_s, ascendc_by_s, lambda row: row.tflops)
    annotate_ratio(ax_ms, ptoisa_by_s, ascendc_by_s, lambda row: row.ms)

    for ax in (ax_tflops, ax_ms):
        ax.set_xscale("log", base=2)
        ax.margins(y=0.18)
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.legend(frameon=False)

    ax_tflops.set_ylabel("TFLOPS")
    ax_ms.set_yscale("log")
    ax_ms.set_ylabel("Time (ms)")
    ax_ms.set_xlabel("Sequence length S")
    ax_ms.set_xticks(xs)
    ax_ms.set_xticklabels([f"{s // 1024}k" for s in xs], rotation=60, ha="right")
    fig.suptitle(title_args(args))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_dir = os.path.dirname(path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(path, dpi=180)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--Q", type=int)
    p.add_argument("--S", type=int)
    p.add_argument("--H", type=int, default=1)
    p.add_argument("--q-heads", type=int)
    p.add_argument("--kv-heads", type=int)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--cube-s0", type=int)
    p.add_argument("--tile-s1", type=int, default=256)
    p.add_argument("--qk-preload", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--force-jit", action="store_true")
    p.add_argument("--warmup-iters", type=int, default=25)
    p.add_argument("--benchmark-iters", type=int, default=40)
    p.add_argument("--no-flush-cache", action="store_true")
    check = p.add_mutually_exclusive_group()
    check.add_argument("--check", dest="check", action="store_true", default=True)
    check.add_argument("--no-check", dest="check", action="store_false")
    p.add_argument("--plot", default="ptoisa_event_compare.png")
    p.add_argument("--pattern", default="random",
                   help="diagnostic qkv pattern; 'grid' runs all patterns for isolation")
    args = p.parse_args()

    b = args.B
    q_heads = args.q_heads or args.H
    kv_heads = args.kv_heads or args.H
    d = args.D
    if q_heads % kv_heads != 0:
        p.error("--q-heads must be divisible by --kv-heads")

    S_values = [2**n for n in range(10, 16)]
    S_values.extend([48*128*n for n in range(1, 12)])
    if args.S:
        S_values = [args.S]
    flash = jit_compile_flash(
        head_size=d,
        cube_s0=args.cube_s0,
        tile_s1=args.tile_s1,
        qk_preload=args.qk_preload,
        causal=args.causal,
        force=args.force_jit,
    )

    patterns = ["random", "constV", "uniform", "onehot"] if args.pattern == "grid" else [args.pattern]

    results = []
    S_values = S_values*args.repeats
    for s_len in S_values:
        q_len = args.Q if args.Q else s_len
        for pattern in patterns:
            print(f"=== pattern={pattern}  B={b} Q={q_len} S={s_len} QH={q_heads} KVH={kv_heads} causal={args.causal} ===")
            q, k, v = make_qkv(pattern, b, q_heads, kv_heads, q_len, s_len, d)

            torch.npu.synchronize()
            ascendc_result = bench_ascendc(q, k, v, args, q_len, s_len, q_heads, kv_heads)
            ptoisa_result = bench_ptoisa(flash, q, k, v, args, q_len, s_len, q_heads, kv_heads)
            results.extend([ptoisa_result, ascendc_result])

            for result in (ptoisa_result, ascendc_result):
                print(
                    f"{result.name:<45} B={result.b:<3} Q={result.q:<6} S={result.s:<6} "
                    f"{result.ms:.3f} ms  {result.tflops:.2f} TFLOPS"
                )

    if args.plot and args.pattern == "random":
        plot_results(results, args.plot, args)
        #print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
