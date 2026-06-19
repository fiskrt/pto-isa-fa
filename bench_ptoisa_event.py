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


def make_ptoisa_runner(flash, q, k, v, output, q_heads, kv_heads):
    n_rep = q_heads // kv_heads
    b = q.shape[0]

    def run_once():
        for batch_idx in range(b):
            for q_head_idx in range(q_heads):
                kv_head_idx = q_head_idx // n_rep
                out = flash(q[batch_idx, q_head_idx], k[batch_idx, kv_head_idx], v[batch_idx, kv_head_idx])
                output[batch_idx, q_head_idx].copy_(out)
        return output

    return run_once


def bench_ptoisa(flash, q, k, v, args, q_len, s_len, q_heads, kv_heads):
    output = torch.empty(args.B, q_heads, q_len, args.D, dtype=torch.float32)
    run_once = make_ptoisa_runner(flash, q, k, v, output, q_heads, kv_heads)

    if args.check:
        run_once()
        torch.npu.synchronize()
        ref_output = ref_native_attention(q, k, v, args.causal)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-3, atol=1e-3)
        print(f"PTOISA Check Passed! B={args.B} Q={q_len} S={s_len}")

    us = bench_callable(run_once, args)
    flops = attn_flops(args.B, q_len, s_len, q_heads, args.D, args.causal)
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
    p.add_argument("--force-jit", action="store_true")
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--benchmark-iters", type=int, default=15)
    p.add_argument("--no-flush-cache", action="store_true")
    check = p.add_mutually_exclusive_group()
    check.add_argument("--check", dest="check", action="store_true", default=True)
    check.add_argument("--no-check", dest="check", action="store_false")
    p.add_argument("--plot", default="ptoisa_event_compare.png")
    args = p.parse_args()

    b = args.B
    q_heads = args.q_heads or args.H
    kv_heads = args.kv_heads or args.H
    d = args.D
    if q_heads % kv_heads != 0:
        p.error("--q-heads must be divisible by --kv-heads")

    S_values = [2**n for n in range(10, 17)]
    S_values.extend([48*128*n for n in range(1, 9)])
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

    results = []
    for s_len in S_values:
        q_len = args.Q if args.Q else s_len
        q = torch.randn(b, q_heads, q_len, d, dtype=torch.float16)
        k = torch.randn(b, kv_heads, s_len, d, dtype=torch.float16)
        v = torch.randn(b, kv_heads, s_len, d, dtype=torch.float16)

        ptoisa_result = bench_ptoisa(flash, q, k, v, args, q_len, s_len, q_heads, kv_heads)
        ascendc_result = bench_ascendc(q, k, v, args, q_len, s_len, q_heads, kv_heads)
        results.extend([ptoisa_result, ascendc_result])

        for result in (ptoisa_result, ascendc_result):
            print(
                f"{result.name:<45} B={result.b:<3} Q={result.q:<6} S={result.s:<6} "
                f"{result.ms:.3f} ms  {result.tflops:.2f} TFLOPS"
            )

    if args.plot:
        plot_results(results, args.plot, args)
        #print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
