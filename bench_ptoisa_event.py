import argparse

import torch
import torch_npu

from flash_atten.jit_util import jit_compile_flash
from utils.bench import do_bench


torch.set_default_device("npu")
torch.manual_seed(0)


def attn_flops(b, q, s, h, d, causal):
    pairs = q * s / (2 if causal else 1)
    return 4 * b * h * pairs * d


def ref_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
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
    p.add_argument("--check", action="store_true", default=True)
    args = p.parse_args()

    b = args.B
    q_heads = args.q_heads or args.H
    kv_heads = args.kv_heads or args.H
    d = args.D
    if q_heads % kv_heads != 0:
        p.error("--q-heads must be divisible by --kv-heads")

    S_values = [2**n for n in range(10, 17)]
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

    for s_len in S_values:
        q_len = args.Q if args.Q else s_len
        q = torch.randn(b, q_heads, q_len, d, dtype=torch.float16)
        k = torch.randn(b, kv_heads, s_len, d, dtype=torch.float16)
        v = torch.randn(b, kv_heads, s_len, d, dtype=torch.float16)
        output = torch.empty(b, q_heads, q_len, d, dtype=torch.float32)

        n_rep = q_heads // kv_heads

        def run_once():
            for batch_idx in range(b):
                for q_head_idx in range(q_heads):
                    kv_head_idx = q_head_idx // n_rep
                    out = flash(q[batch_idx, q_head_idx], k[batch_idx, kv_head_idx], v[batch_idx, kv_head_idx])
                    output[batch_idx, q_head_idx].copy_(out)

        if args.check:
            run_once()
            torch.npu.synchronize()
            ref_output = ref_flash_attn(q, k, v, args.causal)
            torch.npu.synchronize()
            torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
            print(f"Check Passed! B={b} Q={q_len} S={s_len}")

        torch.npu.synchronize()
        us = float(
            do_bench(
                run_once,
                warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters,
                flush_cache=not args.no_flush_cache,
            )
        )
        torch.npu.synchronize()
        flops = attn_flops(b, q_len, s_len, q_heads, d, args.causal)
        print(f"B={b:<3} Q={q_len:<6} S={s_len:<6} {us / 1000:.3f} ms  {flops / us / 1e6:.2f} TFLOPS")


if __name__ == "__main__":
    main()
