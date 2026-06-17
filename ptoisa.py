import argparse

import torch
import torch_npu

from flash_atten.jit_util import jit_compile_flash


torch.set_default_device("npu")
torch.manual_seed(0)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="batch size")
    parser.add_argument("--Q", type=int, default=None, help="query seq len, defaults to S")
    parser.add_argument("--S", type=int, default=1024, help="seq len")
    parser.add_argument("--H", type=int, default=1, help="attention head size")
    parser.add_argument("--q-heads", type=int, default=None, help="query head count")
    parser.add_argument("--kv-heads", type=int, default=None, help="kv head count")
    parser.add_argument("--D", type=int, default=128, help="hidden dim")
    parser.add_argument("--causal", action="store_true", help="enable causal mask")
    parser.add_argument("--no-check", action="store_true", help="disable reference check")
    parser.add_argument("--cube-s0", type=int, default=None, help="CUBE_S0, defaults to 128 when possible")
    parser.add_argument("--tile-s1", type=int, default=256, help="TILE_S1")
    parser.add_argument("--qk-preload", type=int, default=4, help="qkPreloadNum")
    parser.add_argument("--force-jit", action="store_true", help="rebuild the JIT shared library")
    parser.add_argument("--verbose-jit", action="store_true", help="print the JIT compile command")
    args = parser.parse_args()

    bsz = args.B
    q_len = args.Q or args.S
    s_len = args.S
    q_heads = args.q_heads or args.H
    kv_heads = args.kv_heads or args.H
    head_size = args.D

    if q_heads % kv_heads != 0:
        parser.error("--q-heads must be divisible by --kv-heads")

    q = torch.randn(bsz, q_heads, q_len, head_size, dtype=torch.float16)
    k = torch.randn(bsz, kv_heads, s_len, head_size, dtype=torch.float16)
    v = torch.randn(bsz, kv_heads, s_len, head_size, dtype=torch.float16)
    output = torch.empty(bsz, q_heads, q_len, head_size, dtype=torch.float32)
    print("init successful!")

    flash = jit_compile_flash(
        head_size=head_size,
        s0=q_len,
        s1=s_len,
        cube_s0=args.cube_s0,
        tile_s1=args.tile_s1,
        qk_preload=args.qk_preload,
        causal=args.causal,
        force=args.force_jit,
        verbose=args.verbose_jit,
    )

    n_rep = q_heads // kv_heads
    for batch_idx in range(bsz):
        for q_head_idx in range(q_heads):
            kv_head_idx = q_head_idx // n_rep
            out = flash(q[batch_idx, q_head_idx], k[batch_idx, kv_head_idx], v[batch_idx, kv_head_idx])
            output[batch_idx, q_head_idx].copy_(out)

    torch.npu.synchronize()

    if not args.no_check:
        ref_output = ref_flash_attn(q, k, v, args.causal)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
        print("Test Passed!")


if __name__ == "__main__":
    main()
