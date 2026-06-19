import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch_npu


def make_right_down_keep_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(q_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return kv_pos <= q_pos + (kv_len - q_len)


def make_left_up_keep_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(q_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return kv_pos <= q_pos


def native_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep_mask: torch.Tensor):
    # scale = (1.0 / q.shape[-1]) ** 0.5
    # atten_mask = ~keep_mask
    # scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    # scores = scores.masked_fill(atten_mask[None, None, :, :], torch.finfo(scores.dtype).min)
    # probs = torch.softmax(scores, dim=-1)
    # probs = probs.masked_fill(atten_mask[None, None, :, :], 0.0)
    # return torch.matmul(probs, v.float())
    


def check_fused(
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref: torch.Tensor,
    atten_mask: torch.Tensor,
    sparse_mode: int,
    expected: str,
    pre_tokens: int = 65535,
    next_tokens: int = 65535,
) -> None:
    out = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=atten_mask,
        num_heads=q.shape[1],
        scale=(1.0 / q.shape[-1]) ** 0.5,
        input_layout="BNSD",
        num_key_value_heads=k.shape[1],
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sparse_mode=sparse_mode,
    )[0].float()
    torch.npu.synchronize()
    
    print(f"{name:<42} {expected:<15} max_abs={float((out - ref).abs().max()):.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(0)
    torch.set_printoptions(threshold=1000, linewidth=150)


    b, h, q_len, kv_len, d = 1, 1, args.q_len, args.kv_len, args.head_dim
    q = torch.randn(b, h, q_len, d, dtype=torch.float16)
    k = torch.randn(b, h, kv_len, d, dtype=torch.float16)
    v = torch.randn(b, h, kv_len, d, dtype=torch.float16)

    right_down_keep_mask = make_right_down_keep_mask(q_len, kv_len, q.device)
    left_up_keep_mask = make_left_up_keep_mask(q_len, kv_len, q.device)
    full_keep_mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device)
    ref = native_attention(q, k, v, right_down_keep_mask)
    right_down_atten_mask = ~right_down_keep_mask
    left_up_atten_mask = ~left_up_keep_mask
    no_atten_mask = ~full_keep_mask
    optimized_tri_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)

    print("right-down keep mask, where 1 means keep:")
    print(right_down_keep_mask.int().cpu())
    print("right-down torch_npu atten_mask, where 1 means masked out:")
    print(right_down_atten_mask.int().cpu())
    print("left-up keep mask negative control:")
    print(left_up_keep_mask.int().cpu())
    print()

    print("Compared against the right-down causal reference:")
    check_fused("mode0 no explicit masking", q, k, v, ref, None, sparse_mode=0, expected="should differ")
    check_fused("mode0 full explicit right-down", q, k, v, ref, right_down_atten_mask, sparse_mode=0, expected="should match")
    check_fused("mode0 full explicit left-up", q, k, v, ref, left_up_atten_mask, sparse_mode=0, expected="should differ")
    check_fused("mode1 full explicit right-down", q, k, v, ref, right_down_atten_mask, sparse_mode=1, expected="should match")
    check_fused("mode1 full explicit left-up", q, k, v, ref, left_up_atten_mask, sparse_mode=1, expected="should differ")
    check_fused("mode2 optimized left-up", q, k, v, ref, optimized_tri_mask, sparse_mode=2, expected="should differ")
    check_fused("mode3 optimized right-down", q, k, v, ref, optimized_tri_mask, sparse_mode=3, expected="should match")
    check_fused(
        "mode4 band pre=kv_len next=0",
        q,
        k,
        v,
        ref,
        optimized_tri_mask,
        sparse_mode=4,
        expected="should match",
        pre_tokens=kv_len,
        next_tokens=0,
    )
    check_fused(
        "mode4 band pre=1 next=0",
        q,
        k,
        v,
        ref,
        optimized_tri_mask,
        sparse_mode=4,
        expected="should differ",
        pre_tokens=1,
        next_tokens=0,
    )


if __name__ == "__main__":
    main()


# this is using left-up mask, not the flashattention 2.1 mask we expect
# so when we increase KV(S1) seq length there is no additional work cuz there's just zeros
# however long S1 is there is always only 3 matmuls
# 1 0 0 0 0
# 1 1 0 0 0
# this is the right down causal mode =3
# 1 1 1 1 0
# 1 1 1 1 1
# return torch.nn.functional.scaled_dot_product_attention(
#         q.float(),
#         k.float(),
#         v.float(),
#         attn_mask=None,
#         dropout_p=0.0,
#         is_causal=True,
#     )

# mode0 no explicit masking                  should differ   max_abs=2.78796
# mode0 full explicit right-down             should match    max_abs=2.30664
# mode0 full explicit left-up                should differ   max_abs=0.000682592
# mode1 full explicit right-down             should match    max_abs=2.30664
# mode1 full explicit left-up                should differ   max_abs=0.000682592
# mode2 optimized left-up                    should differ   max_abs=0.000682592
# mode3 optimized right-down                 should match    max_abs=2.30664
# mode4 band pre=kv_len next=0               should match    max_abs=2.30664
# mode4 band pre=1 next=0                    should differ   max_abs=2.50748