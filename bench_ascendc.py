import argparse
import csv
import glob
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


SEQ = [1 << i for i in range(10, 16, 1)]
ELEM_BYTES = 2


def attn_pairs(q, s, causal):
    return q * s / (2 if causal else 1)


def batch_for_s(args, s):
    return int(args.B) * (max(SEQ) // s if args.fixed_tokens else 1)


def attn_flops(args, b, q, s):
    return 4 * b * int(args.q_heads or args.H) * attn_pairs(q, s, args.causal) * int(args.D)


def attn_bytes(args, b, q, s):
    d = int(args.D)
    qh, kvh = int(args.q_heads or args.H), int(args.kv_heads or args.H)
    return ELEM_BYTES * b * d * (2 * qh * q + 2 * kvh * s)


def title_args(args):
    qh, kvh = int(args.q_heads or args.H), int(args.kv_heads or args.H)
    b = f"{args.B}*{max(SEQ)}//S" if args.fixed_tokens else args.B
    q = args.Q or "S"
    return f"B={b}, Q={q}, H={args.H}, QH={qh}, KVH={kvh}, D={args.D}, causal={args.causal}"


def op_time_us(out_dir, op_type):
    time.sleep(2)
    files = glob.glob(os.path.join(out_dir, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv"))
    if not files:
        return None
    with open(max(files, key=os.path.getctime), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("OP Type") or row.get("Op Type")) == op_type:
                val = row.get("Total Time(us)")
                return None if not val or val == "N/A" else float(val)
    return None


def profile(script, args, s, out_dir):
    b = batch_for_s(args, s)
    q = args.Q or s
    app = [sys.executable, script, "--B", b, "--Q", q, "--S", s, "--H", args.H, "--D", args.D, "--no-check"]
    if args.q_heads:
        app += ["--q-heads", args.q_heads]
    if args.kv_heads:
        app += ["--kv-heads", args.kv_heads]
    if args.causal:
        app += ["--causal"]
    cmd = ["msprof", f"--output={out_dir}", f"--application={shlex.join(map(str, app))}"]
    subprocess.run(cmd, check=True)
    return op_time_us(out_dir, args.op)


def plot(rows, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ms, tflops = zip(*[(s, us / 1000, flops / us / 1e6) for _, _, s, us, flops, _ in rows if us])
    fig, ax_ms = plt.subplots(figsize=(6, 4))
    ax_tflops = ax_ms.twinx()
    ax_ms.plot(xs, ms, marker="o", color="tab:blue", label="ms")
    ax_tflops.plot(xs, tflops, marker="s", color="tab:orange", label="TFLOPS")
    ax_ms.set_xscale("log", base=2)
    ax_ms.set_xticks(xs, [f"{s // 1024}k" for s in xs])
    ax_ms.set_xlabel("Sequence length S")
    ax_ms.set_ylabel("FlashAttentionScore time (ms)", color="tab:blue")
    ax_tflops.set_ylabel("TFLOPS", color="tab:orange")
    ax_ms.set_title(title_args(args))
    ax_ms.grid(True, which="both", ls=":")
    ax_ms.legend(loc="upper left")
    ax_tflops.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)


def plot_bw(rows, path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, tbps = zip(*[(s, bytes_ / us / 1e6) for _, _, s, us, _, bytes_ in rows if us])
    plt.figure(figsize=(6, 4))
    plt.plot(xs, tbps, marker="o", color="tab:green")
    plt.xscale("log", base=2)
    plt.xticks(xs, [f"{s // 1024}k" for s in xs])
    plt.xlabel("Sequence length S")
    plt.ylabel("Estimated bandwidth (TB/s)")
    plt.title(title_args(args) + "\nBW assumes fp16 bytes = 2*B*D*(2*QH*Q + 2*KVH*S)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(path, dpi=180)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="ascendc.py")
    p.add_argument("--B", default="1")
    p.add_argument("--Q", default=None)
    p.add_argument("--H", default="1")
    p.add_argument("--q-heads")
    p.add_argument("--kv-heads")
    p.add_argument("--D", default="128")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--fixed-tokens", "--fixed-batch", action="store_true", dest="fixed_tokens")
    p.add_argument("--op", default="FlashAttentionScore")
    p.add_argument("--log-dir", default="log")
    p.add_argument("--plot", default="npu_fusion_attention_msprof.png")
    p.add_argument("--bw-plot", default="npu_fusion_attention_bandwidth.png")
    args = p.parse_args()
    if args.fixed_tokens and args.Q is not None:
        p.error("--fixed-tokens/--fixed-batch is not compatible with setting --Q")

    run_dir = os.path.join(args.log_dir, "npu_fusion_attention_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rows = []
    for s in SEQ:
        b = batch_for_s(args, s)
        q = int(args.Q or s)
        out_dir = os.path.join(run_dir, f"S{s}")
        os.makedirs(out_dir, exist_ok=True)
        us = profile(args.script, args, s, out_dir)
        flops = attn_flops(args, b, q, s)
        bytes_ = attn_bytes(args, b, q, s)
        rows.append((b, q, s, us, flops, bytes_))
        print(
            f"B={b:<3} Q={q:<6} S={s:<6} {us / 1000:.3f} ms  {flops / us / 1e6:.2f} TFLOPS  {bytes_ / us / 1e6:.2f} TB/s"
            if us
            else f"B={b:<3} Q={q:<6} S={s:<6} N/A"
        )

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [("B", "Q", "S", "causal", "time_us", "flops", "tflops", "bytes", "bandwidth_TBps")]
            + [
                (b, q, s, args.causal, us, flops, flops / us / 1e6 if us else None, bytes_, bytes_ / us / 1e6 if us else None)
                for b, q, s, us, flops, bytes_ in rows
            ]
        )
    if any(us for _, _, _, us, _, _ in rows):
        plot(rows, os.path.join(run_dir, args.plot), args)
        plot_bw(rows, os.path.join(run_dir, args.bw_plot), args)
    print(run_dir)


if __name__ == "__main__":
    main()
