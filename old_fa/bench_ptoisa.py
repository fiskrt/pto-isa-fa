import argparse
import csv
import glob
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


SEQ = [1 << i for i in range(10, 16)]


def attn_flops(args, b, q, s):
    pairs = q * s / (2 if args.causal else 1)
    return 4 * b * int(args.q_heads or args.H) * pairs * int(args.D)


def op_time_us(out_dir, op):
    time.sleep(2)
    files = glob.glob(os.path.join(out_dir, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv"))
    if not files:
        return None
    with open(max(files, key=os.path.getctime), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            op_type = row.get("OP Type") or row.get("Op Type") or ""
            if op_type == op or op in op_type:
                val = row.get("Total Time(us)")
                return None if not val or val == "N/A" else float(val)
    return None


def profile(script, args, s, out_dir):
    q = args.Q or s
    app = [sys.executable, script, "--B", args.B, "--Q", q, "--S", s, "--H", args.H, "--D", args.D, "--no-check"]
    if args.q_heads:
        app += ["--q-heads", args.q_heads]
    if args.kv_heads:
        app += ["--kv-heads", args.kv_heads]
    if args.causal:
        app += ["--causal"]
    if args.cube_s0:
        app += ["--cube-s0", args.cube_s0]
    app += ["--tile-s1", args.tile_s1, "--qk-preload", args.qk_preload]
    if args.force_jit:
        app += ["--force-jit"]

    subprocess.run(["msprof", f"--output={out_dir}", f"--application={shlex.join(map(str, app))}"], check=True)
    return op_time_us(out_dir, args.op)


def plot(rows, path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [(s, us / 1000, flops / us / 1e6) for _, q, s, us, flops in rows if us]
    xs, ms, tflops = zip(*data)
    fig, ax_ms = plt.subplots(figsize=(6, 4))
    ax_tf = ax_ms.twinx()
    ax_ms.plot(xs, ms, marker="o", color="tab:blue", label="ms")
    ax_tf.plot(xs, tflops, marker="s", color="tab:orange", label="TFLOPS")
    ax_ms.set_xscale("log", base=2)
    ax_ms.set_xticks(xs, [f"{s // 1024}k" for s in xs])
    ax_ms.set_xlabel("Sequence length S")
    ax_ms.set_ylabel("runTFA time (ms)", color="tab:blue")
    ax_tf.set_ylabel("TFLOPS", color="tab:orange")
    ax_ms.set_title(f"ptoisa.py B={args.B}, Q={args.Q or 'S'}, H={args.H}, D={args.D}, causal={args.causal}")
    ax_ms.grid(True, which="both", ls=":")
    ax_ms.legend(loc="upper left")
    ax_tf.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="ptoisa.py")
    p.add_argument("--B", default="1")
    p.add_argument("--Q", type=int)
    p.add_argument("--H", default="1")
    p.add_argument("--q-heads")
    p.add_argument("--kv-heads")
    p.add_argument("--D", default="128")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--cube-s0")
    p.add_argument("--tile-s1", default="256")
    p.add_argument("--qk-preload", default="4")
    p.add_argument("--force-jit", action="store_true")
    p.add_argument("--op", default="runTFA")
    p.add_argument("--log-dir", default="log")
    p.add_argument("--plot", default="ptoisa_msprof.png")
    args = p.parse_args()

    run_dir = os.path.join(args.log_dir, "ptoisa_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    rows = []
    for s in SEQ:
        q = args.Q or s
        out_dir = os.path.join(run_dir, f"S{s}")
        os.makedirs(out_dir, exist_ok=True)
        us = profile(args.script, args, s, out_dir)
        flops = attn_flops(args, int(args.B), q, s)
        rows.append((int(args.B), q, s, us, flops))
        print(
            f"B={args.B:<3} Q={q:<6} S={s:<6} {us / 1000:.3f} ms  {flops / us / 1e6:.2f} TFLOPS"
            if us
            else f"B={args.B:<3} Q={q:<6} S={s:<6} N/A"
        )

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [("B", "Q", "S", "causal", "time_us", "flops", "tflops")]
            + [(b, q, s, args.causal, us, flops, flops / us / 1e6 if us else None) for b, q, s, us, flops in rows]
        )
    if any(us for _, _, _, us, _ in rows):
        plot(rows, os.path.join(run_dir, args.plot), args)
    print(run_dir)


if __name__ == "__main__":
    main()
