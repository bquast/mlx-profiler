#!/usr/bin/env python3
"""
mlx-profiler CLI

Usage:
  mlx-profiler view trace.json
  mlx-profiler html trace.json -o report.html
  mlx-profiler run myscript.py --html
  mlx-profiler demo
"""

import argparse
import sys
import os


def cmd_view(args):
    from mlx_profiler.trace import Trace
    from mlx_profiler.report import print_report
    trace = Trace.load(args.trace)
    print_report(trace)


def cmd_html(args):
    from mlx_profiler.trace import Trace
    from mlx_profiler.report import render_html
    trace = Trace.load(args.trace)
    out = args.output or args.trace.replace(".json", ".html")
    render_html(trace, out)
    print(f"HTML report: {out}")


def cmd_demo(args):
    """Run a synthetic demo trace to show what the profiler output looks like."""
    import time
    import random
    from mlx_profiler.trace import Trace, OpRecord
    from mlx_profiler.report import print_report, render_html

    random.seed(42)
    trace = Trace(name="demo_transformer_forward")
    trace.metadata = {
        "chip": {"chip": "Apple M3 Max", "memory_gb": 128, "os": "Darwin"},
        "model": "Qwen-7B-Instruct-4bit",
        "batch_size": 1,
        "seq_len": 512,
    }

    t = 1_000_000  # start ns

    ops_spec = [
        # (name, category, dur_us, in_shapes, out_shapes, dtype, device)
        ("embedding", "embedding", 320, [[1, 512]], [[1, 512, 4096]], "float16", "gpu"),
        ("rms_norm", "compute", 85, [[1, 512, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 4100, [[1, 512, 4096], [4096, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 4050, [[1, 512, 4096], [4096, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 4200, [[1, 512, 4096], [4096, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("reshape", "memory", 12, [[1, 512, 4096]], [[1, 32, 512, 128]], "float16", "gpu"),
        ("transpose", "memory", 18, [[1, 32, 512, 128]], [[1, 32, 128, 512]], "float16", "gpu"),
        ("matmul", "compute", 6800, [[1, 32, 512, 128], [1, 32, 128, 512]], [[1, 32, 512, 512]], "float16", "gpu"),
        ("multiply", "elementwise", 45, [[1, 32, 512, 512]], [[1, 32, 512, 512]], "float16", "gpu"),
        ("softmax", "activation", 280, [[1, 32, 512, 512]], [[1, 32, 512, 512]], "float16", "gpu"),
        ("matmul", "compute", 6500, [[1, 32, 512, 512], [1, 32, 512, 128]], [[1, 32, 512, 128]], "float16", "gpu"),
        ("reshape", "memory", 14, [[1, 32, 512, 128]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 4300, [[1, 512, 4096], [4096, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("add", "elementwise", 38, [[1, 512, 4096], [1, 512, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("rms_norm", "compute", 82, [[1, 512, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 9800, [[1, 512, 4096], [4096, 11008]], [[1, 512, 11008]], "float16", "gpu"),
        ("silu", "activation", 120, [[1, 512, 11008]], [[1, 512, 11008]], "float16", "gpu"),
        ("linear", "compute", 9700, [[1, 512, 4096], [4096, 11008]], [[1, 512, 11008]], "float16", "gpu"),
        ("multiply", "elementwise", 48, [[1, 512, 11008], [1, 512, 11008]], [[1, 512, 11008]], "float16", "gpu"),
        ("linear", "compute", 9900, [[1, 512, 11008], [11008, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("add", "elementwise", 40, [[1, 512, 4096], [1, 512, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("rms_norm", "compute", 80, [[1, 512, 4096]], [[1, 512, 4096]], "float16", "gpu"),
        ("linear", "compute", 2200, [[1, 512, 4096], [4096, 32000]], [[1, 512, 32000]], "float16", "gpu"),
        ("softmax", "activation", 350, [[1, 512, 32000]], [[1, 512, 32000]], "float16", "gpu"),
        ("quantized_matmul", "quantize", 3800, [[1, 512, 4096], [4096, 4096]], [[1, 512, 4096]], "int4", "gpu"),
        ("dequantize", "quantize", 420, [[4096, 512]], [[4096, 512]], "int4", "gpu"),
        ("mx.eval", "compute", 1200, [[1, 512, 32000]], [], "float16", "gpu"),
    ]

    for name, cat, dur_us, in_s, out_s, dtype, dev in ops_spec:
        jitter = random.gauss(1.0, 0.12)
        actual_dur = max(10, int(dur_us * jitter))
        op = OpRecord(
            name=name,
            category=cat,
            start_ns=t,
            end_ns=t + actual_dur * 1000,
            input_shapes=in_s,
            output_shapes=out_s,
            dtype=dtype,
            device=dev,
        )
        trace.ops.append(op)
        t += actual_dur * 1000 + random.randint(1000, 5000)

    print_report(trace)

    out_html = "mlx_profiler_demo.html"
    render_html(trace, out_html)
    print(f"\nDemo HTML report saved to: {os.path.abspath(out_html)}")

    out_json = "mlx_profiler_demo.json"
    trace.save(out_json)
    print(f"Demo trace saved to:       {os.path.abspath(out_json)}")


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-profiler",
        description="MLX Profiler — operation-level profiling for Apple Silicon"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_view = sub.add_parser("view", help="Print terminal report from saved trace")
    p_view.add_argument("trace", help="Path to .json trace file")

    p_html = sub.add_parser("html", help="Render HTML dashboard from trace")
    p_html.add_argument("trace", help="Path to .json trace file")
    p_html.add_argument("-o", "--output", help="Output .html path")

    p_demo = sub.add_parser("demo", help="Run a synthetic demo and show output")

    args = parser.parse_args()

    if args.cmd == "view":
        cmd_view(args)
    elif args.cmd == "html":
        cmd_html(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
