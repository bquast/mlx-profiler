"""
Example: Manual op recording without MLX installed.

Useful when:
  - You're on a non-Apple machine testing your pipeline logic
  - You want to profile a custom kernel, CoreML call, or C extension
  - You want to include Python-side preprocessing in the trace
  - You want to mock a model's op graph before running it on real hardware

Run with:
    python examples/manual_recording.py

No MLX required — numpy is used purely to simulate realistic latency.
Shape and dtype metadata passed to op_timer are what matter for the report.
"""

import time
import random
import platform
import numpy as np

import mlx_profiler as mp
from mlx_profiler.trace import Trace
from mlx_profiler.profiler import op_timer


def run_transformer_layer(trace: Trace, seq_len=512, dim=4096, n_heads=32):
    B, H, D = 1, n_heads, dim // n_heads
    ffn = dim * 4

    # Latency budgets per category (microseconds), loosely based on M3 Max measurements
    LAT = {"compute": 4000, "quantize": 3500, "activation": 250,
           "elementwise": 40, "memory": 15, "embedding": 300}

    def sim(name, cat, in_shapes, out_shapes, dtype="float16", device="gpu", **meta):
        """Record one op: sleep for a jittered latency then add to trace."""
        us = LAT.get(cat, 200) * random.gauss(1.0, 0.08)
        with op_timer(trace, name, input_shapes=in_shapes, output_shapes=out_shapes,
                      dtype=dtype, device=device, category=cat, metadata=meta):
            time.sleep(max(us, 5) / 1e6)

    sim("embedding",       "embedding",   [[seq_len]],            [[B, seq_len, dim]])
    sim("rms_norm",        "compute",     [[B, seq_len, dim]],    [[B, seq_len, dim]])
    sim("linear_q",        "compute",     [[B, seq_len, dim], [dim, dim]], [[B, seq_len, dim]], projection="Q")
    sim("linear_k",        "compute",     [[B, seq_len, dim], [dim, dim]], [[B, seq_len, dim]], projection="K")
    sim("linear_v",        "compute",     [[B, seq_len, dim], [dim, dim]], [[B, seq_len, dim]], projection="V")
    sim("matmul",          "compute",     [[B, H, seq_len, D], [B, H, D, seq_len]], [[B, H, seq_len, seq_len]], op="QK^T")
    sim("softmax",         "activation",  [[B, H, seq_len, seq_len]], [[B, H, seq_len, seq_len]])
    sim("matmul",          "compute",     [[B, H, seq_len, seq_len], [B, H, seq_len, D]], [[B, seq_len, dim]], op="AV")
    sim("linear_out",      "compute",     [[B, seq_len, dim], [dim, dim]], [[B, seq_len, dim]])
    sim("add",             "elementwise", [[B, seq_len, dim], [B, seq_len, dim]], [[B, seq_len, dim]])
    sim("rms_norm",        "compute",     [[B, seq_len, dim]],    [[B, seq_len, dim]])
    sim("quantized_matmul","quantize",    [[B, seq_len, dim], [dim, ffn]], [[B, seq_len, ffn]], dtype="int4", role="gate")
    sim("silu",            "activation",  [[B, seq_len, ffn]],    [[B, seq_len, ffn]])
    sim("quantized_matmul","quantize",    [[B, seq_len, dim], [dim, ffn]], [[B, seq_len, ffn]], dtype="int4", role="up")
    sim("multiply",        "elementwise", [[B, seq_len, ffn], [B, seq_len, ffn]], [[B, seq_len, ffn]])
    sim("quantized_matmul","quantize",    [[B, seq_len, ffn], [ffn, dim]], [[B, seq_len, dim]], dtype="int4", role="down")
    sim("add",             "elementwise", [[B, seq_len, dim], [B, seq_len, dim]], [[B, seq_len, dim]])
    # LM head typically dispatches to CPU in MLX inference stacks
    sim("linear_lm_head",  "compute",     [[B, seq_len, dim], [dim, 32000]], [[B, seq_len, 32000]], device="cpu")
    sim("softmax",         "activation",  [[B, seq_len, 32000]], [[B, seq_len, 32000]], dtype="float32", device="cpu")


if __name__ == "__main__":
    random.seed(42)

    trace = Trace("manual_transformer_layer")
    trace.metadata = {
        "chip": {
            "chip": platform.processor() or "Apple Silicon (simulation)",
            "memory_gb": "?",
            "os": platform.system(),
        },
        "note": "Manual recording — shapes are accurate, timing is simulated",
    }

    print("Recording transformer layer (simulated timings)...")
    t0 = time.perf_counter()
    run_transformer_layer(trace, seq_len=512, dim=4096, n_heads=32)
    print(f"Done in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    mp.print_report(trace)
    trace.save("manual_trace.json")
    mp.render_html(trace, "manual_report.html")
    print("Saved: manual_trace.json, manual_report.html")
