"""
Example 1: Profile an mlx.nn.Module (requires MLX installed)
"""

# ─── With MLX ────────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_profiler as mp

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 2048)
            self.fc2 = nn.Linear(2048, 512)

        def __call__(self, x):
            return self.fc2(nn.gelu(self.fc1(x)))

    model = MLP()
    x = mx.random.normal([8, 512])

    # Option A: context manager
    with mp.profile("mlp_forward") as prof:
        out = model(x)
        mx.eval(out)

    prof.report()
    prof.html("mlp_report.html")
    prof.save("mlp_trace.json")

    # Option B: Profiler wrapper
    wrapped = mp.Profiler(model, name="mlp")
    out = wrapped(x)
    wrapped.report()

except ImportError:
    print("MLX not installed — run: pip install mlx")


# ─── Without MLX (manual op recording) ───────────────────────────────────────
import time
import mlx_profiler as mp
from mlx_profiler.trace import Trace
from mlx_profiler.profiler import op_timer

trace = Trace("manual_example")

with op_timer(trace, "matmul",
              input_shapes=[[512, 4096], [4096, 4096]],
              output_shapes=[[512, 4096]],
              dtype="float16", device="gpu"):
    time.sleep(0.004)  # simulate 4ms matmul

with op_timer(trace, "softmax",
              input_shapes=[[8, 32, 512, 512]],
              output_shapes=[[8, 32, 512, 512]],
              dtype="float16", device="gpu"):
    time.sleep(0.001)

mp.print_report(trace)


# ─── Load and re-render a saved trace ────────────────────────────────────────
# from mlx_profiler.trace import Trace
# trace = Trace.load("my_trace.json")
# from mlx_profiler.report import render_html, print_report
# print_report(trace)
# render_html(trace, "my_report.html")
