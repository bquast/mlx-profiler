# mlx-profiler

Operation-level profiler for Apple Silicon / MLX, a  `torch.profiler` equivalent for the MLX ecosystem.

## features

- **Op-level timing** — every `matmul`, `softmax`, `rms_norm`, layer call, etc., measured individually
- **FLOPs estimation** — multiply-add counts for matmuls, convolutions
- **Arithmetic intensity** — FLOPs/byte ratio per op (roofline model X-axis)
- **Memory bandwidth estimate** — tensor traffic through each operation
- **Device attribution** — GPU / CPU / Neural Engine breakdown
- **Terminal report** — human-readable ANSI table
- **Interactive HTML dashboard** — flame timeline, roofline scatter, searchable op table, category donut
- **Trace persistence** — save/load JSON, render HTML from CLI

## install

```bash
pip install mlx-profiler          # without MLX (manual recording only)
pip install "mlx-profiler[mlx]"   # with MLX for automatic interception
```

## quick start

```python
import mlx.core as mx
import mlx.nn as nn
import mlx_profiler as mp

model = MyTransformer()
x = mx.random.normal([1, 512])

with mp.profile("my_model") as prof:
    out = model(x)
    mx.eval(out)

prof.report()              # terminal output
prof.html("report.html")   # interactive dashboard
prof.save("trace.json")    # save for later
```

## CLI

```bash
# Generate a demo trace + report
mlx-profiler demo

# View a saved trace in the terminal
mlx-profiler view trace.json

# Render HTML from a saved trace
mlx-profiler html trace.json -o report.html
```

## manual op recording (no MLX required)

```python
from mlx_profiler.trace import Trace
from mlx_profiler.profiler import op_timer
import mlx_profiler as mp

trace = Trace("my_trace")

with op_timer(trace, "matmul",
              input_shapes=[[512, 4096], [4096, 4096]],
              output_shapes=[[512, 4096]],
              dtype="float16"):
    result = my_matmul(a, b)

mp.print_report(trace)
```

## output

| Op | FLOPs formula | Notes |
|---|---|---|
| `matmul` / `linear` | 2·M·K·N | multiply-add counted as 2 ops |
| `conv2d` | 2·Ho·Wo·Co·Ci·Kh·Kw | per batch element |
| All ops | input + output tensor bytes | memory bandwidth estimate |

## design notes

**Why wall-clock, not Metal GPU timers?**

Metal's `MTLCommandBuffer.GPUStartTime` gives you true on-chip execution time but requires you to instrument command buffers — not feasible when wrapping MLX's lazy evaluation model. Wall-clock with a forced `mx.eval()` boundary gives you "latency as the model experiences it", which is the number that matters for interactive applications. A future `metal_backend` module can add real GPU timers via Metal's performance counter API once the graph is materialized.

**Lazy evaluation and timing**

MLX uses lazy evaluation — operations aren't executed until `mx.eval()` is called. The profiler inserts `mx.eval()` calls at layer boundaries to force materialization and get accurate per-layer timing. This means profiling adds overhead; real-world timings will be slightly slower than unproduced runs.

## roadmap

- [ ] Metal GPU timestamp counters (true on-chip time)
- [ ] Neural Engine vs GPU attribution via `os_signpost`
- [ ] Continuous batching aware profiling
- [ ] Multi-pass aggregation (average over N forward passes)
- [ ] FlashAttention kernel detection
- [ ] TurboQuant / quantized op analysis
- [ ] Comparison mode: profile A vs profile B

## license

[MIT](https://opensource.org/license/mit)
