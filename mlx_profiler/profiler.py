"""
Profiler core: intercepts MLX operations and records timing + metadata.

Two interception strategies:
  1. mx.eval() hook  — wraps mx.eval so we can force materialization and
     record how long evaluation of the pending graph took.
  2. nn.Module wrapping — wraps forward() of each submodule so we can
     record layer-level timings with shape info.

We deliberately keep Metal/low-level APIs out of the hot path.  Timing is
wall-clock (time.perf_counter_ns) which is "good enough" for the profiling
use-case while being completely portable.
"""

from __future__ import annotations

import time
import functools
import contextlib
import platform
import subprocess
from typing import Any, Optional, Generator
from contextlib import contextmanager

from .trace import Trace, OpRecord

# ── Optional MLX import ───────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    mx = None
    nn = None
    HAS_MLX = False


# ── Op category table ─────────────────────────────────────────────────────────
_OP_CATEGORIES = {
    # compute-heavy
    "matmul": "compute",
    "linear": "compute",
    "conv2d": "compute",
    "conv1d": "compute",
    "batch_norm": "compute",
    "layer_norm": "compute",
    "rms_norm": "compute",
    "group_norm": "compute",
    "attention": "compute",
    "scaled_dot_product_attention": "compute",
    # activations
    "relu": "activation",
    "gelu": "activation",
    "silu": "activation",
    "sigmoid": "activation",
    "softmax": "activation",
    "tanh": "activation",
    # memory / reshape
    "reshape": "memory",
    "transpose": "memory",
    "concatenate": "memory",
    "split": "memory",
    "slice": "memory",
    "pad": "memory",
    "copy": "memory",
    # quantize
    "quantize": "quantize",
    "dequantize": "quantize",
    "quantized_matmul": "quantize",
    # embedding
    "embedding": "embedding",
    # misc
    "add": "elementwise",
    "multiply": "elementwise",
    "divide": "elementwise",
    "subtract": "elementwise",
    "exp": "elementwise",
    "log": "elementwise",
    "sqrt": "elementwise",
    "mean": "reduction",
    "sum": "reduction",
    "max": "reduction",
    "min": "reduction",
}

_DEVICE_MAP = {
    "gpu": "gpu",
    "cpu": "cpu",
}


def _guess_category(name: str) -> str:
    return _OP_CATEGORIES.get(name.lower(), "other")


def _get_chip_info() -> dict:
    info = {"chip": "unknown", "memory_gb": 0, "os": platform.system()}
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if "Chip" in line:
                info["chip"] = line.split(":")[-1].strip()
            if "Memory" in line and "GB" in line:
                try:
                    info["memory_gb"] = int(line.split()[-2])
                except Exception:
                    pass
    except Exception:
        pass
    return info


class ProfileContext:
    """The object returned by the `profile()` context manager."""

    def __init__(self, trace: Trace):
        self.trace = trace

    def report(self):
        from .report import print_report
        print_report(self.trace)

    def save(self, path: str):
        self.trace.save(path)
        print(f"Trace saved to {path}")

    def html(self, path: str):
        from .report import render_html
        render_html(self.trace, path)
        print(f"HTML report saved to {path}")

    def __repr__(self):
        return (
            f"<ProfileContext '{self.trace.name}' "
            f"{len(self.trace.ops)} ops, "
            f"{self.trace.total_duration_ms:.1f} ms>"
        )


class Profiler:
    """
    Wraps an nn.Module (or any callable) and profiles every forward pass.

    Example:
        profiler = Profiler(model)
        out = profiler(x)
        profiler.context.report()
    """

    def __init__(self, module, name: str = "model"):
        self._module = module
        self._name = name
        self.context: Optional[ProfileContext] = None
        self._patched = False

    def __call__(self, *args, **kwargs):
        with profile(self._name) as ctx:
            if HAS_MLX and nn is not None and isinstance(self._module, nn.Module):
                _patch_module(self._module, ctx.trace)
                try:
                    out = self._module(*args, **kwargs)
                    if HAS_MLX:
                        mx.eval(out)
                finally:
                    _unpatch_module(self._module)
            else:
                out = self._module(*args, **kwargs)
        self.context = ctx
        return out

    def report(self):
        if self.context:
            self.context.report()

    def html(self, path: str):
        if self.context:
            self.context.html(path)


@contextmanager
def profile(name: str = "trace") -> Generator[ProfileContext, None, None]:
    """
    Context manager that profiles all MLX operations within its scope.

    Usage:
        with mlx_profiler.profile("attention_block") as p:
            out = block(x)
        p.report()
    """
    trace = Trace(name=name)
    trace.metadata["chip"] = _get_chip_info()
    ctx = ProfileContext(trace)

    if HAS_MLX:
        _install_mx_hooks(trace)

    try:
        yield ctx
    finally:
        if HAS_MLX:
            # Force evaluation of any pending graph before stopping
            try:
                mx.eval()
            except Exception:
                pass
            _remove_mx_hooks(trace)


# ── mx.eval hook ─────────────────────────────────────────────────────────────

_original_mx_eval = None
_active_traces: list[Trace] = []


def _install_mx_hooks(trace: Trace):
    global _original_mx_eval, _active_traces
    _active_traces.append(trace)
    if _original_mx_eval is None and HAS_MLX:
        _original_mx_eval = mx.eval

        def _patched_eval(*args, **kwargs):
            # Record the evaluation event
            start = time.perf_counter_ns()
            result = _original_mx_eval(*args, **kwargs)
            end = time.perf_counter_ns()

            # Try to extract shape/dtype from args
            in_shapes = []
            out_shapes = []
            dtype = "unknown"
            for a in args:
                if HAS_MLX and hasattr(a, 'shape'):
                    in_shapes.append(list(a.shape))
                    dtype = str(getattr(a, 'dtype', 'unknown')).replace('mlx.core.', '')
                elif isinstance(a, (list, tuple)):
                    for item in a:
                        if hasattr(item, 'shape'):
                            out_shapes.append(list(item.shape))

            op = OpRecord(
                name="mx.eval",
                category="compute",
                start_ns=start,
                end_ns=end,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                dtype=dtype,
                device="gpu",
            )
            for t in _active_traces:
                t.add(op)
            return result

        mx.eval = _patched_eval


def _remove_mx_hooks(trace: Trace):
    global _original_mx_eval, _active_traces
    if trace in _active_traces:
        _active_traces.remove(trace)
    if not _active_traces and _original_mx_eval is not None and HAS_MLX:
        mx.eval = _original_mx_eval
        _original_mx_eval = None


# ── nn.Module layer patching ──────────────────────────────────────────────────

_PATCHED_ATTR = "_mlxprof_original_call"


def _patch_module(module, trace: Trace, prefix: str = ""):
    """Recursively wrap each submodule's __call__ to record timing."""
    if not HAS_MLX or not isinstance(module, nn.Module):
        return

    for name, child in module.named_modules():
        full_name = f"{prefix}/{name}" if prefix else name
        _wrap_layer(child, full_name, trace)


def _wrap_layer(layer, name: str, trace: Trace):
    if hasattr(layer, _PATCHED_ATTR):
        return  # already patched

    original_call = layer.__call__

    @functools.wraps(original_call)
    def _timed_call(*args, **kwargs):
        start = time.perf_counter_ns()
        out = original_call(*args, **kwargs)

        # Force eval to get real timing
        if HAS_MLX:
            try:
                if isinstance(out, (list, tuple)):
                    mx.eval(*out)
                else:
                    mx.eval(out)
            except Exception:
                pass

        end = time.perf_counter_ns()

        in_shapes = []
        dtype = "unknown"
        for a in args:
            if hasattr(a, 'shape'):
                in_shapes.append(list(a.shape))
                raw_dtype = str(getattr(a, 'dtype', 'unknown'))
                dtype = raw_dtype.split('.')[-1]

        out_shapes = []
        if isinstance(out, (list, tuple)):
            for o in out:
                if hasattr(o, 'shape'):
                    out_shapes.append(list(o.shape))
        elif hasattr(out, 'shape'):
            out_shapes = [list(out.shape)]

        layer_type = type(layer).__name__.lower()
        op_name = layer_type if layer_type != "module" else name.split("/")[-1]

        op = OpRecord(
            name=op_name,
            category=_guess_category(op_name),
            start_ns=start,
            end_ns=end,
            input_shapes=in_shapes,
            output_shapes=out_shapes,
            dtype=dtype,
            device="gpu",
            metadata={"layer_path": name},
        )
        trace.add(op)
        return out

    setattr(layer, _PATCHED_ATTR, original_call)
    layer.__call__ = _timed_call


def _unpatch_module(module):
    if not HAS_MLX or not isinstance(module, nn.Module):
        return
    for _, child in module.named_modules():
        if hasattr(child, _PATCHED_ATTR):
            child.__call__ = getattr(child, _PATCHED_ATTR)
            delattr(child, _PATCHED_ATTR)


# ── Manual op recording (for use without MLX) ────────────────────────────────

class op_timer:
    """
    Manual context manager for recording a single operation.

    Usage:
        with mlx_profiler.op_timer(trace, "custom_attention", input_shapes=[[4,512,512]]):
            result = my_custom_attention(q, k, v)
    """

    def __init__(
        self,
        trace: Trace,
        name: str,
        input_shapes: list = None,
        output_shapes: list = None,
        dtype: str = "float16",
        device: str = "gpu",
        category: str = None,
        metadata: dict = None,
    ):
        self.trace = trace
        self.name = name
        self.input_shapes = input_shapes or []
        self.output_shapes = output_shapes or []
        self.dtype = dtype
        self.device = device
        self.category = category or _guess_category(name)
        self.metadata = metadata or {}
        self._start = 0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *_):
        end = time.perf_counter_ns()
        op = OpRecord(
            name=self.name,
            category=self.category,
            start_ns=self._start,
            end_ns=end,
            input_shapes=self.input_shapes,
            output_shapes=self.output_shapes,
            dtype=self.dtype,
            device=self.device,
            metadata=self.metadata,
        )
        self.trace.add(op)
