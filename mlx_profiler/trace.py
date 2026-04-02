"""
Data structures for storing profiling traces.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class OpRecord:
    """A single recorded MLX operation."""
    name: str                        # e.g. "matmul", "linear", "softmax"
    category: str                    # "compute", "memory", "quantize", "io"
    start_ns: int                    # wall-clock start (nanoseconds)
    end_ns: int                      # wall-clock end (nanoseconds)
    input_shapes: list[list[int]]    # shapes of input tensors
    output_shapes: list[list[int]]   # shapes of output tensors
    dtype: str                       # "float16", "bfloat16", "float32", "int4" ...
    device: str                      # "gpu", "cpu", "neural_engine", "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

    def flops_estimate(self) -> Optional[int]:
        """
        Best-effort FLOPs estimation for common ops.
        Returns None for ops where we can't estimate.
        """
        if self.name in ("matmul", "linear"):
            if len(self.input_shapes) >= 2:
                a, b = self.input_shapes[0], self.input_shapes[1]
                # MxK @ KxN = 2*M*K*N multiply-adds
                if len(a) >= 2 and len(b) >= 2:
                    M = a[-2]
                    K = a[-1]
                    N = b[-1]
                    batch = 1
                    for d in a[:-2]:
                        batch *= d
                    return 2 * batch * M * K * N
        if self.name in ("conv2d",):
            # rough: 2 * out_h * out_w * out_c * in_c * kh * kw
            if self.output_shapes and self.input_shapes:
                o = self.output_shapes[0]
                i = self.input_shapes[0]
                k = self.metadata.get("kernel_size", [3, 3])
                if len(o) == 4:
                    return 2 * o[1] * o[2] * o[3] * i[1] * k[0] * k[1]
        return None

    def memory_bytes(self) -> int:
        """Estimate bytes transferred through this op."""
        dtype_bytes = {
            "float32": 4, "float16": 2, "bfloat16": 2,
            "int8": 1, "int4": 0.5, "uint8": 1,
        }.get(self.dtype, 4)

        total = 0
        for shape in self.input_shapes + self.output_shapes:
            elems = 1
            for d in shape:
                elems *= d
            total += elems * dtype_bytes
        return int(total)

    def arithmetic_intensity(self) -> Optional[float]:
        """FLOPs / bytes — Roofline model X-axis."""
        flops = self.flops_estimate()
        mem = self.memory_bytes()
        if flops is not None and mem > 0:
            return flops / mem
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_us"] = self.duration_us
        d["flops"] = self.flops_estimate()
        d["memory_bytes"] = self.memory_bytes()
        d["arithmetic_intensity"] = self.arithmetic_intensity()
        return d


@dataclass
class Trace:
    """A complete profiling trace."""
    name: str
    start_time: float = field(default_factory=time.time)
    ops: list[OpRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, op: OpRecord):
        self.ops.append(op)

    @property
    def total_duration_ms(self) -> float:
        if not self.ops:
            return 0.0
        earliest = min(o.start_ns for o in self.ops)
        latest = max(o.end_ns for o in self.ops)
        return (latest - earliest) / 1e6

    def by_category(self) -> dict[str, list[OpRecord]]:
        cats: dict[str, list[OpRecord]] = {}
        for op in self.ops:
            cats.setdefault(op.category, []).append(op)
        return cats

    def by_name(self) -> dict[str, list[OpRecord]]:
        names: dict[str, list[OpRecord]] = {}
        for op in self.ops:
            names.setdefault(op.name, []).append(op)
        return names

    def top_ops(self, n: int = 10) -> list[tuple[str, float, int]]:
        """Return (op_name, total_ms, count) sorted by total time."""
        by_name = self.by_name()
        results = []
        for name, ops in by_name.items():
            total = sum(o.duration_ms for o in ops)
            results.append((name, total, len(ops)))
        return sorted(results, key=lambda x: x[1], reverse=True)[:n]

    def device_breakdown(self) -> dict[str, float]:
        """Total ms per device."""
        breakdown: dict[str, float] = {}
        for op in self.ops:
            breakdown[op.device] = breakdown.get(op.device, 0) + op.duration_ms
        return breakdown

    def total_flops(self) -> int:
        total = 0
        for op in self.ops:
            f = op.flops_estimate()
            if f:
                total += f
        return total

    def total_memory_bytes(self) -> int:
        return sum(op.memory_bytes() for op in self.ops)

    def save(self, path: str):
        data = {
            "name": self.name,
            "start_time": self.start_time,
            "metadata": self.metadata,
            "ops": [op.to_dict() for op in self.ops],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path) as f:
            data = json.load(f)
        trace = cls(name=data["name"], start_time=data.get("start_time", 0))
        trace.metadata = data.get("metadata", {})
        for od in data.get("ops", []):
            op = OpRecord(
                name=od["name"],
                category=od["category"],
                start_ns=od["start_ns"],
                end_ns=od["end_ns"],
                input_shapes=od["input_shapes"],
                output_shapes=od["output_shapes"],
                dtype=od["dtype"],
                device=od["device"],
                metadata=od.get("metadata", {}),
            )
            trace.ops.append(op)
        return trace
