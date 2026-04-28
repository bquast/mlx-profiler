"""
mlx-profiler: Operation-level profiling for Apple Silicon / MLX

Usage:
    import mlx_profiler as mp
    import mlx.core as mx

    with mp.profile("my_model") as prof:
        output = model(input)

    prof.report()          # print to terminal
    prof.save("trace.json")
    prof.html("report.html")
"""

from .profiler import Profiler, profile, ProfileContext
from .report import print_report, render_html
from .trace import Trace, OpRecord

__all__ = [
    "Profiler",
    "profile",
    "ProfileContext",
    "print_report",
    "render_html",
    "Trace",
    "OpRecord",
]

__version__ = "0.1.0"

