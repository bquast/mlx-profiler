"""
Report rendering: terminal (ANSI) and HTML dashboard.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trace import Trace


# ── Terminal report ───────────────────────────────────────────────────────────

def _fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms/1000:.2f} s"


def _fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    return f"{b/1024**3:.2f} GB"


def _fmt_flops(f: int) -> str:
    if f < 1e3:
        return str(f)
    if f < 1e6:
        return f"{f/1e3:.1f}K"
    if f < 1e9:
        return f"{f/1e6:.1f}M"
    if f < 1e12:
        return f"{f/1e9:.2f}G"
    return f"{f/1e12:.2f}T"


def print_report(trace: "Trace"):
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    RED = "\033[31m"

    total_ms = trace.total_duration_ms
    total_flops = trace.total_flops()
    total_mem = trace.total_memory_bytes()

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD} MLX Profiler — {trace.trace if hasattr(trace,'trace') else trace.name}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    chip = trace.metadata.get("chip", {})
    if isinstance(chip, dict) and chip.get("chip") != "unknown":
        print(f"  Chip:   {chip.get('chip','')}")
        if chip.get("memory_gb"):
            print(f"  Memory: {chip.get('memory_gb')} GB unified")
    print(f"  Ops:    {len(trace.ops)}")
    print(f"  Total:  {_fmt_ms(total_ms)}")
    if total_flops:
        gflops = total_flops / total_ms / 1e6 if total_ms else 0
        print(f"  FLOPs:  {_fmt_flops(total_flops)} ({gflops:.1f} GFLOPS)")
    print(f"  Memory: {_fmt_bytes(total_mem)} estimated bandwidth")
    print()

    # Device breakdown
    dev = trace.device_breakdown()
    if len(dev) > 1:
        print(f"{BOLD}  Device breakdown:{RESET}")
        for d, ms in sorted(dev.items(), key=lambda x: x[1], reverse=True):
            pct = (ms / total_ms * 100) if total_ms else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {d:<18} {bar} {pct:5.1f}%  {_fmt_ms(ms)}")
        print()

    # Category breakdown
    cats = trace.by_category()
    if cats:
        print(f"{BOLD}  By category:{RESET}")
        for cat, ops in sorted(cats.items(), key=lambda x: sum(o.duration_ms for o in x[1]), reverse=True):
            ms = sum(o.duration_ms for o in ops)
            pct = (ms / total_ms * 100) if total_ms else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {cat:<18} {bar} {pct:5.1f}%  {_fmt_ms(ms)}  ({len(ops)} calls)")
        print()

    # Top ops table
    top = trace.top_ops(15)
    if top:
        print(f"{BOLD}  Top operations:{RESET}")
        print(f"  {'Operation':<28} {'Total':>10} {'Calls':>7} {'Avg':>10} {'% Time':>8}")
        print(f"  {'─'*28} {'─'*10} {'─'*7} {'─'*10} {'─'*8}")
        for op_name, total, count in top:
            avg = total / count if count else 0
            pct = (total / total_ms * 100) if total_ms else 0
            color = CYAN if pct > 20 else (YELLOW if pct > 5 else RESET)
            print(f"  {color}{op_name:<28}{RESET} "
                  f"{_fmt_ms(total):>10} {count:>7} "
                  f"{_fmt_ms(avg):>10} {pct:>7.1f}%")

    print(f"\n{DIM}  Use .html('report.html') for the interactive dashboard.{RESET}\n")


# ── HTML report ───────────────────────────────────────────────────────────────

def render_html(trace: "Trace", path: str):
    """Render a full interactive HTML profiling dashboard."""
    ops_json = json.dumps([op.to_dict() for op in trace.ops])
    top_ops = trace.top_ops(20)
    top_ops_json = json.dumps([[n, round(ms, 3), c] for n, ms, c in top_ops])
    cats = trace.by_category()
    cat_data = {k: round(sum(o.duration_ms for o in v), 3) for k, v in cats.items()}
    cat_json = json.dumps(cat_data)
    dev_data = {k: round(v, 3) for k, v in trace.device_breakdown().items()}
    dev_json = json.dumps(dev_data)
    chip = trace.metadata.get("chip", {})
    chip_str = chip.get("chip", "Apple Silicon") if isinstance(chip, dict) else "Apple Silicon"
    mem_gb = chip.get("memory_gb", "?") if isinstance(chip, dict) else "?"
    total_ms = round(trace.total_duration_ms, 2)
    total_flops = trace.total_flops()
    total_ops = len(trace.ops)
    gflops = round(total_flops / total_ms / 1e6, 1) if total_ms and total_flops else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLX Profiler — {trace.name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg: #0e0f11;
    --surface: #161719;
    --surface2: #1d1f22;
    --border: #2a2c30;
    --text: #e8eaed;
    --text2: #9aa0a6;
    --accent: #4fc3f7;
    --accent2: #81c995;
    --warn: #ffb74d;
    --danger: #ef5350;
    --purple: #ce93d8;
    --font: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    --sans: -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.5; }}
  a {{ color: var(--accent); text-decoration: none; }}

  header {{
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }}
  .logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.3px;
  }}
  .logo-icon {{
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    color: #000;
  }}
  .trace-name {{
    color: var(--text2);
    font-size: 13px;
    font-family: var(--font);
  }}

  .main {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}

  .chip-banner {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 13px;
    color: var(--text2);
  }}
  .chip-name {{ color: var(--accent); font-family: var(--font); font-size: 14px; font-weight: 600; }}

  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .metric {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }}
  .metric-label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
  .metric-value {{ font-size: 26px; font-weight: 600; font-family: var(--font); letter-spacing: -0.5px; }}
  .metric-sub {{ font-size: 11px; color: var(--text2); margin-top: 2px; }}
  .metric.accent {{ border-color: var(--accent); }}
  .metric.accent .metric-value {{ color: var(--accent); }}

  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .grid3 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{
    .grid2, .grid3 {{ grid-template-columns: 1fr; }}
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }}
  .card-title {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text2);
    margin-bottom: 16px;
    font-weight: 500;
  }}

  .chart-wrap {{ position: relative; width: 100%; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text2); text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); font-weight: 500; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1f2124; font-family: var(--font); font-size: 13px; }}
  tr:hover td {{ background: var(--surface2); }}
  tr:last-child td {{ border-bottom: none; }}
  .bar-cell {{ min-width: 120px; }}
  .bar-bg {{ height: 4px; background: var(--border); border-radius: 2px; }}
  .bar-fill {{ height: 4px; border-radius: 2px; background: var(--accent); transition: width 0.3s; }}
  .pct {{ color: var(--text2); font-size: 12px; }}

  .tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: var(--font);
    font-weight: 500;
  }}
  .tag-compute {{ background: #1a2e4a; color: var(--accent); }}
  .tag-memory {{ background: #2a2040; color: var(--purple); }}
  .tag-activation {{ background: #1f3a2a; color: var(--accent2); }}
  .tag-quantize {{ background: #3a2a10; color: var(--warn); }}
  .tag-elementwise {{ background: #2a2020; color: #ef9a9a; }}
  .tag-reduction {{ background: #1a2a2a; color: #80cbc4; }}
  .tag-embedding {{ background: #202a1a; color: #a5d6a7; }}
  .tag-other {{ background: #222; color: var(--text2); }}

  .timeline {{ position: relative; margin-top: 8px; }}
  .tl-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 5px; font-size: 12px; }}
  .tl-label {{ width: 140px; color: var(--text2); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; text-align: right; font-family: var(--font); }}
  .tl-track {{ flex: 1; height: 16px; background: var(--border); border-radius: 3px; position: relative; overflow: hidden; }}
  .tl-bar {{ position: absolute; height: 100%; border-radius: 3px; opacity: 0.85; }}
  .tl-dur {{ width: 50px; color: var(--text2); font-size: 11px; font-family: var(--font); }}

  .roofline-note {{ font-size: 12px; color: var(--text2); margin-top: 8px; }}

  #ops-search {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 8px 12px;
    font-size: 13px;
    width: 100%;
    margin-bottom: 12px;
    font-family: var(--font);
    outline: none;
  }}
  #ops-search:focus {{ border-color: var(--accent); }}

  .footer {{
    padding: 20px 32px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--text2);
    margin-top: 32px;
  }}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">P</div>
    MLX Profiler
  </div>
  <div class="trace-name">{trace.name}</div>
</header>

<div class="main">

  <div class="chip-banner">
    <div>
      <div class="chip-name">{chip_str}</div>
      <div>{mem_gb} GB Unified Memory</div>
    </div>
    <div style="width: 1px; height: 36px; background: var(--border);"></div>
    <div>Framework: <span style="color:var(--text);">Apple MLX</span></div>
    <div>Trace: <span style="color:var(--text); font-family: var(--font);">{trace.name}</span></div>
  </div>

  <div class="metrics">
    <div class="metric accent">
      <div class="metric-label">Total Time</div>
      <div class="metric-value">{_fmt_ms_html(total_ms)}</div>
      <div class="metric-sub">wall clock</div>
    </div>
    <div class="metric">
      <div class="metric-label">Operations</div>
      <div class="metric-value">{total_ops}</div>
      <div class="metric-sub">recorded</div>
    </div>
    <div class="metric">
      <div class="metric-label">GFLOPS</div>
      <div class="metric-value">{gflops}</div>
      <div class="metric-sub">estimated throughput</div>
    </div>
    <div class="metric">
      <div class="metric-label">Est. BW</div>
      <div class="metric-value">{_fmt_bytes_html(trace.total_memory_bytes())}</div>
      <div class="metric-sub">tensor traffic</div>
    </div>
    <div class="metric">
      <div class="metric-label">Total FLOPs</div>
      <div class="metric-value">{_fmt_flops_html(total_flops)}</div>
      <div class="metric-sub">multiply-adds</div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <div class="card-title">Time by operation</div>
      <div class="chart-wrap" style="height:280px"><canvas id="topOpsChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Category breakdown</div>
      <div class="chart-wrap" style="height:280px"><canvas id="catChart"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title">Flame timeline (top 30 ops by start time)</div>
    <div id="timeline" class="timeline"></div>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title">Operation table</div>
    <input id="ops-search" placeholder="Filter operations..." />
    <div style="overflow-x:auto">
      <table id="ops-table">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Category</th>
            <th>Duration</th>
            <th>% Time</th>
            <th>Input shapes</th>
            <th>Output shapes</th>
            <th>Dtype</th>
            <th>Device</th>
            <th>Est. FLOPs</th>
            <th>AI</th>
          </tr>
        </thead>
        <tbody id="ops-body"></tbody>
      </table>
    </div>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <div class="card-title">Arithmetic intensity (roofline)</div>
    <div class="chart-wrap" style="height:260px"><canvas id="rooflineChart"></canvas></div>
    <p class="roofline-note">X-axis: arithmetic intensity (FLOPs/byte). Y-axis: estimated GFLOPS. Dashed line = memory-bound / compute-bound boundary at ~200 FLOPs/byte for Apple Silicon.</p>
  </div>

</div>

<div class="footer">
  Generated by mlx-profiler &bull; {trace.name} &bull; {total_ops} operations
</div>

<script>
const OPS = {ops_json};
const TOP_OPS = {top_ops_json};
const CAT_DATA = {cat_json};
const DEV_DATA = {dev_json};
const TOTAL_MS = {total_ms};

const CAT_COLORS = {{
  compute: '#4fc3f7', memory: '#ce93d8', activation: '#81c995',
  quantize: '#ffb74d', elementwise: '#ef9a9a', reduction: '#80cbc4',
  embedding: '#a5d6a7', other: '#9aa0a6'
}};

// ── Top ops bar chart ────────────────────────────────────────────────────────
(function() {{
  const labels = TOP_OPS.map(r => r[0]);
  const data = TOP_OPS.map(r => r[1]);
  new Chart(document.getElementById('topOpsChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{ data, backgroundColor: '#4fc3f7', borderRadius: 3, borderSkipped: false }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{
        callbacks: {{ label: ctx => ' ' + ctx.parsed.x.toFixed(2) + ' ms' }}
      }} }},
      scales: {{
        x: {{ grid: {{ color: '#2a2c30' }}, ticks: {{ color: '#9aa0a6', font: {{ family: 'SF Mono, Fira Code, monospace', size: 11 }} }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#e8eaed', font: {{ family: 'SF Mono, Fira Code, monospace', size: 11 }} }} }}
      }}
    }}
  }});
}})();

// ── Category donut ───────────────────────────────────────────────────────────
(function() {{
  const labels = Object.keys(CAT_DATA);
  const data = labels.map(k => CAT_DATA[k]);
  const colors = labels.map(k => CAT_COLORS[k] || '#9aa0a6');
  new Chart(document.getElementById('catChart'), {{
    type: 'doughnut',
    data: {{ labels, datasets: [{{ data, backgroundColor: colors, borderWidth: 0, hoverOffset: 4 }}] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {{
        legend: {{
          display: true, position: 'right',
          labels: {{ color: '#9aa0a6', font: {{ size: 11 }}, padding: 12, boxWidth: 12 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => ` ${{ctx.label}}: ${{ctx.parsed.toFixed(2)}} ms (${{(ctx.parsed/TOTAL_MS*100).toFixed(1)}}%)`
          }}
        }}
      }}
    }}
  }});
}})();

// ── Flame timeline ───────────────────────────────────────────────────────────
(function() {{
  const sorted = [...OPS].sort((a,b) => a.start_ns - b.start_ns).slice(0, 30);
  const tl = document.getElementById('timeline');
  if (!sorted.length) return;
  const minT = sorted[0].start_ns;
  const maxT = Math.max(...sorted.map(o => o.end_ns));
  const span = maxT - minT || 1;
  const COLORS = ['#4fc3f7','#81c995','#ce93d8','#ffb74d','#ef9a9a','#80cbc4','#a5d6a7'];
  const catIdx = {{}};
  let ci = 0;
  sorted.forEach(op => {{
    if (!(op.category in catIdx)) catIdx[op.category] = ci++ % COLORS.length;
    const pctStart = ((op.start_ns - minT) / span * 100).toFixed(2);
    const pctWidth = Math.max(((op.end_ns - op.start_ns) / span * 100), 0.3).toFixed(2);
    const dur = op.duration_us < 1000 ? op.duration_us.toFixed(0) + ' µs' : (op.duration_us/1000).toFixed(2) + ' ms';
    const row = document.createElement('div');
    row.className = 'tl-row';
    row.innerHTML = `
      <div class="tl-label" title="${{op.metadata && op.metadata.layer_path || op.name}}">${{op.name}}</div>
      <div class="tl-track">
        <div class="tl-bar" style="left:${{pctStart}}%;width:${{pctWidth}}%;background:${{COLORS[catIdx[op.category]]}}"></div>
      </div>
      <div class="tl-dur">${{dur}}</div>
    `;
    tl.appendChild(row);
  }});
}})();

// ── Ops table ────────────────────────────────────────────────────────────────
(function() {{
  function fmtShapes(shapes) {{
    if (!shapes || !shapes.length) return '—';
    return shapes.map(s => '[' + s.join('×') + ']').join(' ');
  }}
  function fmtFlops(f) {{
    if (!f) return '—';
    if (f < 1e6) return (f/1e3).toFixed(1)+'K';
    if (f < 1e9) return (f/1e6).toFixed(1)+'M';
    return (f/1e9).toFixed(2)+'G';
  }}
  function catTag(cat) {{
    return `<span class="tag tag-${{cat || 'other'}}">${{cat || 'other'}}</span>`;
  }}

  function renderTable(ops) {{
    const tbody = document.getElementById('ops-body');
    tbody.innerHTML = '';
    ops.forEach(op => {{
      const pct = TOTAL_MS ? (op.duration_us / 1000 / TOTAL_MS * 100) : 0;
      const dur = op.duration_us < 1000
        ? op.duration_us.toFixed(0) + ' µs'
        : (op.duration_us/1000).toFixed(2) + ' ms';
      const ai = op.arithmetic_intensity ? op.arithmetic_intensity.toFixed(1) : '—';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color:#e8eaed;font-weight:500">${{op.name}}</td>
        <td>${{catTag(op.category)}}</td>
        <td>${{dur}}</td>
        <td class="bar-cell">
          <div style="display:flex;align-items:center;gap:6px">
            <div class="bar-bg" style="flex:1"><div class="bar-fill" style="width:${{Math.min(pct,100)}}%"></div></div>
            <span class="pct">${{pct.toFixed(1)}}%</span>
          </div>
        </td>
        <td style="color:#9aa0a6">${{fmtShapes(op.input_shapes)}}</td>
        <td style="color:#9aa0a6">${{fmtShapes(op.output_shapes)}}</td>
        <td style="color:#ce93d8">${{op.dtype}}</td>
        <td style="color:#81c995">${{op.device}}</td>
        <td>${{fmtFlops(op.flops)}}</td>
        <td>${{ai}}</td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  renderTable(OPS);

  document.getElementById('ops-search').addEventListener('input', function() {{
    const q = this.value.toLowerCase();
    renderTable(OPS.filter(op =>
      op.name.toLowerCase().includes(q) ||
      op.category.toLowerCase().includes(q) ||
      op.dtype.toLowerCase().includes(q) ||
      op.device.toLowerCase().includes(q)
    ));
  }});
}})();

// ── Roofline scatter ─────────────────────────────────────────────────────────
(function() {{
  const pts = OPS.filter(o => o.arithmetic_intensity && o.duration_us > 0 && o.flops).map(o => ({{
    x: parseFloat(o.arithmetic_intensity.toFixed(2)),
    y: parseFloat((o.flops / o.duration_us / 1000).toFixed(2)),
    name: o.name,
    cat: o.category,
  }}));

  if (!pts.length) {{
    document.getElementById('rooflineChart').closest('.card').style.display = 'none';
    return;
  }}

  const CAT_COLORS_RL = {{
    compute:'#4fc3f7', memory:'#ce93d8', activation:'#81c995',
    quantize:'#ffb74d', other:'#9aa0a6'
  }};

  new Chart(document.getElementById('rooflineChart'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Operations',
        data: pts,
        backgroundColor: pts.map(p => (CAT_COLORS_RL[p.cat] || '#9aa0a6') + 'bb'),
        pointRadius: 5,
        pointHoverRadius: 7,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ` ${{ctx.raw.name}}: AI=${{ctx.raw.x}}, ${{ctx.raw.y}} GFLOPS`
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Arithmetic Intensity (FLOPs/byte)', color: '#9aa0a6' }},
          grid: {{ color: '#2a2c30' }},
          ticks: {{ color: '#9aa0a6' }}
        }},
        y: {{
          title: {{ display: true, text: 'GFLOPS', color: '#9aa0a6' }},
          grid: {{ color: '#2a2c30' }},
          ticks: {{ color: '#9aa0a6' }}
        }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


def _fmt_ms_html(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f}µs"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms/1000:.2f}s"


def _fmt_bytes_html(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b/1024:.1f}KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.1f}GB"


def _fmt_flops_html(f: int) -> str:
    if not f:
        return "0"
    if f < 1e6:
        return f"{f/1e3:.1f}K"
    if f < 1e9:
        return f"{f/1e6:.1f}M"
    if f < 1e12:
        return f"{f/1e9:.2f}G"
    return f"{f/1e12:.2f}T"
