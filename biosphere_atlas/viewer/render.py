"""
Self-contained HTML viewer generation.
========================================

Generates a single HTML file containing all data and JavaScript needed
for an interactive Poincare disk viewer.  No external dependencies —
the file can be opened in any modern browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .data import ViewerData


def generate_viewer_html(
    data: ViewerData,
    output_path: str,
    title: Optional[str] = None,
    canvas_size: int = 900,
    color_by: str = "rank",
) -> str:
    """Generate a self-contained HTML Poincare disk viewer.

    Args:
        data: ViewerData prepared by the data layer.
        output_path: where to write the HTML file.
        title: page title (defaults to data.title).
        canvas_size: canvas width/height in pixels.
        color_by: coloring scheme ('rank', 'depth', 'none').

    Returns:
        Path to the written HTML file.
    """
    if title is None:
        title = data.title

    payload = data.to_json()
    html = _build_html(title, canvas_size, payload, color_by)

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _build_html(
    title: str, canvas_size: int, data_json: str, color_by: str
) -> str:
    """Assemble the complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0a0a0f; color: #e0e0e0; overflow: hidden; }}
#container {{ display: flex; height: 100vh; }}
#canvas-wrap {{ flex: 1; display: flex; justify-content: center; align-items: center;
                position: relative; }}
canvas {{ cursor: crosshair; }}
#sidebar {{ width: 280px; padding: 20px; background: #12121a;
            border-left: 1px solid #2a2a3a; overflow-y: auto; }}
h3 {{ color: #7eb8da; margin-bottom: 12px; font-size: 15px; letter-spacing: 1px; }}
.stat {{ font-size: 12px; color: #888; margin: 4px 0; }}
.stat b {{ color: #aaa; }}
#hover-info {{ position: absolute; top: 20px; left: 20px; background: rgba(18,18,26,0.9);
               padding: 10px 14px; border-radius: 6px; font-size: 12px; color: #ccc;
               pointer-events: none; display: none; border: 1px solid #2a2a3a; }}
#hover-info .taxon {{ color: #7eb8da; font-weight: 600; font-size: 14px; }}
#hover-info .detail {{ color: #888; margin-top: 4px; }}
.btn {{ padding: 6px 14px; margin: 4px 2px; cursor: pointer; background: #1e1e2e;
        border: 1px solid #3a3a4a; color: #ccc; border-radius: 4px; font-size: 12px; }}
.btn:hover {{ background: #2a2a3a; }}
#legend {{ margin-top: 16px; }}
.legend-item {{ display: flex; align-items: center; margin: 3px 0; font-size: 11px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}
.controls {{ margin: 12px 0; }}
#variance {{ font-size: 11px; color: #666; margin-top: 12px; }}
</style>
</head>
<body>
<div id="container">
  <div id="canvas-wrap">
    <canvas id="disk" width="{canvas_size}" height="{canvas_size}"></canvas>
    <div id="hover-info">
      <div class="taxon" id="hi-taxon"></div>
      <div class="detail" id="hi-detail"></div>
    </div>
  </div>
  <div id="sidebar">
    <h3>BIOSPHEREATLAS</h3>
    <div class="stat"><b>Organisms:</b> <span id="s-count">0</span></div>
    <div class="stat"><b>Edges:</b> <span id="s-edges">0</span></div>
    <div class="stat"><b>&kappa;:</b> <span id="s-kappa">1.247</span></div>
    <div class="stat"><b>Zoom:</b> <span id="s-zoom">1.0</span>x</div>
    <div class="controls">
      <button class="btn" onclick="app.reset()">Reset View</button>
    </div>
    <div id="legend"></div>
    <div id="variance"></div>
    <div style="margin-top: 20px; font-size: 10px; color: #444;">
      Click to navigate. Scroll to zoom. Drag to rotate.
    </div>
  </div>
</div>

<script id="viewer-data" type="application/json">{data_json}</script>

<script>
"use strict";

// -- Complex arithmetic (inline for self-contained viewer) --
const C = {{
  mul(a, b) {{ return [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]]; }},
  conj(a) {{ return [a[0], -a[1]]; }},
  div(a, b) {{
    const d = b[0]*b[0]+b[1]*b[1]+1e-30;
    return [(a[0]*b[0]+a[1]*b[1])/d, (a[1]*b[0]-a[0]*b[1])/d];
  }},
  add(a, b) {{ return [a[0]+b[0], a[1]+b[1]]; }},
  scale(a, s) {{ return [a[0]*s, a[1]*s]; }},
  abs2(a) {{ return a[0]*a[0]+a[1]*a[1]; }},
}};

// -- Mobius transform on the unit disk --
class Mobius {{
  constructor(a, b) {{ this.a = a || [1,0]; this.b = b || [0,0]; }}

  static identity() {{ return new Mobius([1,0], [0,0]); }}

  static translateToOrigin(p, R) {{
    // f(z) = (z - p)/(1 - conj(p)*z)  on unit disk
    // SU(1,1): a = 1/sqrt(1-|p|^2), b = -p/sqrt(1-|p|^2)
    const pu = [p[0]/R, p[1]/R]; // normalize to unit disk
    const pn2 = C.abs2(pu);
    if (pn2 > 0.98) return Mobius.identity();
    const s = 1/Math.sqrt(Math.max(1-pn2, 1e-10));
    return new Mobius([s, 0], [-pu[0]*s, -pu[1]*s]);
  }}

  static rotation(angle) {{
    const h = angle/2;
    return new Mobius([Math.cos(h), Math.sin(h)], [0, 0]);
  }}

  apply(z, R) {{
    // f(z) = (a*z + b)/(conj(b)*z + conj(a))  on unit disk
    const zu = [z[0]/R, z[1]/R];
    const num = C.add(C.mul(this.a, zu), this.b);
    const den = C.add(C.mul(C.conj(this.b), zu), C.conj(this.a));
    const r = C.div(num, den);
    return [r[0]*R, r[1]*R];
  }}

  compose(other) {{
    // SU(1,1) matrix multiply
    const a = C.add(C.mul(this.a, other.a), C.mul(this.b, C.conj(other.b)));
    const b = C.add(C.mul(this.a, other.b), C.mul(this.b, C.conj(other.a)));
    return new Mobius(a, b);
  }}

  interpolate(other, t) {{
    // Smoothstep ease
    const s = t*t*(3-2*t);
    const a = [this.a[0]*(1-s)+other.a[0]*s, this.a[1]*(1-s)+other.a[1]*s];
    const b = [this.b[0]*(1-s)+other.b[0]*s, this.b[1]*(1-s)+other.b[1]*s];
    // Re-normalize to SU(1,1)
    const det = C.abs2(a) - C.abs2(b);
    if (det > 1e-10) {{
      const sc = 1/Math.sqrt(det);
      a[0]*=sc; a[1]*=sc; b[0]*=sc; b[1]*=sc;
    }}
    return new Mobius(a, b);
  }}
}}

// -- Rank colors --
const RANK_COLORS = {{
  0: '#e74c3c', 1: '#e67e22', 2: '#f1c40f', 3: '#2ecc71',
  4: '#3498db', 5: '#9b59b6', 6: '#34495e', '-1': '#555555'
}};
const RANK_NAMES = ['Domain','Phylum','Class','Order','Family','Genus','Species'];

// -- Main viewer class --
class PoincareViewer {{
  constructor(canvasId, rawData) {{
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.W = this.canvas.width;
    this.H = this.canvas.height;
    this.cx = this.W/2;
    this.cy = this.H/2;

    this.data = rawData;
    this.kappa = rawData.kappa || 1.247;
    this.R = 1/Math.sqrt(this.kappa);  // ball radius
    this.scale = (this.W/2 - 30) / this.R;  // pixels per unit

    // Mobius navigation
    this.mobius = Mobius.identity();
    this.targetMobius = Mobius.identity();
    this.animT = 1;
    this.animSpeed = 3;
    this.zoomLevel = 1;

    // Interaction state
    this.dragging = false;
    this.dragStart = null;
    this.lastMouse = null;
    this.hoveredIdx = -1;

    this._setupUI();
    this._setupEvents();
    this._startLoop();
  }}

  _setupUI() {{
    document.getElementById('s-count').textContent = this.data.nOrganisms;
    document.getElementById('s-edges').textContent = this.data.edges.length;
    document.getElementById('s-kappa').textContent = this.kappa.toFixed(3);

    // Legend
    const legend = document.getElementById('legend');
    const usedRanks = new Set(this.data.ranks.filter(r => r >= 0));
    usedRanks.forEach(r => {{
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = '<div class="legend-dot" style="background:' +
        (RANK_COLORS[r]||'#555') + '"></div>' + (RANK_NAMES[r]||'Unknown');
      legend.appendChild(item);
    }});

    // Variance explained
    if (this.data.projectionVariance && this.data.projectionVariance.length) {{
      const pct = this.data.projectionVariance.map(v => (v*100).toFixed(1)+'%').join(' + ');
      document.getElementById('variance').textContent = 'PCA variance: ' + pct;
    }}
  }}

  _setupEvents() {{
    const c = this.canvas;
    c.addEventListener('click', e => this._onClick(e));
    c.addEventListener('mousedown', e => {{ this.dragging = true; this.dragStart = [e.offsetX, e.offsetY]; this.lastMouse = [e.offsetX, e.offsetY]; }});
    c.addEventListener('mousemove', e => this._onMouseMove(e));
    c.addEventListener('mouseup', () => {{ this.dragging = false; }});
    c.addEventListener('mouseleave', () => {{ this.dragging = false; this.hoveredIdx = -1; }});
    c.addEventListener('wheel', e => {{ e.preventDefault(); this._onScroll(e); }}, {{passive: false}});
  }}

  _onClick(e) {{
    if (this.dragging && this.dragStart) {{
      const dx = e.offsetX - this.dragStart[0], dy = e.offsetY - this.dragStart[1];
      if (Math.sqrt(dx*dx+dy*dy) > 5) return; // was a drag, not a click
    }}
    const dp = this._screenToDisk(e.offsetX, e.offsetY);
    if (C.abs2([dp[0]/this.R, dp[1]/this.R]) > 0.95) return;
    const t = Mobius.translateToOrigin(dp, this.R);
    this.targetMobius = t.compose(this.mobius);
    this.animT = 0;
    this.zoomLevel = Math.min(this.zoomLevel + 1, 10);
    document.getElementById('s-zoom').textContent = this.zoomLevel.toFixed(1);
  }}

  _onMouseMove(e) {{
    if (this.dragging && this.lastMouse) {{
      const dx = e.offsetX - this.lastMouse[0];
      const dy = e.offsetY - this.lastMouse[1];
      const angle = Math.atan2(dy, dx) * 0.005;
      const rot = Mobius.rotation(angle);
      this.mobius = rot.compose(this.mobius);
      this.targetMobius = this.mobius;
      this.lastMouse = [e.offsetX, e.offsetY];
      return;
    }}
    // Hover detection
    const dp = this._screenToDisk(e.offsetX, e.offsetY);
    let closest = -1, closestD = 20;
    for (let i = 0; i < this.data.coords.length; i++) {{
      const p = this.mobius.apply(this.data.coords[i], this.R);
      const sp = this._diskToScreen(p);
      const d = Math.sqrt((sp[0]-e.offsetX)**2 + (sp[1]-e.offsetY)**2);
      if (d < closestD) {{ closestD = d; closest = i; }}
    }}
    this.hoveredIdx = closest;
    const hi = document.getElementById('hover-info');
    if (closest >= 0 && closestD < 20) {{
      hi.style.display = 'block';
      document.getElementById('hi-taxon').textContent = this.data.taxonIds[closest];
      const rank = this.data.ranks[closest];
      const lin = this.data.lineages[closest] || '';
      document.getElementById('hi-detail').textContent =
        (rank >= 0 ? RANK_NAMES[rank] : '') + (lin ? ' | ' + lin : '');
    }} else {{
      hi.style.display = 'none';
    }}
  }}

  _onScroll(e) {{
    // Zoom: scale points toward/away from center
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    this.zoomLevel = Math.max(0.5, Math.min(10, this.zoomLevel * factor));
    document.getElementById('s-zoom').textContent = this.zoomLevel.toFixed(1);
  }}

  _screenToDisk(sx, sy) {{
    return [(sx - this.cx) / this.scale, (this.cy - sy) / this.scale];
  }}

  _diskToScreen(dp) {{
    return [dp[0] * this.scale + this.cx, this.cy - dp[1] * this.scale];
  }}

  reset() {{
    this.targetMobius = Mobius.identity();
    this.animT = 0;
    this.zoomLevel = 1;
    document.getElementById('s-zoom').textContent = '1.0';
  }}

  _startLoop() {{
    let lastTime = performance.now();
    const loop = (now) => {{
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      if (this.animT < 1) {{
        this.animT = Math.min(1, this.animT + dt * this.animSpeed);
        this.mobius = this.mobius.interpolate(this.targetMobius, this.animT);
        if (this.animT >= 1) this.mobius = this.targetMobius;
      }}
      this._render();
      requestAnimationFrame(loop);
    }};
    requestAnimationFrame(loop);
  }}

  _render() {{
    const ctx = this.ctx;
    const W = this.W, H = this.H;

    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, W, H);

    // Boundary circle
    const rPx = this.R * this.scale;
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(this.cx, this.cy, rPx, 0, 2*Math.PI);
    ctx.stroke();

    // Rank bands
    for (const band of this.data.rankBands) {{
      const re = band.euclideanRadius * this.scale;
      if (re < 2 || re > rPx) continue;
      ctx.strokeStyle = band.color;
      ctx.globalAlpha = 0.12;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(this.cx, this.cy, re, 0, 2*Math.PI);
      ctx.stroke();
      // Label
      ctx.globalAlpha = 0.25;
      ctx.font = '9px sans-serif';
      ctx.fillStyle = band.color;
      ctx.fillText(band.label, this.cx + re + 4, this.cy - 4);
      ctx.globalAlpha = 1;
    }}

    // Transform all points
    const transformed = this.data.coords.map(p => this.mobius.apply(p, this.R));

    // Edges
    ctx.strokeStyle = 'rgba(100,140,180,0.15)';
    ctx.lineWidth = 0.7;
    for (const [i, j, _len] of this.data.edges) {{
      if (i >= transformed.length || j >= transformed.length) continue;
      const p1 = transformed[i], p2 = transformed[j];
      const n1 = C.abs2([p1[0]/this.R, p1[1]/this.R]);
      const n2 = C.abs2([p2[0]/this.R, p2[1]/this.R]);
      if (n1 > 0.98 || n2 > 0.98) continue;
      const s1 = this._diskToScreen(p1), s2 = this._diskToScreen(p2);
      ctx.beginPath();
      ctx.moveTo(s1[0], s1[1]);
      ctx.lineTo(s2[0], s2[1]);
      ctx.stroke();
    }}

    // Points
    for (let i = 0; i < transformed.length; i++) {{
      const p = transformed[i];
      const n2 = C.abs2([p[0]/this.R, p[1]/this.R]);
      if (n2 > 0.98) continue;

      const sp = this._diskToScreen(p);
      const rank = this.data.ranks[i];
      const color = RANK_COLORS[rank] || RANK_COLORS['-1'];
      const isHovered = (i === this.hoveredIdx);

      // Size: closer to center = larger (post-transform)
      const distFromCenter = Math.sqrt(n2);
      const sz = isHovered ? 6 : Math.max(2, 5 * (1 - distFromCenter * 0.7));

      ctx.fillStyle = color;
      ctx.globalAlpha = isHovered ? 1 : 0.8;
      ctx.beginPath();
      ctx.arc(sp[0], sp[1], sz, 0, 2*Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Label for hovered or large/near-center points
      if (isHovered || (distFromCenter < 0.2 && i < 30)) {{
        ctx.fillStyle = '#ccc';
        ctx.font = (isHovered ? '12' : '9') + 'px sans-serif';
        ctx.fillText(this.data.taxonIds[i], sp[0] + sz + 4, sp[1] + 3);
      }}
    }}

    // LUCA marker at origin
    const origin = this.mobius.apply([0, 0], this.R);
    const oScreen = this._diskToScreen(origin);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(oScreen[0], oScreen[1], 4, 0, 2*Math.PI);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '9px sans-serif';
    ctx.fillText('LUCA', oScreen[0] + 8, oScreen[1] + 3);
  }}
}}

// -- Boot --
window.addEventListener('load', () => {{
  const raw = JSON.parse(document.getElementById('viewer-data').textContent);
  window.app = new PoincareViewer('disk', raw);
}});
</script>
</body>
</html>"""
