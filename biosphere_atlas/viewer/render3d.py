"""
Self-contained 3D hyperbolic tree viewer.
==========================================

Generates a single dark-mode HTML file using Plotly.js (CDN) for an
interactive 3D scatter plot of the BiosphereAtlas Poincaré ball.

  atlas-viewer render --mode 3d --tree tree.json -o atlas_3d.html

Design:
  · Organisms as 3D scatter points colored by domain / rank
  · Tree edges as Line3d traces (thin, translucent white)
  · Origin sphere = LUCA
  · Rank-band shells at phylum / class / family geodesic radii
  · Full hover tooltip: taxon, rank, lineage, depth
  · Plotly camera controls: rotate, pan, zoom
  · Fully self-contained — one HTML, no server needed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .data import ViewerData


# Domain colors (matching 2D viewer palette)
_DOMAIN_COLORS = {
    "Bacteria":  "#e74c3c",   # warm red
    "Archaea":   "#3498db",   # blue
    "Eukaryota": "#2ecc71",   # green
    "Viruses":   "#9b59b6",   # purple
}
_RANK_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f",
    "#2ecc71", "#3498db", "#9b59b6", "#34495e",
]
_UNKNOWN_COLOR = "#555555"


def _infer_domain(lineage: str, taxon_id: str) -> str:
    """Heuristic domain from lineage or taxon prefix."""
    for dom in ("Bacteria", "Archaea", "Eukaryota", "Viruses"):
        if dom in lineage or dom in taxon_id:
            return dom
    return "Unknown"


def _color_for_node(rank: int, lineage: str, taxon_id: str, color_by: str) -> str:
    if color_by == "domain":
        dom = _infer_domain(lineage, taxon_id)
        return _DOMAIN_COLORS.get(dom, _UNKNOWN_COLOR)
    if color_by == "rank":
        if 0 <= rank < len(_RANK_COLORS):
            return _RANK_COLORS[rank]
    return _UNKNOWN_COLOR


def generate_3d_viewer_html(
    data: ViewerData,
    output_path: str,
    title: Optional[str] = None,
    color_by: str = "domain",
) -> str:
    """Generate a self-contained dark-mode 3D Poincaré ball viewer.

    Args:
        data: ViewerData (must have coords_3d populated).
        output_path: destination HTML file.
        title: page title.
        color_by: 'domain' or 'rank'.

    Returns:
        Path to the written HTML file.
    """
    if title is None:
        title = data.title

    if data.coords_3d is None:
        raise ValueError(
            "ViewerData.coords_3d is None. "
            "Rebuild ViewerData from embeddings (not pre-loaded 2D coords)."
        )

    coords = data.coords_3d.tolist()       # [[x,y,z], ...]
    N = len(coords)

    # Per-point colors, labels, hover text
    colors, labels, hovers = [], [], []
    for i in range(N):
        tid = data.taxon_ids[i] if i < len(data.taxon_ids) else f"node_{i}"
        lin = data.lineages[i] if i < len(data.lineages) else ""
        rank = data.ranks[i] if i < len(data.ranks) else -1
        colors.append(_color_for_node(rank, lin, tid, color_by))
        labels.append(tid)
        hovers.append(f"{tid}<br>rank: {rank}<br>{lin}")

    # Edge coordinates for Line3d
    # Each edge is a triplet of x/y/z lists: [x0,x1,None], [y0,y1,None], [z0,z1,None]
    ex, ey, ez = [], [], []
    for src, tgt, _ in data.edges:
        if src < N and tgt < N:
            ex += [coords[src][0], coords[tgt][0], None]
            ey += [coords[src][1], coords[tgt][1], None]
            ez += [coords[src][2], coords[tgt][2], None]

    # Rank-band spheres (wireframe) — just the LUCA origin sphere for now
    # Rendered as a small marker at origin
    origin = [0.0, 0.0, 0.0]

    # Payload to embed in HTML
    payload = {
        "coords":    coords,
        "colors":    colors,
        "labels":    labels,
        "hovers":    hovers,
        "edges":     {"x": ex, "y": ey, "z": ez},
        "kappa":     data.kappa,
        "nOrgs":     N,
        "origin":    origin,
        "title":     title,
        "variance":  data.projection_variance_3d,
    }
    payload_json = json.dumps(payload)
    html = _build_3d_html(title, payload_json)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _build_3d_html(title: str, payload_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#060608; color:#e0e0e0;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       overflow:hidden; }}
#app {{ display:flex; height:100vh; }}
#plot {{ flex:1; }}
#panel {{ width:260px; background:#0c0c14; border-left:1px solid #1a1a2e;
          padding:20px; overflow-y:auto; display:flex; flex-direction:column; gap:14px; }}
h2 {{ font-size:13px; letter-spacing:2px; color:#5a9fd4; text-transform:uppercase; }}
.stat {{ font-size:11px; color:#555; }}
.stat b {{ color:#888; }}
.stat span {{ color:#aaa; }}
.legend {{ display:flex; flex-direction:column; gap:4px; }}
.li {{ display:flex; align-items:center; gap:8px; font-size:11px; color:#888; }}
.dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
.badge {{ padding:2px 8px; border-radius:10px; font-size:10px; border:1px solid #2a2a3a;
          color:#666; cursor:pointer; user-select:none; }}
.badge:hover {{ border-color:#4a4a6a; color:#999; }}
footer {{ margin-top:auto; font-size:9px; color:#333; }}
</style>
</head>
<body>
<div id="app">
  <div id="plot"></div>
  <div id="panel">
    <h2>BiosphereAtlas</h2>
    <div class="stat"><b>Organisms:</b> <span id="s-count">—</span></div>
    <div class="stat"><b>κ (curvature):</b> <span id="s-kappa">—</span></div>
    <div class="stat"><b>PCA variance:</b> <span id="s-var">—</span></div>
    <div>
      <div class="stat" style="margin-bottom:8px"><b>Domains</b></div>
      <div class="legend" id="legend"></div>
    </div>
    <div>
      <div class="stat" style="margin-bottom:6px"><b>View</b></div>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        <span class="badge" onclick="resetCamera()">Reset</span>
        <span class="badge" onclick="toggleEdges()">Edges</span>
        <span class="badge" onclick="toggleSpin()">Spin</span>
      </div>
    </div>
    <footer>BiosphereAtlas 3D — Poincaré ball projection<br>
      Radial distance ≈ evolutionary depth.<br>
      Angular position ≈ phylogenetic similarity.
    </footer>
  </div>
</div>

<!-- Plotly CDN — self-contained after first load (cached) -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script id="atlas-data" type="application/json">{payload_json}</script>

<script>
"use strict";

const D = JSON.parse(document.getElementById('atlas-data').textContent);
const DOMAIN_COLORS = {{
  Bacteria: '#e74c3c', Archaea: '#3498db', Eukaryota: '#2ecc71',
  Viruses: '#9b59b6', Unknown: '#555555'
}};

// -- Stats ---------------------------------------------------------------
document.getElementById('s-count').textContent = D.nOrgs;
document.getElementById('s-kappa').textContent = D.kappa.toFixed(4);
if (D.variance && D.variance.length) {{
  const pcts = D.variance.map(v => (v*100).toFixed(1)+'%').join(' + ');
  document.getElementById('s-var').textContent = pcts;
}}

// -- Legend --------------------------------------------------------------
const legend = document.getElementById('legend');
Object.entries(DOMAIN_COLORS).forEach(([name, color]) => {{
  if (name === 'Unknown') return;
  const item = document.createElement('div');
  item.className = 'li';
  item.innerHTML = `<div class="dot" style="background:${{color}}"></div>${{name}}`;
  legend.appendChild(item);
}});

// -- Traces --------------------------------------------------------------
const coords = D.coords;
const xs = coords.map(c => c[0]);
const ys = coords.map(c => c[1]);
const zs = coords.map(c => c[2]);

// Points
const pointTrace = {{
  type: 'scatter3d', mode: 'markers',
  x: xs, y: ys, z: zs,
  text: D.hovers,
  hoverinfo: 'text',
  marker: {{
    size: 3.5,
    color: D.colors,
    opacity: 0.85,
    line: {{ width: 0 }},
  }},
  name: 'Organisms',
}};

// Edges
let edgesVisible = true;
const edgeTrace = {{
  type: 'scatter3d', mode: 'lines',
  x: D.edges.x, y: D.edges.y, z: D.edges.z,
  hoverinfo: 'skip',
  line: {{ color: 'rgba(120,160,200,0.18)', width: 1 }},
  name: 'Tree edges',
}};

// LUCA marker at origin
const lucaTrace = {{
  type: 'scatter3d', mode: 'markers+text',
  x: [0], y: [0], z: [0],
  text: ['LUCA'], textposition: 'top center',
  textfont: {{ size: 10, color: 'rgba(255,255,255,0.5)' }},
  marker: {{ size: 6, color: 'rgba(255,255,255,0.6)', symbol: 'diamond' }},
  hoverinfo: 'text',
  hovertext: ['LUCA — Last Universal Common Ancestor'],
  name: 'LUCA',
}};

// -- Layout --------------------------------------------------------------
const layout = {{
  paper_bgcolor: '#060608',
  plot_bgcolor:  '#060608',
  margin: {{ l:0, r:0, t:0, b:0 }},
  scene: {{
    bgcolor: '#060608',
    xaxis: {{ showgrid:false, zeroline:false, showticklabels:false, title:'' }},
    yaxis: {{ showgrid:false, zeroline:false, showticklabels:false, title:'' }},
    zaxis: {{ showgrid:false, zeroline:false, showticklabels:false, title:'' }},
    camera: {{
      eye: {{ x:1.4, y:1.0, z:0.8 }},
      up:  {{ x:0, y:0, z:1 }},
    }},
    aspectmode: 'cube',
  }},
  showlegend: false,
  hoverlabel: {{
    bgcolor: '#12121e',
    bordercolor: '#2a2a4a',
    font: {{ size:11, color:'#cccccc' }},
  }},
}};

const config = {{
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
}};

Plotly.newPlot('plot', [edgeTrace, pointTrace, lucaTrace], layout, config);

// -- Controls ------------------------------------------------------------
function resetCamera() {{
  Plotly.relayout('plot', {{
    'scene.camera': {{ eye: {{x:1.4,y:1.0,z:0.8}}, up:{{x:0,y:0,z:1}} }}
  }});
}}

let spinning = false;
let spinInterval = null;
let angle = 0;
function toggleSpin() {{
  spinning = !spinning;
  if (spinning) {{
    spinInterval = setInterval(() => {{
      angle += 0.5;
      const r = 1.8;
      const rad = angle * Math.PI / 180;
      Plotly.relayout('plot', {{
        'scene.camera.eye': {{ x: r*Math.cos(rad), y: r*Math.sin(rad), z: 0.8 }}
      }});
    }}, 30);
  }} else {{
    clearInterval(spinInterval);
  }}
}}

function toggleEdges() {{
  edgesVisible = !edgesVisible;
  Plotly.restyle('plot', {{ visible: edgesVisible }}, [0]);
}}
</script>
</body>
</html>"""

