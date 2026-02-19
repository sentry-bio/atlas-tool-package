# biosphere-atlas

**Geometric genomics tools for the tree of life.**

The BiosphereAtlas encoder compresses genomes into coordinates on a hyperbolic manifold (κ = 5/4). Every tool in this package is an interface to that manifold — no alignment, no k-mer index, no database download.

```
pip install biosphere-atlas
pip install biosphere-atlas[all]    # + geoopt, plotly
```

---

## Quick start

```python
from biosphere_atlas import Atlas

# Load a checkpoint — works with V13, V15, V15.1 checkpoints
atlas = Atlas.from_checkpoint("v15_1/best.pt", tokenizer_path="bpe_vocab.json")

# Embed any genome sequence
emb = atlas.encode("ACGTACGT...")   # → torch.Tensor (129,)
print(f"κ = {atlas.kappa:.4f}")     # 1.2500
```

---

## Tools

### `atlas-place` — Phylogenetic placement

Drop-in replacement for pplacer / GTDB-Tk. Place query genomes into the
reference tree using geodesic nearest-neighbour search on the Poincaré ball.

```bash
# Build reference database from training manifest
atlas-place build-ref \
  --model v15_1/best.pt \
  --tokenizer bpe_vocab.json \
  --manifest training.csv \
  --output ref.pkl

# Place query sequences
atlas-place place \
  --reference ref.pkl \
  --model v15_1/best.pt \
  --tokenizer bpe_vocab.json \
  query.fasta \
  --output placements.tsv
```

Python API:
```python
from biosphere_atlas.place.reference import ReferenceDB
from biosphere_atlas.place.placer import PlacementEngine

db = ReferenceDB.load("ref.pkl")
engine = PlacementEngine(db, kappa=db.kappa)
result = engine.place(atlas.encode(sequence))
print(result.best_placement.taxon_id)   # e.g. "f__Bacteria|Streptomycetaceae"
```

### `atlas-tree` — Phylogenetic tree construction

Build ultrafast neighbor-joining trees from prototype embeddings. No alignment step.
Quartet-consistent by construction.

```bash
atlas-tree build \
  --reference ref.pkl \
  --output tree.json \
  --newick tree.nwk \
  --svg tree.svg

atlas-tree quality --reference ref.pkl
# Quartet consistency: 100.0%  |  δ = 0.124
```

### `atlas-dark` — Dark matter triage

Map regions of high uncertainty in the manifold and triage unseen genomes
into `redundant / novel_certain / novel_uncertain`.

```bash
# Build uncertainty field
atlas-dark map --reference ref.pkl --output dark_map.json

# Triage unseen FASTA
atlas-dark triage \
  --fasta unseen.fasta \
  --reference ref.pkl \
  --dark-map dark_map.json \
  --model v15_1/best.pt \
  --tokenizer bpe_vocab.json \
  --output triage.tsv

# Triage feeds directly into the next training manifest:
#   redundant     → skip
#   novel_certain → add to training data
#   novel_uncertain → flag for review
```

### `atlas-viewer` — Interactive visualization

**2D Poincaré disk** — Möbius-navigable, runs in any browser, no server needed:

```bash
atlas-viewer render --tree tree.json -o viewer.html
atlas-viewer render --embeddings embs.pt --taxa taxa.txt -o viewer.html
open viewer.html
```

**3D Poincaré ball** — Dark-mode Plotly viewer, radial depth visible:

```bash
atlas-viewer render --mode 3d --tree tree.json -o viewer_3d.html
open viewer_3d.html
```

Python API:
```python
from biosphere_atlas.viewer import from_tree, generate_viewer_html, generate_3d_viewer_html

data = from_tree("tree.json", kappa=1.25)
generate_viewer_html(data, "viewer.html")
generate_3d_viewer_html(data, "viewer_3d.html")
```

### `atlas-chimera` — Chimera detection

Detect chimeric sequences by sliding-window embedding divergence.

```bash
atlas-chimera detect \
  --model v15_1/best.pt \
  --tokenizer bpe_vocab.json \
  contigs.fasta \
  --output chimeras.tsv
```

### `atlas-novelty` — Novelty scoring

Score sequences by geodesic distance to the nearest reference prototype.

```bash
atlas-novelty detect \
  --reference ref.pkl \
  --model v15_1/best.pt \
  --tokenizer bpe_vocab.json \
  query.fasta \
  --output novelty.tsv
```

### `atlas-hplg` — HPLG classification

Hierarchical Poincaré Learning Graph classification with conformal calibration
and fallback zones.

```bash
atlas-hplg classify \
  --model v15_1/best.pt \
  --reference ref.pkl \
  query.fasta \
  --output classifications.tsv
```

---

## The autocatalytic loop

The tools form a self-improving data pipeline:

```
Checkpoint
    ↓
atlas-place build-ref   →   Reference DB
    ↓                             ↓
atlas-tree              atlas-dark map + triage
    ↓                             ↓
tree quality metrics     novel_certain manifest
                                  ↓
                         next training run (V15.1, V16...)
                                  ↓
                            better checkpoint
```

Each cycle: more charted manifold → better triage → more informative training data → richer geometry.

---

## Geometric coherence reference values (V15 / V15.1)

| Metric | V13 | V15 | Target |
|--------|-----|-----|--------|
| κ deviation from 5/4 | 0.017% | **0.00008%** | < 0.01% |
| Quartet consistency | 70.2% | **100%** | > 65% |
| Mean δ hyperbolicity | 0.197 | **0.124** | < 0.25 |
| Bacteria family top-1 | 83.2% | 82% | — |
| Atlas vs Kraken2 (family top-1) | 4.5× | 4.5× | > 2× |
| Radial std (‖z‖) | ~0.002 | ~0.002 | growing (V15.1) |
| α crystallization | — | — | **discovered (V15.1)** |

Full diagnostic playbook: [COHERENCE_PLAYBOOK.md](COHERENCE_PLAYBOOK.md)

---

## Architecture

```
biosphere_atlas/
├── core/
│   ├── atlas.py          # Atlas.from_checkpoint() — unified loader
│   ├── hyperbolic.py     # Single canonical geometry: poincare_distance,
│   │                     # exp_map_0, log_map_0, karcher_mean, KAPPA_*
│   ├── tokenizer.py      # SimpleBPETokenizer (V13/V15 compatible)
│   ├── coordinates.py    # BiosphereCoordinate (r, θ)
│   └── io.py             # FASTA reading, TSV/JSONL writing
├── place/                # Phylogenetic placement engine
├── tree/                 # NJ tree construction + quartet quality
├── dark/                 # Uncertainty field, dark matter mapping, triage
├── viewer/               # 2D Poincaré disk + 3D Poincaré ball
├── chimera/              # Sliding-window chimera detection
├── novelty/              # Geodesic novelty scoring
└── hplg/                 # HPLG classification with calibration
```

---

## License

MIT © 2025 Sentry Bio, Inc.
