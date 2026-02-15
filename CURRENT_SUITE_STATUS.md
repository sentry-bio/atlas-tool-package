# Biosphere Atlas Suite Status (Quick V14-Era Read)

Date: 2026-02-15

## Checkpoint verdict

- **Mature analysis anchor:** `/zfs_raid/SentryBio/working/checkpoints/v13_living_geometry/best.pt`
  - Rationale: full coherence/placement/tree/dark artifacts already validated.
- **Current training frontier:** `/zfs_raid/SentryBio/working/checkpoints/v14_triage_living_geometry/best.pt`
  - Current best observed in training log: `combined=43.85`, `kappa=1.2502` (early/mid run).
  - Loader compatibility confirmed via `Atlas.from_checkpoint()` smoke.

## Operational status by tool

- **core (`Atlas.from_checkpoint`)**: operational
  - V14 smoke: embedding shape `(129,)`, `kappa≈1.25022`.
- **place**: production-ready
  - Label-aware calibrated benchmark (V13): top1 `0.366`, top5 `0.488`, throughput `~812 seq/s`.
  - Side-by-side (300) vs Kraken2: atlas family top1 `0.393` vs kraken2 `0.087`.
- **tree**: validated
  - Tree quality (333-family subset): `delta=0.1965`, quartet consistency `70.18%`.
  - Artifacts stable: JSON/Newick/SVG.
- **chimera**: integrated and callable through unified package
  - No regressions observed in package integration pass.
- **hplg**: integrated with calibrated fallback-safe path
  - No breakage observed in unified package smoke pass.
- **novelty**: integrated and smoke-tested
  - Built on unified `Atlas` + `ReferenceDB` interfaces.
- **dark (v0.1)**: integrated and field-validated on real data
  - Dark map (family refs): `52 dark / 976 charted` (5.06% dark).
  - Full unseen triage (5757): `3802 redundant`, `1955 novel_certain`, `0 novel_uncertain`.
- **viewer**: integrated, integrity-fixed, real checkpoint rendered
  - Real tree info: `Organisms=664`, `Edges=663`, `Rank bands=7`, `PC1=41.7%`, `PC2=11.9%`.

## What to trust for embedding coherence (quick decision guide)

- **Primary coherence indicators**
  - `atlas-tree` (`delta`, quartet consistency): best structural signal.
  - `atlas-dark` (dark fraction + triage categories): best uncertainty/coverage signal.
  - `atlas-place` (calibrated accept precision + per-domain top-k): best operational classification signal.
- **Support indicators**
  - `atlas-viewer`: sanity/diagnostic interpretability signal.
  - `atlas-novelty`: thresholded novelty boundary behavior.

## Quick interpretation of new sensing mechanisms

- The manifold appears **coherent but coverage-skewed**:
  - Strong charted regions and meaningful tree structure.
  - Dark pockets are sparse but non-zero (mostly long-tail families).
  - Triage split (`redundant` vs `novel_certain`) is highly actionable for dataset expansion.
- V14 is currently preserving curvature stability (`kappa ~1.2502`) while improving combined score over warm-up, but should still be treated as **in-flight** until post-Phase C metrics settle.

