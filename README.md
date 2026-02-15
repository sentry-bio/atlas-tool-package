# biosphere-atlas

Unified Bioinformatics 2.0 toolkit with shared geometry and Atlas loader.

biosphere-atlas v0.1.0
├── core/                          ← The shared geometric foundation
│   ├── hyperbolic.py              ← ONE canonical Poincaré ball geometry
│   │   ├── poincare_distance      (geodesic distance)
│   │   ├── exp_map / log_map      (manifold ↔ tangent space)
│   │   ├── karcher_mean           (Fréchet mean on the ball)
│   │   ├── geodesic_interpolation (manifold-safe lerp)
│   │   ├── mobius_addition        (Poincaré group operation)
│   │   ├── KAPPA_DEFAULT          = 1.247
│   │   ├── KAPPA_FUNCTIONAL       = 1.0   (liquid phase)
│   │   └── KAPPA_PHYLOGENETIC     = 1.2475 (crystal phase)
│   ├── atlas.py                   ← Atlas.from_checkpoint() — THE unified loader
│   │   ├── encode()               (sequence → Poincaré embedding)
│   │   ├── encode_batch()         (batch sequences → embeddings)
│   │   ├── encode_tokens()        (pre-tokenized → embeddings)
│   │   ├── encode_subsequences()  (windowed chimera encoding)
│   │   ├── coordinate()           (embedding → BiosphereCoordinate)
│   │   └── kappa (property)       (live κ from manifold)
│   ├── coordinates.py             ← BiosphereCoordinate (r, θ)
│   ├── tokenizer.py               ← SimpleBPETokenizer (V13-compatible)
│   └── io.py                      ← read_fasta, write_tsv, write_jsonl
│
├── place/                         ← Phylogenetic placement (replaces pplacer/GTDB-Tk)
│   ├── PlacementEngine            (nearest-prototype search)
│   ├── ReferenceDB                (prototype bank with Karcher aggregation)
│   ├── build_reference.py         (manifest + checkpoint → reference DB)
│   ├── PlacementCalibrator        (conformal prediction sets)
│   └── CLI: atlas-place
│       ├── place                  (FASTA → placement TSV/JSON)
│       └── build-ref              (manifest → reference .pkl)
│
├── tree/                          ← Tree construction (replaces RAxML alignment step)
│   ├── build_tree()               (embeddings → PhyloTree + quartet report)
│   ├── neighbor_joining()         (geodesic NJ on Poincaré distances)
│   ├── check_quartet_consistency()(topology validation)
│   ├── four_point_delta()         (Gromov δ-hyperbolicity)
│   ├── estimate_tree_quality()    (embedding space treeness)
│   ├── PhyloTree / TreeNode       (data structures)
│   ├── export: Newick, JSON, SVG  (standard phylogenetics formats)
│   └── CLI: atlas-tree
│       ├── build                  (embeddings → tree file)
│       ├── check                  (quartet consistency audit)
│       └── info                   (tree summary)
│
├── chimera/                       ← Chimera detection (replaces UCHIME/ChimeraSlayer)
│   ├── detect_chimeras()          (FASTA → chimera calls + coordinates)
│   ├── score_chimera()            (sub-embedding variance + bimodality)
│   ├── ChimeraScore / ChimeraResult
│   └── CLI: atlas-chimera
│
├── hplg/                          ← Calibrated classification with coverage guarantees
│   ├── HPLGClassifier             (3-zone: accept/escalate/fallback)
│   ├── MondrianConformalCalibrator(formal coverage ≥ 1−ε)
│   ├── DualBankPrototypes         (teacher + student EMA)
│   ├── CurvatureAdapter           (κ phase-aware thresholds)
│   ├── Taxonomy / Rank            (GTDB-style hierarchy)
│   └── CLI: atlas-hplg
│
└── novelty/                       ← Novel organism detection
    ├── detect_novel_sequences()   (FASTA → novelty calls)
    ├── score_embedding_novelty()  (geodesic distance to nearest prototype)
    ├── estimate_threshold_from_reference() (auto-calibrated from leave-one-out)
    ├── NoveltyResult
    └── CLI: atlas-novelty detect
