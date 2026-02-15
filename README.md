# biosphere-atlas

Unified Bioinformatics 2.0 toolkit with shared geometry and Atlas loader.

biosphere-atlas v0.1.0
├── core/                          ← The shared geometric foundation
│   ├── hyperbolic.py              ← ONE canonical Poincaré ball geometry
│   │  
│   ├── atlas.py                   ← Atlas.from_checkpoint() — THE unified loader
│   
│
├── place/                         ← Phylogenetic placement (replaces pplacer/GTDB-Tk)
│
├── tree/                          ← Tree construction (replaces RAxML alignment step)
│
│
├── chimera/                       ← Chimera detection (replaces UCHIME/ChimeraSlayer)
│
├── hplg/                          ← Calibrated classification with coverage guarantees

└── novelty/                       ← Novel organism detection
    ├── detect_novel_sequences()   (FASTA → novelty calls)
    ├
