# Coherence Playbook

**How to read the manifold — and how to let the manifold read itself.**

This document maps every diagnostic question you might ask about a BiosphereAtlas
checkpoint to the tool, metric, and interpretation that answers it. It also
describes how these tools can be integrated into the training loop itself,
turning passive observation into active geometric guidance.

---

## Part I: Question → Tool → Metric

### "Is the geometry collapsing?"

| Tool | Metric | Healthy | Unhealthy |
|------|--------|---------|-----------|
| **GCAS** | `r_eff` (effective rank) | 20–40 | < 10 (degenerate) or > 80 (diffuse) |
| **GCAS** | `variance_2d` | 25–45% | > 70% (flat pancake) or < 10% (isotropic blob) |
| **atlas-tree** | `delta` (hyperbolicity) | < 0.25 | > 0.40 (not tree-like anymore) |
| **atlas-tree** | quartet consistency | > 65% | < 50% (random) |

### "Is κ behaving?"

| Tool | Metric | Healthy | Unhealthy |
|------|--------|---------|-----------|
| **GCAS** | `kappa` | 0.9–1.02 (Phase B), ~1.25 (Phase C) | > 1.4 (gas phase) |
| **Training log** | κ drift per epoch | < 0.005/epoch | > 0.02/epoch (runaway) |
| **Heads vs encoder** | `|head_κ - encoder_κ|` | 0 (V13+ dynamic heads) | > 0.1 (V12-style impedance) |

### "Is the model actually learning domain X?"

| Tool | Metric | Healthy | Unhealthy |
|------|--------|---------|-----------|
| **atlas-place** | domain-specific retrieval top1 | rising epoch-over-epoch | flat at baseline |
| **Training log** | domain family accuracy | rising | stuck at `1/num_classes` |
| **atlas-dark** | % dark prototypes in domain | decreasing | stable or increasing |

### "Are the embeddings more informative than the classification heads suggest?"

| Tool | Metric | Signal |
|------|--------|--------|
| **atlas-place** | retrieval top1 vs head accuracy | If retrieval > head: geometry has untapped structure |
| **atlas-tree** | quartet consistency | > 65% means branching hierarchy exists regardless of head accuracy |
| **atlas-place** | inter/intra distance ratio | > 1.1 means families are geometrically separable |
| **atlas-place** | margin (correct vs wrong) | Large gap = confident geometry, even if head disagrees |

### "Where should I add training data next?"

| Tool | Metric | Decision |
|------|--------|----------|
| **atlas-dark** | dark regions list | Families in dark regions need more training examples |
| **atlas-dark triage** | novel_certain count | These genomes extend coverage without ambiguity |
| **atlas-dark triage** | novel_uncertain count | These genomes are in uncharted territory — highest value if labeled |
| **atlas-place** | per-domain retrieval accuracy | The domain with lowest retrieval needs the most data |

### "Is a new checkpoint better than the last one?"

| Tool | Metric | Better if... |
|------|--------|-------------|
| **atlas-place** | overall top1/top5 | higher |
| **atlas-tree** | quartet consistency | higher |
| **atlas-tree** | delta | lower |
| **atlas-dark** | % charted | higher |
| **atlas-dark** | mean sigma (charted) | lower (tighter coverage) |
| **GCAS** | r_eff | stable or increasing (not collapsing) |

---

## Part II: Training-Loop Integration

### The insight

GCAS already runs inside the training loop as a periodic probe. But GCAS only
measures bulk geometry statistics (SVD of embedding samples). The atlas tools
measure *relational* properties — how embeddings relate to each other
taxonomically. These are the properties that actually matter for downstream
utility, and they can be measured cheaply during training.

### The key realization

The training loop already has everything atlas-place needs:
- **Prototypes** are the `HyperbolicMLR.prototypes` parameters (already in GPU memory)
- **Query embeddings** come from the validation forward pass (already computed)
- **κ** is `model.live_kappa` (already tracked)

No reference DB needs to be built. No files need to be loaded. The "placement"
is just: compute geodesic distance from each val embedding to each prototype,
find the nearest one, check if its label matches. This is essentially what the
classification head already does — but without the softmax, without the loss
function, and without the gradient. Pure geometric retrieval.

### Proposed: `GeometricCoherenceProbe`

A lightweight module that runs every N epochs (alongside or replacing GCAS)
and reports metrics from the atlas tool suite. Sketch:

```python
@torch.no_grad()
def geometric_coherence_probe(
    model, val_loader, device, log,
    n_samples=500, n_quartets=1000,
):
    """
    Periodic geometric health check using atlas-tool primitives.
    Runs inside the training loop — no disk I/O, no reference DB.
    """
    model.eval()
    kappa = model.live_kappa

    # ── 1. Collect embeddings + labels from val set ──────────────
    embeddings, domains, families = [], [], []
    for batch in val_loader:
        z = model.encode(batch["tokens"].to(device)).float().cpu()
        embeddings.append(z)
        domains.append(batch["domain_idx"])
        families.append(batch["fam_idx"])
        if sum(e.shape[0] for e in embeddings) >= n_samples:
            break
    embeddings = torch.cat(embeddings)[:n_samples]
    domains = torch.cat(domains)[:n_samples]
    families = torch.cat(families)[:n_samples]

    # ── 2. Place-probe: pure geometric retrieval ─────────────────
    #    Use prototype parameters directly from the model
    for d, (name, head) in enumerate([
        ("Bact", model.bact_fam),
        ("Arch", model.arch_fam),
        ("Euk", model.euk_fam),
    ]):
        mask = domains == d
        if not mask.any():
            continue
        z_d = embeddings[mask]
        f_d = families[mask]
        protos = head.prototypes.detach().cpu().float()

        # Project both to ball at current kappa
        z_d = project_to_ball(z_d, kappa)
        protos = project_to_ball(protos, kappa)

        # Geodesic nearest-prototype retrieval
        dists = poincare_distance(
            z_d.unsqueeze(1), protos.unsqueeze(0), kappa
        )  # (N, C)
        pred = dists.argmin(dim=1)
        geo_acc = (pred == f_d).float().mean().item()
        
        # Inter/intra separation
        # (subsample for speed)
        ...

        log(f"  GeoProbe {name}: retrieval={geo_acc*100:.1f}%, "
            f"n={mask.sum().item()}")

    # ── 3. Tree-probe: quartet consistency on prototypes ─────────
    #    Collect all family prototypes, build quick NJ tree, 
    #    check quartet consistency
    all_protos = torch.cat([
        model.bact_fam.prototypes.detach().cpu(),
        model.arch_fam.prototypes.detach().cpu(),
        model.euk_fam.prototypes.detach().cpu(),
    ])
    all_protos = project_to_ball(all_protos.float(), kappa)

    # Pairwise distance matrix
    D = poincare_distance(
        all_protos.unsqueeze(1),
        all_protos.unsqueeze(0),
        kappa
    )
    # delta-hyperbolicity on a random subset
    ...

    # ── 4. Dark-probe: uncertainty field from prototypes ─────────
    #    Compute k-NN sigma for each prototype
    #    Report: mean sigma, fraction above dark threshold
    ...

    return metrics
```

### What this gives you during training

Every N epochs, the training log would show something like:

```
[B] Ep 15 | Loss: 0.857 | κ: 1.2502 | Domain: 89.5% | Bact: 84.6 | Arch: 25.8 | Euk: 21.1
  GCAS:     r_eff=29.7, 2D_var=31.1%, κ=1.2502
  GeoProbe: Bact retrieval=86.2%, Arch retrieval=28.1%, Euk retrieval=23.4%
  GeoProbe: quartet_consistency=71.2%, delta=0.19, dark_pct=4.8%
  GeoProbe: inter/intra=1.18, mean_margin=1.42
```

The first line is what the heads say. The GeoProbe lines are what the
geometry says. When these diverge, you know the heads are the bottleneck.
When they converge, you know the geometry is faithfully represented.

### Integration priority

1. **Place-probe** (highest value, lowest cost): Pure geodesic retrieval
   accuracy from existing prototype parameters. Add to training loop immediately.
   Estimated overhead: < 2 seconds per probe on 500 samples.

2. **GCAS** (already integrated): Keep as-is. It measures bulk statistics
   that the other probes don't.

3. **Tree-probe** (medium value, medium cost): Quartet consistency from
   prototype distance matrix. Run every 10-20 epochs. Estimated overhead:
   ~5 seconds for 1000 quartets.

4. **Dark-probe** (strategic value, low cost): k-NN sigma field over
   prototypes. Run every 10-20 epochs. Tells you if the manifold is
   getting denser or sparser in specific regions.

---

## Part III: The Autocatalytic Loop

The full loop, as it now exists:

```
Train (V14)
  ↓
Checkpoint
  ↓
atlas-place build-ref  →  Reference DB
  ↓                          ↓
atlas-tree              atlas-dark map
  ↓                          ↓
quartet consistency     dark regions
tree quality metrics    uncertainty field
  ↓                          ↓
                        atlas-dark triage (on unseen genomes)
                             ↓
                    ┌────────┼────────┐
                    ↓        ↓        ↓
               redundant  novel     novel
               (skip)     certain   uncertain
                          (train)   (flag for review)
                             ↓
                        V15 manifest  ←──  enriched with novel-certain
                             ↓
                        Train (V15)
                             ↓
                           ...
```

Each cycle should produce:
- **More charted regions** (dark fraction decreasing)
- **Higher retrieval accuracy** (especially for minority domains that gained data)
- **Higher quartet consistency** (denser prototype coverage → more resolvable quartets)
- **Classification accuracy** may improve modestly, but the geometric metrics
  should improve faster — because they measure what the manifold actually is,
  not what a softmax head can extract from it.

### The deeper point

Classification accuracy is the *training signal*. Geometric quality is the
*product*. The tools measure the product. Integrating them into the training
loop means the model can see its own product quality in real time, and the
human (or the next training run) can make decisions based on what actually
matters.

This is what "Bioinformatics 2.0" looks like in practice: the model doesn't
just classify sequences — it builds a coordinate system for the tree of life,
and the tools are how that coordinate system is read, validated, and refined.

---

## Part IV: Reference Values from V13/V14

Baseline metrics from the current best checkpoint (V13 living geometry,
κ=1.2502) to compare future runs against:

| Metric | Value | Source |
|--------|-------|--------|
| κ | 1.2502 | Training log |
| Combined accuracy | 43.85 | Training log |
| Bacteria family top1 (retrieval) | 83.2% | atlas-place |
| Archaea family top1 (retrieval) | 19.5% | atlas-place |
| Eukaryota family top1 (retrieval) | 22.0% | atlas-place |
| Overall retrieval top1 | 36.6% | atlas-place |
| Overall retrieval top5 | 48.8% | atlas-place |
| Quartet consistency | 70.2% | atlas-tree (333 prototypes, 5000 quartets) |
| Delta hyperbolicity | 0.197 | atlas-tree |
| Inter/intra distance ratio | 1.162 | atlas-place |
| Dark prototypes | 52 / 1028 (5.1%) | atlas-dark |
| Charted mean sigma | 7.44 | atlas-dark |
| Dark mean sigma | 10.84 | atlas-dark |
| atlas-place throughput | 812 seq/s | atlas-place benchmark |
| Kraken2 family top1 (same test set) | 8.7% | side-by-side comparison |
| Atlas family top1 (same test set) | 39.3% | side-by-side comparison |
| r_eff | 28.2 | GCAS |
| variance_2d | 32.8% | GCAS |

These are the numbers to beat. If a future checkpoint improves retrieval
accuracy and quartet consistency while maintaining κ stability, it's a
strictly better manifold — regardless of what the classification loss says.

