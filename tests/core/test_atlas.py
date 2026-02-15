import torch
from biosphere_atlas.core.atlas import Atlas

def test_atlas_kmer_encode():
    atlas = Atlas()
    emb = atlas.encode("ACGT"*100)
    assert emb.ndim == 1
    assert torch.isfinite(emb).all()
