"""Unified BiosphereAtlas tool suite."""

__version__ = "0.1.0"

from biosphere_atlas.core.atlas import Atlas
from biosphere_atlas.novelty import detect_novel_sequences
from biosphere_atlas.dark import triage_genomes
from biosphere_atlas.viewer import from_tree, from_embeddings, generate_viewer_html

__all__ = ["Atlas", "detect_novel_sequences", "triage_genomes", "from_tree", "from_embeddings", "generate_viewer_html"]

