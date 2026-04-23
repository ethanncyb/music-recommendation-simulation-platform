"""
Shared graph state for the EchoSphere-RAG pipeline.

The state is a ``TypedDict`` so LangGraph can validate field names and merge
partial updates returned from each node.

Fields
------
user_request
    The raw natural-language query the user typed (e.g. "I need a high-energy
    pop track").
dna_profile
    The user's specialised music-taste "DNA profile". A mapping of audio-feature
    targets plus optional categorical preferences. See ``DEFAULT_DNA_PROFILE``
    for the canonical keys.
retrieved_tracks
    Tracks returned by the Ingestor node after vector search + post-filtering.
    Each entry is a ``dict`` with the full catalog row plus a ``distance`` key
    reporting the Chroma distance.
artist_trivia
    Mapping of ``artist_name -> trivia_string`` added by the Researcher node.
explanations
    DJ-style per-track explanations produced by the Reasoning node. The list is
    index-aligned with ``retrieved_tracks``.
error
    Populated by any node that fails soft (missing collection, LLM timeout,
    etc.). Downstream nodes skip their work when ``error`` is set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class EchoState(TypedDict, total=False):
    """TypedDict flowing through the EchoSphere-RAG LangGraph."""

    user_request: str
    dna_profile: Dict[str, Any]
    retrieved_tracks: List[Dict[str, Any]]
    artist_trivia: Dict[str, str]
    explanations: List[str]
    error: Optional[str]


# Canonical DNA-profile keys. Callers only need to override what they care about;
# ``ingestor_node`` will fill in the rest from these defaults so the query vector
# always has the expected dimensionality.
DEFAULT_DNA_PROFILE: Dict[str, Any] = {
    # Audio-feature targets (all 0..1 except tempo which is BPM)
    "energy": 0.6,
    "tempo_bpm": 110.0,
    "valence": 0.6,
    "danceability": 0.6,
    "acousticness": 0.4,
    "instrumentalness": 0.3,
    "speechiness": 0.1,
    # Categorical / soft preferences used for post-filtering + prompting
    "genre": None,
    "mood": None,
    "likes_acoustic": False,
    # How many tracks the Ingestor should return
    "top_k": 5,
}


__all__ = ["EchoState", "DEFAULT_DNA_PROFILE"]
