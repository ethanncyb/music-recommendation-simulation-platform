"""
ChromaDB vector-store adapter for EchoSphere-RAG.

We intentionally embed tracks using their *audio feature vector* instead of a
text embedding model. The new design calls for "mathematical matches" on
physical audio features (energy, tempo, acousticness, instrumentalness,
speechiness, ...), so a direct numeric embedding is both faithful to the spec
and keeps the stack fully offline (no sentence-transformers download).

The feature vector (7-dim) used for both ingestion and queries is:

    [energy, tempo_norm, valence, danceability,
     acousticness, instrumentalness, speechiness]

where ``tempo_norm`` is min-max normalised to [0, 1] using ``TEMPO_MIN`` /
``TEMPO_MAX`` constants so it is comparable to the other 0..1 features.

Entry points
------------
- ``get_collection(persist_dir=...)`` — idempotent, returns the Chroma
  collection, auto-seeding from ``data/songs.json`` on first access.
- ``ingest_catalog(catalog_path, client=None)`` — (re)builds the collection from a
  JSON catalog file. Safe to call repeatedly.
- ``build_query_vector(dna_profile)`` — converts a DNA profile into the same
  7-dim vector used for documents.
- ``python -m src.echosphere.vector_store`` — CLI seeding helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # Optional at import time so unit tests can mock the module.
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except Exception:  # pragma: no cover - chromadb missing in some envs
    chromadb = None  # type: ignore[assignment]
    ClientAPI = Any  # type: ignore[assignment,misc]
    Collection = Any  # type: ignore[assignment,misc]
    Documents = Any  # type: ignore[assignment,misc]
    Embeddings = Any  # type: ignore[assignment,misc]

    class EmbeddingFunction:  # type: ignore[no-redef]
        """Fallback stub so type references work when chromadb isn't installed."""

        def __call__(self, input: Any) -> Any:  # pragma: no cover
            raise RuntimeError("chromadb is not installed")


# ── Feature encoding ────────────────────────────────────────────────────────

FEATURE_KEYS: Sequence[str] = (
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
)
TEMPO_MIN = 40.0
TEMPO_MAX = 200.0
EMBEDDING_DIM = len(FEATURE_KEYS)


def _normalise_tempo(tempo_bpm: float) -> float:
    """Min-max normalise tempo into [0, 1] with clamping."""
    value = (float(tempo_bpm) - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def song_feature_vector(row: Dict[str, Any]) -> List[float]:
    """Convert a catalog row dict (strings or floats) into the 7-dim vector."""
    return [
        float(row["energy"]),
        _normalise_tempo(row["tempo_bpm"]),
        float(row["valence"]),
        float(row["danceability"]),
        float(row["acousticness"]),
        float(row["instrumentalness"]),
        float(row["speechiness"]),
    ]


def build_query_vector(dna_profile: Dict[str, Any]) -> List[float]:
    """Convert a DNA profile mapping into the same 7-dim vector space."""
    return [
        float(dna_profile.get("energy", 0.5)),
        _normalise_tempo(dna_profile.get("tempo_bpm", 110.0)),
        float(dna_profile.get("valence", 0.5)),
        float(dna_profile.get("danceability", 0.5)),
        float(dna_profile.get("acousticness", 0.5)),
        float(dna_profile.get("instrumentalness", 0.5)),
        float(dna_profile.get("speechiness", 0.1)),
    ]


# ── No-op embedding function ────────────────────────────────────────────────

class FeatureVectorEmbedding(EmbeddingFunction):
    """Placeholder embedding function.

    Chroma requires every collection to have *some* embedding function. We
    always pass ``embeddings=`` explicitly on add/query (because our vectors
    are deterministic numeric features, not text), so this callable should
    never actually run. If it does, we surface a clear error rather than
    silently downloading a default text-embedding model.
    """

    def __init__(self) -> None:  # noqa: D401 - Chroma >=0.5 requires __init__
        super().__init__()

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        raise RuntimeError(
            "FeatureVectorEmbedding was invoked without pre-computed vectors. "
            "Always pass embeddings=/query_embeddings= explicitly."
        )

    # Chroma >=0.5 probes for a stable name to persist with the collection.
    @classmethod
    def name(cls) -> str:  # pragma: no cover - trivial
        return "echosphere-feature-vector"

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        """Return a serialisable config blob for Chroma's schema registry."""
        return {"type": "echosphere-feature-vector", "dim": EMBEDDING_DIM}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "FeatureVectorEmbedding":  # pragma: no cover
        return FeatureVectorEmbedding()


# ── Client / collection helpers ─────────────────────────────────────────────

DEFAULT_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "echosphere_tracks"
DEFAULT_CATALOG_PATH = "data/songs.json"

_CACHED_CLIENT: Optional[ClientAPI] = None
_CACHED_COLLECTION: Optional[Collection] = None


def _require_chromadb() -> None:
    if chromadb is None:  # pragma: no cover
        raise ImportError(
            "chromadb is not installed. Run `pip install -r requirements.txt`."
        )


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> ClientAPI:
    """Return a persistent Chroma client, cached per-process."""
    global _CACHED_CLIENT
    _require_chromadb()
    if _CACHED_CLIENT is None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _CACHED_CLIENT = chromadb.PersistentClient(path=persist_dir)
    return _CACHED_CLIENT


def get_collection(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    catalog_path: str = DEFAULT_CATALOG_PATH,
    client: Optional[ClientAPI] = None,
    auto_ingest: bool = True,
) -> Collection:
    """Return the EchoSphere collection, seeding it from JSON if empty."""
    global _CACHED_COLLECTION
    _require_chromadb()
    if client is None:
        client = get_client(persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=FeatureVectorEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
    if auto_ingest and collection.count() == 0 and Path(catalog_path).exists():
        ingest_catalog(catalog_path=catalog_path, client=client, collection=collection)
    _CACHED_COLLECTION = collection
    return collection


# ── Ingestion ───────────────────────────────────────────────────────────────

_NUMERIC_FIELDS = {
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
}
_INT_FIELDS = {"popularity", "release_year", "time_signature", "id"}


def _coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert catalog fields into floats / ints where appropriate."""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = value
            continue
        if key in _NUMERIC_FIELDS:
            out[key] = float(value)
        elif key in _INT_FIELDS:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
        else:
            out[key] = value
    return out


def _load_rows(catalog_path: str) -> List[Dict[str, Any]]:
    with open(catalog_path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [_coerce_row(row) for row in rows]


def ingest_catalog(
    catalog_path: str = DEFAULT_CATALOG_PATH,
    client: Optional[ClientAPI] = None,
    collection: Optional[Collection] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Dict[str, Any]:
    """Idempotently (re)build the EchoSphere collection from ``catalog_path``.

    Returns a small summary dict with the collection size and embedding dim.
    """
    _require_chromadb()
    if client is None:
        client = get_client(persist_dir)
    if collection is None:
        # Recreate to guarantee a clean state on re-ingest.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=FeatureVectorEmbedding(),
            metadata={"hnsw:space": "cosine"},
        )

    rows = _load_rows(catalog_path)
    ids = [str(row["id"]) for row in rows]
    embeddings = [song_feature_vector(row) for row in rows]
    documents = [
        f"{row['title']} by {row['artist']} ({row['genre']}, {row['mood']})"
        for row in rows
    ]
    # Chroma metadatas must be str/int/float/bool. Drop Nones just in case.
    metadatas = [
        {k: v for k, v in row.items() if v is not None and v != ""}
        for row in rows
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return {
        "collection": COLLECTION_NAME,
        "count": collection.count(),
        "embedding_dim": EMBEDDING_DIM,
        "catalog_path": os.path.abspath(catalog_path),
    }


# ── CLI entry ───────────────────────────────────────────────────────────────

def _main() -> None:
    """``python -m src.echosphere.vector_store`` — seed the Chroma DB."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed the EchoSphere ChromaDB.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG_PATH, help="Catalog JSON path")
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="ChromaDB persistence directory",
    )
    args = parser.parse_args()

    summary = ingest_catalog(catalog_path=args.catalog, persist_dir=args.persist_dir)
    print("EchoSphere-RAG ChromaDB seeded:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _main()


__all__ = [
    "FEATURE_KEYS",
    "EMBEDDING_DIM",
    "COLLECTION_NAME",
    "FeatureVectorEmbedding",
    "song_feature_vector",
    "build_query_vector",
    "get_client",
    "get_collection",
    "ingest_catalog",
]
