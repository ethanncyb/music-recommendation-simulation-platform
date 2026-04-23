"""Tests for :mod:`src.echosphere.vector_store`.

These tests use an in-memory ChromaDB client so they don't touch
``data/chroma``. If ``chromadb`` is not installed they are skipped.
"""

from __future__ import annotations

import csv
import os

import pytest

chromadb = pytest.importorskip("chromadb")

from src.echosphere import vector_store as vs


SAMPLE_ROWS = [
    {
        "id": "1",
        "title": "Sunrise City",
        "artist": "Neon Echo",
        "genre": "pop",
        "mood": "happy",
        "energy": "0.82",
        "tempo_bpm": "118",
        "valence": "0.84",
        "danceability": "0.79",
        "acousticness": "0.18",
        "instrumentalness": "0.02",
        "speechiness": "0.05",
        "popularity": "75",
        "release_year": "2023",
        "key_signature": "C Major",
        "time_signature": "4",
        "detailed_moods": "happy|upbeat|bright",
    },
    {
        "id": "17",
        "title": "Iron Curtain",
        "artist": "Nocturn",
        "genre": "metal",
        "mood": "angry",
        "energy": "0.95",
        "tempo_bpm": "168",
        "valence": "0.35",
        "danceability": "0.58",
        "acousticness": "0.06",
        "instrumentalness": "0.08",
        "speechiness": "0.15",
        "popularity": "52",
        "release_year": "2016",
        "key_signature": "D Minor",
        "time_signature": "4",
        "detailed_moods": "angry|heavy|dark",
    },
    {
        "id": "4",
        "title": "Library Rain",
        "artist": "Paper Lanterns",
        "genre": "lofi",
        "mood": "chill",
        "energy": "0.35",
        "tempo_bpm": "72",
        "valence": "0.60",
        "danceability": "0.58",
        "acousticness": "0.86",
        "instrumentalness": "0.90",
        "speechiness": "0.04",
        "popularity": "65",
        "release_year": "2022",
        "key_signature": "D Major",
        "time_signature": "4",
        "detailed_moods": "chill|peaceful|rainy",
    },
]


@pytest.fixture()
def fake_csv(tmp_path):
    path = tmp_path / "songs.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SAMPLE_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(SAMPLE_ROWS)
    return str(path)


@pytest.fixture()
def ephemeral_client():
    """An in-memory Chroma client — no disk writes."""
    return chromadb.EphemeralClient()


def test_feature_vector_dim_and_normalisation():
    vec = vs.song_feature_vector({
        "energy": 0.5,
        "tempo_bpm": 120,
        "valence": 0.5,
        "danceability": 0.5,
        "acousticness": 0.5,
        "instrumentalness": 0.5,
        "speechiness": 0.5,
    })
    assert len(vec) == vs.EMBEDDING_DIM == 7
    # Tempo 120 with MIN=40, MAX=200 -> (120-40)/160 = 0.5
    assert vec[1] == pytest.approx(0.5, abs=1e-6)


def test_build_query_vector_defaults():
    vec = vs.build_query_vector({})
    assert len(vec) == vs.EMBEDDING_DIM
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_ingest_catalog_populates_collection(fake_csv, ephemeral_client):
    summary = vs.ingest_catalog(csv_path=fake_csv, client=ephemeral_client)
    assert summary["count"] == len(SAMPLE_ROWS)
    assert summary["embedding_dim"] == vs.EMBEDDING_DIM

    collection = ephemeral_client.get_collection(
        name=vs.COLLECTION_NAME,
        embedding_function=vs.FeatureVectorEmbedding(),
    )
    assert collection.count() == len(SAMPLE_ROWS)


def test_query_returns_closest_pop_track_first(fake_csv, ephemeral_client):
    vs.ingest_catalog(csv_path=fake_csv, client=ephemeral_client)
    collection = ephemeral_client.get_collection(
        name=vs.COLLECTION_NAME,
        embedding_function=vs.FeatureVectorEmbedding(),
    )

    high_energy_pop_dna = {
        "energy": 0.82,
        "tempo_bpm": 118,
        "valence": 0.84,
        "danceability": 0.79,
        "acousticness": 0.18,
        "instrumentalness": 0.02,
        "speechiness": 0.05,
    }
    result = collection.query(
        query_embeddings=[vs.build_query_vector(high_energy_pop_dna)],
        n_results=3,
    )
    top_titles = [m["title"] for m in result["metadatas"][0]]
    assert top_titles[0] == "Sunrise City"
