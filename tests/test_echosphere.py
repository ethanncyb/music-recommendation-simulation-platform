"""Tests for the EchoSphere-RAG LangGraph pipeline.

Covers all three agent nodes individually plus the compiled graph. External
dependencies (ChromaDB and ChatOllama) are replaced with in-process fakes so
the suite runs offline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.echosphere import nodes as echo_nodes
from src.echosphere.state import DEFAULT_DNA_PROFILE


# ── Test fixtures ───────────────────────────────────────────────────────────

IRON_CURTAIN = {
    "id": "17",
    "title": "Iron Curtain",
    "artist": "Nocturn",
    "genre": "metal",
    "mood": "angry",
    "energy": 0.95,
    "tempo_bpm": 168,
    "valence": 0.35,
    "danceability": 0.58,
    "acousticness": 0.06,
    "instrumentalness": 0.08,
    "speechiness": 0.15,
}

LIBRARY_RAIN = {
    "id": "4",
    "title": "Library Rain",
    "artist": "Paper Lanterns",
    "genre": "lofi",
    "mood": "chill",
    "energy": 0.35,
    "tempo_bpm": 72,
    "valence": 0.60,
    "danceability": 0.58,
    "acousticness": 0.86,
    "instrumentalness": 0.90,
    "speechiness": 0.04,
}

RAP_TRACK = {
    "id": "11",
    "title": "Golden Chain",
    "artist": "Flow State",
    "genre": "hip-hop",
    "mood": "confident",
    "energy": 0.78,
    "tempo_bpm": 95,
    "valence": 0.72,
    "danceability": 0.85,
    "acousticness": 0.08,
    "instrumentalness": 0.00,
    "speechiness": 0.45,
}

POP_TRACK = {
    "id": "1",
    "title": "Sunrise City",
    "artist": "Neon Echo",
    "genre": "pop",
    "mood": "happy",
    "energy": 0.82,
    "tempo_bpm": 118,
    "valence": 0.84,
    "danceability": 0.79,
    "acousticness": 0.18,
    "instrumentalness": 0.02,
    "speechiness": 0.05,
}


class FakeCollection:
    """Minimal stand-in for a Chroma collection."""

    def __init__(self, tracks: List[Dict[str, Any]]):
        self._tracks = tracks

    def query(self, query_embeddings, n_results=10, **_kwargs):  # noqa: D401
        tracks = self._tracks[:n_results]
        return {
            "ids": [[str(t["id"]) for t in tracks]],
            "metadatas": [tracks],
            # Distance from index so the order is deterministic.
            "distances": [[0.1 * (i + 1) for i in range(len(tracks))]],
        }


@pytest.fixture()
def patch_collection(monkeypatch):
    """Return a helper that installs a FakeCollection for the ingestor node."""

    def _install(tracks):
        fake = FakeCollection(tracks)
        monkeypatch.setattr(
            echo_nodes,
            "get_collection",
            lambda *args, **kwargs: fake,
        )
        return fake

    return _install


# ── Ingestor node ──────────────────────────────────────────────────────────

def test_ingestor_filters_vocals_from_instrumental_query(patch_collection):
    """A high-instrumentalness DNA should exclude a high-speechiness track.

    This mirrors the design doc example: an "angry, low-acousticness metal
    track like Iron Curtain" must stay strictly separated from a "mellow,
    acoustic track like Library Rain" when the user asks for instrumental
    focus music.
    """
    patch_collection([LIBRARY_RAIN, IRON_CURTAIN, RAP_TRACK])

    state = {
        "user_request": "instrumental calm focus",
        "dna_profile": {
            **DEFAULT_DNA_PROFILE,
            "instrumentalness": 0.9,
            "speechiness": 0.05,
            "acousticness": 0.8,
            "likes_acoustic": True,
            "top_k": 5,
        },
    }
    out = echo_nodes.ingestor_node(state)
    titles = [t["title"] for t in out["retrieved_tracks"]]
    assert "Library Rain" in titles
    assert "Iron Curtain" not in titles  # fails instrumentalness >= 0.4 filter
    assert "Golden Chain" not in titles  # fails speechiness <= 0.25 filter


def test_ingestor_returns_top_k_when_no_filters_configured(patch_collection):
    patch_collection([POP_TRACK, IRON_CURTAIN, LIBRARY_RAIN])
    state = {
        "user_request": "anything",
        "dna_profile": {
            **DEFAULT_DNA_PROFILE,
            "instrumentalness": 0.2,  # disable instrumental filter
            "speechiness": 0.5,       # disable speech filter
            "likes_acoustic": False,
            "top_k": 2,
        },
    }
    out = echo_nodes.ingestor_node(state)
    assert len(out["retrieved_tracks"]) == 2
    # distances should be attached and monotonically increasing (fake order)
    distances = [t["distance"] for t in out["retrieved_tracks"]]
    assert distances == sorted(distances)


# ── Researcher node ────────────────────────────────────────────────────────

def test_researcher_appends_trivia_for_every_artist():
    state = {
        "retrieved_tracks": [POP_TRACK, IRON_CURTAIN, {"artist": "Unknown Band", "title": "x"}],
    }
    out = echo_nodes.researcher_node(state)
    trivia = out["artist_trivia"]
    assert "Neon Echo" in trivia and trivia["Neon Echo"]
    assert "Nocturn" in trivia and trivia["Nocturn"]
    # Unknown artist still gets an entry (placeholder string).
    assert "Unknown Band" in trivia
    assert "No trivia on file" in trivia["Unknown Band"]


def test_researcher_handles_empty_retrieved_tracks():
    out = echo_nodes.researcher_node({"retrieved_tracks": []})
    assert out == {"artist_trivia": {}}


# ── Reasoning node ─────────────────────────────────────────────────────────

class FakeChatOllama:
    """Records inputs and returns a formatted string."""

    def __init__(self):
        self.calls: List[List[Any]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        # Pull the last HumanMessage content for echoing.
        user_content = messages[-1].content
        return type("Resp", (), {"content": f"Because: {user_content[:40]}..."})()


def test_reasoning_node_uses_provided_llm_per_track():
    fake = FakeChatOllama()
    state = {
        "user_request": "high-energy pop",
        "dna_profile": {**DEFAULT_DNA_PROFILE, "genre": "pop", "energy": 0.9},
        "retrieved_tracks": [POP_TRACK, IRON_CURTAIN],
        "artist_trivia": {"Neon Echo": "debut on Game Boy", "Nocturn": "heavy rig"},
    }
    out = echo_nodes.reasoning_node(state, llm=fake)
    assert len(out["explanations"]) == 2
    assert all(isinstance(e, str) and e for e in out["explanations"])
    # LLM was invoked once per track.
    assert len(fake.calls) == 2


def test_reasoning_node_empty_tracks_returns_empty_explanations():
    out = echo_nodes.reasoning_node(
        {"user_request": "x", "dna_profile": {}, "retrieved_tracks": []},
        llm=FakeChatOllama(),
    )
    assert out == {"explanations": []}


# ── Full graph ─────────────────────────────────────────────────────────────

def test_graph_compiles_and_runs_end_to_end(monkeypatch, patch_collection):
    pytest.importorskip("langgraph")

    patch_collection([POP_TRACK, IRON_CURTAIN, LIBRARY_RAIN])

    fake = FakeChatOllama()
    monkeypatch.setattr(echo_nodes, "_build_llm", lambda: fake)

    from src.echosphere.graph import build_graph

    graph = build_graph()
    final_state = graph.invoke({
        "user_request": "I need a high-energy pop track",
        "dna_profile": {
            **DEFAULT_DNA_PROFILE,
            "genre": "pop",
            "energy": 0.9,
            "instrumentalness": 0.1,
            "speechiness": 0.1,
            "top_k": 2,
        },
    })

    assert "retrieved_tracks" in final_state and final_state["retrieved_tracks"]
    assert "artist_trivia" in final_state and final_state["artist_trivia"]
    assert "explanations" in final_state and final_state["explanations"]
    # Every retrieved track has a matching explanation (order-preserving).
    assert len(final_state["explanations"]) == len(final_state["retrieved_tracks"])
    # Iron Curtain has speechiness < 0.25 but pop+low-instrumental query means
    # it is NOT excluded by any filter; the graph just orders by cosine distance.
    # We only assert the pop track is in the result set.
    titles = [t["title"] for t in final_state["retrieved_tracks"]]
    assert "Sunrise City" in titles


# ── ConfidenceScorer integration ───────────────────────────────────────────

def test_confidence_scorer_accepts_agentic_state():
    from src.confidence import ConfidenceScorer

    scorer = ConfidenceScorer([POP_TRACK, IRON_CURTAIN, LIBRARY_RAIN])
    state = {
        "retrieved_tracks": [
            {**POP_TRACK, "distance": 0.05},
            {**IRON_CURTAIN, "distance": 0.30},
        ],
        "explanations": [
            "High energy and genre pop match the request.",
            "Lower mood overlap; acoustic low.",
        ],
    }
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False}
    report = scorer.compute(prefs, state)
    assert 0.0 <= report.overall_confidence <= 1.0
    assert report.confidence_label in {"high", "medium", "low", "very_low"}
