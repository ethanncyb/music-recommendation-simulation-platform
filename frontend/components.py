"""
Shared UI components and state management for the Streamlit frontend.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.recommender import (
    load_songs, recommend_songs, score_song,
    DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED, RankingStrategy,
)
from src.rag import load_knowledge
from src.confidence import ConfidenceScorer, ConfidenceReport
from src.guardrails import apply_guardrails


STRATEGIES = {
    "Default": DEFAULT,
    "Genre-First": GENRE_FIRST,
    "Mood-First": MOOD_FIRST,
    "Energy-Focused": ENERGY_FOCUSED,
}

MODES = {
    "Fast (deterministic)": "fast",
    "Agentic (EchoSphere-RAG)": "agentic",
}


def load_shared_state():
    """Initialize session_state with shared data (songs, knowledge, etc.)."""
    if "songs" not in st.session_state:
        st.session_state.songs = load_songs("data/songs.csv")

    if "knowledge" not in st.session_state:
        try:
            st.session_state.knowledge = load_knowledge()
        except Exception:
            st.session_state.knowledge = None

    if "valid_genres" not in st.session_state:
        st.session_state.valid_genres = sorted(
            set(s["genre"] for s in st.session_state.songs)
        )

    if "valid_moods" not in st.session_state:
        st.session_state.valid_moods = sorted(
            set(s["mood"] for s in st.session_state.songs)
        )

    if "confidence_scorer" not in st.session_state:
        st.session_state.confidence_scorer = ConfidenceScorer(st.session_state.songs)

    if "mode" not in st.session_state:
        st.session_state.mode = "fast"


def render_mode_selector(key: str = "mode_selector") -> str:
    """Sidebar toggle between fast and agentic modes. Returns 'fast'|'agentic'."""
    label = st.sidebar.radio(
        "Recommendation mode",
        list(MODES.keys()),
        key=key,
        help=(
            "Fast = deterministic weighted scoring (no LLM). "
            "Agentic = EchoSphere-RAG LangGraph over ChromaDB + Ollama."
        ),
    )
    mode = MODES[label]
    st.session_state.mode = mode
    return mode


def profile_to_dna(user_prefs: Dict) -> Dict:
    """Map a fast-mode preferences dict onto an EchoSphere DNA profile.

    Mirrors ``src.main.profile_to_dna`` so both entry points produce the same
    DNA for the same preferences.
    """
    from src.echosphere.state import DEFAULT_DNA_PROFILE

    dna = dict(DEFAULT_DNA_PROFILE)
    energy = float(user_prefs.get("energy", dna["energy"]))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    dna.update({
        "genre": user_prefs.get("genre"),
        "mood": user_prefs.get("mood"),
        "energy": energy,
        "likes_acoustic": likes_acoustic,
        "tempo_bpm": 70.0 + energy * 80.0,
        "valence": max(0.0, min(1.0, 0.3 + energy * 0.6)),
        "danceability": max(0.0, min(1.0, 0.4 + energy * 0.5)),
        "acousticness": 0.75 if likes_acoustic else max(0.0, 0.35 - energy * 0.3),
        "instrumentalness": 0.6 if likes_acoustic else 0.2,
        "speechiness": 0.08,
        "top_k": int(user_prefs.get("top_k", 5)),
    })
    return dna


def run_agentic(user_prefs: Dict, query: Optional[str] = None) -> Dict:
    """Execute the EchoSphere-RAG pipeline and return the final state dict."""
    from src.echosphere import run_echosphere

    dna = profile_to_dna(user_prefs)
    if not query:
        query = (
            f"Recommend {dna.get('genre') or 'music'} with "
            f"{dna.get('mood') or 'any'} mood at energy {dna['energy']}."
        )
    return run_echosphere(query, dna)


def render_agentic_results(state: Dict):
    """Render an EchoSphere state dict as a recommendations block."""
    if state.get("error"):
        st.error(f"Pipeline error: {state['error']}")

    retrieved = state.get("retrieved_tracks") or []
    explanations = state.get("explanations") or []
    trivia = state.get("artist_trivia") or {}

    if not retrieved:
        st.info(
            "No tracks returned. Seed the vector store with "
            "`python -m src.echosphere.vector_store` and confirm Ollama is running."
        )
        return

    rows = []
    for rank, track in enumerate(retrieved, 1):
        rows.append({
            "#": rank,
            "Title": track.get("title"),
            "Artist": track.get("artist"),
            "Genre": track.get("genre"),
            "Mood": track.get("mood"),
            "Energy": f"{float(track.get('energy', 0)):.2f}",
            "Distance": f"{float(track.get('distance') or 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for rank, track in enumerate(retrieved, 1):
        with st.expander(f"#{rank} — {track.get('title')} by {track.get('artist')}"):
            if rank - 1 < len(explanations):
                st.markdown(f"**Why:** {explanations[rank - 1]}")
            artist_fact = trivia.get(track.get("artist", ""))
            if artist_fact:
                st.caption(f"Artist trivia: {artist_fact}")


def render_results_table(results: List[Tuple]):
    """Render recommendation results as a styled dataframe."""
    if not results:
        st.info("No results to display.")
        return

    rows = []
    for rank, (song, score, explanation) in enumerate(results, 1):
        rows.append({
            "#": rank,
            "Title": song["title"],
            "Artist": song["artist"],
            "Genre": song["genre"],
            "Mood": song["mood"],
            "Energy": f"{song['energy']:.2f}",
            "Score": f"{score:.4f}",
            "Reasons": explanation.replace("; ", "\n"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_confidence_badge(report: ConfidenceReport):
    """Display confidence as a metric with appropriate styling."""
    label_colors = {
        "high": "normal",
        "medium": "off",
        "low": "off",
        "very_low": "off",
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            label="Confidence",
            value=f"{report.overall_confidence:.2f}",
            delta=report.confidence_label.upper(),
            delta_color=label_colors.get(report.confidence_label, "off"),
        )
    with col2:
        if report.warnings:
            for w in report.warnings:
                st.warning(w, icon="⚠️")
        if report.suggestion:
            st.info(f"Suggestion: {report.suggestion}")


def render_strategy_selector(key: str = "strategy") -> RankingStrategy:
    """Radio buttons for strategy selection. Returns the selected strategy."""
    name = st.radio(
        "Ranking Strategy",
        list(STRATEGIES.keys()),
        horizontal=True,
        key=key,
    )
    return STRATEGIES[name]


def check_ollama_available() -> bool:
    """Check if Ollama is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False
