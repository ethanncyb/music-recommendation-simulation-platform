"""
LangGraph node implementations for EchoSphere-RAG.

Three pure functions that each accept the current ``EchoState`` and return a
partial-state update (LangGraph merges it back into the shared state):

- ``ingestor_node``   — ChromaDB vector search over the audio-feature space,
                         with post-filters on ``instrumentalness`` /
                         ``speechiness`` so a high-instrumental query excludes
                         vocal-heavy tracks ("Iron Curtain" stays separate from
                         "Library Rain" per the design doc).
- ``researcher_node`` — Annotates retrieved artists with mock trivia.
- ``reasoning_node``  — Uses ``langchain_ollama.ChatOllama`` to produce DJ-style
                         per-track explanations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..env_config import load_dotenv
from .state import DEFAULT_DNA_PROFILE, EchoState
from .vector_store import (
    EMBEDDING_DIM,
    build_query_vector,
    get_collection,
)


# ── Ingestor Node ───────────────────────────────────────────────────────────

def _merged_dna(dna: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing DNA fields with defaults so the vector is always 7-dim."""
    merged = dict(DEFAULT_DNA_PROFILE)
    if dna:
        merged.update({k: v for k, v in dna.items() if v is not None})
    return merged


def _passes_feature_filters(meta: Dict[str, Any], dna: Dict[str, Any]) -> bool:
    """Granular post-filters on instrumentalness / speechiness / acousticness.

    The design calls out these features specifically:
      > utilise granular, complex features like Instrumentalness and
      > Speechiness to filter out vocals or separate rap from melody-driven pop.

    We apply three soft rules:
      1. If the user wants highly instrumental tracks
         (``dna['instrumentalness'] >= 0.7``), drop anything with
         ``instrumentalness < 0.4`` — that's how we separate "Library Rain"
         from "Iron Curtain" when the user asks for calm instrumental focus.
      2. If the user explicitly wants low-speech content
         (``dna['speechiness'] <= 0.1``), drop tracks with
         ``speechiness > 0.25`` (keeps rap out of "melody-driven pop" queries).
      3. If the user ``likes_acoustic`` is True, drop tracks with
         ``acousticness < 0.3``.
    """
    try:
        instr = float(meta.get("instrumentalness", 0.0))
        speech = float(meta.get("speechiness", 0.0))
        acoustic = float(meta.get("acousticness", 0.0))
    except (TypeError, ValueError):
        return True  # Be permissive when metadata is malformed.

    if float(dna.get("instrumentalness", 0.0)) >= 0.7 and instr < 0.4:
        return False
    if float(dna.get("speechiness", 0.0)) <= 0.1 and speech > 0.25:
        return False
    if bool(dna.get("likes_acoustic", False)) and acoustic < 0.3:
        return False
    return True


def ingestor_node(state: EchoState) -> EchoState:
    """Retrieve candidate tracks from ChromaDB using the DNA-profile vector."""
    dna = _merged_dna(state.get("dna_profile"))
    top_k = int(dna.get("top_k", 5)) or 5

    try:
        collection = get_collection()
    except Exception as exc:  # pragma: no cover - surfaced through state
        return {"retrieved_tracks": [], "error": f"ingestor: {exc}"}

    query_vec = build_query_vector(dna)
    if len(query_vec) != EMBEDDING_DIM:  # pragma: no cover - defensive
        return {
            "retrieved_tracks": [],
            "error": f"ingestor: query dim {len(query_vec)} != {EMBEDDING_DIM}",
        }

    # Over-fetch so post-filters still leave us with enough candidates.
    try:
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=max(top_k * 3, 10),
        )
    except Exception as exc:  # pragma: no cover
        return {"retrieved_tracks": [], "error": f"ingestor: {exc}"}

    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    retrieved: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metadatas):
        if meta is None:
            continue
        if not _passes_feature_filters(meta, dna):
            continue
        track = dict(meta)
        track["distance"] = float(distances[idx]) if idx < len(distances) else None
        track.setdefault("id", ids[idx] if idx < len(ids) else None)
        retrieved.append(track)
        if len(retrieved) >= top_k:
            break

    # Fallback: if post-filters removed everything, return the raw top-k so the
    # pipeline never produces an empty recommendation list in normal conditions.
    if not retrieved and metadatas:
        for idx, meta in enumerate(metadatas[:top_k]):
            if meta is None:
                continue
            track = dict(meta)
            track["distance"] = (
                float(distances[idx]) if idx < len(distances) else None
            )
            track.setdefault("id", ids[idx] if idx < len(ids) else None)
            retrieved.append(track)

    return {"retrieved_tracks": retrieved}


# ── Researcher Node ─────────────────────────────────────────────────────────

# Mock trivia keyed by the artists that appear in data/songs.json. Per the
# design: "Mock a simple dictionary of artist trivia for this step."
ARTIST_TRIVIA: Dict[str, str] = {
    "Neon Echo": "Cut their debut EP on a hacked Game Boy in 2019.",
    "LoRoom": "Produces every track from a converted 1970s camper van.",
    "Voltline": "Named after the Tesla coil that tripped their first power bill.",
    "Paper Lanterns": "Each release is timed to a local lunar festival.",
    "Max Pulse": "Former pro weightlifter; tempo follows heart-rate zones.",
    "Orbit Bloom": "Writes with field recordings from high-altitude balloons.",
    "Slow Stereo": "Records upright bass through a vintage ribbon mic only.",
    "Indigo Parade": "Got their break busking on the Lisbon tram line 28.",
    "Flow State": "Freestyles the first verse of every track in one take.",
    "Clara Voss": "Trained on a 1911 Bosendorfer she restored herself.",
    "Sable & June": "The duo met at a late-night radio request line.",
    "The Pines": "Tours in a converted logging truck named 'Hazel'.",
    "Pulse Theory": "All drops are designed around 140 BPM club ergonomics.",
    "Ember Frost": "Writes lyrics entirely on postcards before recording.",
    "Nocturn": "Their low-end rig weighs more than the lighting truss.",
    "Coral Drift": "Records vocals on a beach during incoming tides.",
    "Joe Arroyo": "A Colombian salsa icon whose anthems still anchor festivals.",
    "Stevie Wonder": "Has won 25 Grammys and helped define funk-era pop.",
    "Nirvana": "Defined Seattle grunge with a raw dynamic-shift formula.",
    "Tyla": "Broke out globally with Amapiano-infused Afro-pop.",
    "Stan Getz & Joao Gilberto": "Their bossa nova album sold over 2M copies.",
    "Bee Gees": "Wrote the Saturday Night Fever soundtrack in a single weekend.",
    "Drake": "Holds the record for most Billboard Hot 100 entries.",
    "Johnny Cash": "Recorded his iconic Folsom Prison show in 1968.",
    "BLACKPINK": "First K-pop girl group to headline Coachella.",
    "The Specials": "Pioneers of the UK 2-tone ska revival in 1979.",
    "Santana": "Carlos Santana has fused Latin rhythms with rock since 1966.",
    "New Order": "Formed from the ashes of Joy Division in 1980.",
}


def researcher_node(state: EchoState) -> EchoState:
    """Attach mock trivia for every artist in the retrieved tracks."""
    retrieved = state.get("retrieved_tracks") or []
    trivia: Dict[str, str] = {}
    for track in retrieved:
        artist = track.get("artist") if isinstance(track, dict) else None
        if not artist:
            continue
        trivia[artist] = ARTIST_TRIVIA.get(
            artist, f"No trivia on file for {artist} yet."
        )
    return {"artist_trivia": trivia}


# ── Reasoning Node ──────────────────────────────────────────────────────────

REASONING_SYSTEM_PROMPT = (
    "You are EchoSphere, a highly analytical DJ. You explain WHY a specific "
    "track fits the listener's request, citing concrete audio-feature "
    "similarities (energy, tempo, acousticness, instrumentalness, "
    "speechiness, valence, danceability) and weaving in artist trivia when "
    "useful. Keep each explanation to 2-3 sentences, conversational but "
    "precise. Do not invent tracks or artists."
)

REASONING_USER_TEMPLATE = """\
Listener request: "{user_request}"

Taste DNA profile:
{dna_summary}

Track to explain:
- Title: {title}
- Artist: {artist}
- Genre / mood: {genre} / {mood}
- Audio features: energy={energy}, tempo={tempo}, valence={valence}, \
danceability={danceability}, acousticness={acousticness}, \
instrumentalness={instrumentalness}, speechiness={speechiness}
- Distance from listener vector: {distance}

Artist trivia (optional talking point): {trivia}

Write a 2-3 sentence explanation of why this track matches the listener.
"""


def _format_dna_summary(dna: Dict[str, Any]) -> str:
    parts = []
    for key in (
        "genre",
        "mood",
        "energy",
        "tempo_bpm",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "likes_acoustic",
    ):
        if key in dna and dna[key] is not None:
            parts.append(f"  - {key}: {dna[key]}")
    return "\n".join(parts) if parts else "  (no targets specified)"


def _build_llm():
    """Instantiate ChatOllama. Lazy so tests can patch before construction."""
    from langchain_ollama import ChatOllama  # local import — optional dep

    load_dotenv()
    model = os.getenv("ECHO_OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("ECHO_OLLAMA_TEMPERATURE", "0.4"))
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


class _FallbackMessage:
    """Tiny ``.content``-carrying stand-in for LangChain message classes.

    Used only when ``langchain_core`` is not importable (e.g. lightweight test
    environments). Real ChatOllama production use still goes through
    ``langchain_core`` message types when available.
    """

    def __init__(self, role: str, content: str):
        self.type = role
        self.content = content


def _message_factories():
    """Return ``(SystemMessage, HumanMessage)`` preferring langchain_core."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        return SystemMessage, HumanMessage
    except Exception:
        return (
            lambda content: _FallbackMessage("system", content),
            lambda content: _FallbackMessage("human", content),
        )


def _invoke_llm(llm: Any, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return its string content."""
    SystemMessage, HumanMessage = _message_factories()
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # LangChain occasionally returns content as a list of chunks.
        content = "".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in content
        )
    return str(content).strip()


def reasoning_node(state: EchoState, llm: Optional[Any] = None) -> EchoState:
    """Produce DJ-style explanations for every retrieved track."""
    retrieved = state.get("retrieved_tracks") or []
    if not retrieved:
        return {"explanations": []}

    dna = _merged_dna(state.get("dna_profile"))
    trivia = state.get("artist_trivia") or {}
    user_request = state.get("user_request", "")

    if llm is None:
        try:
            llm = _build_llm()
        except Exception as exc:
            return {
                "explanations": [],
                "error": f"reasoning: failed to init ChatOllama: {exc}",
            }

    dna_summary = _format_dna_summary(dna)
    explanations: List[str] = []
    for track in retrieved:
        user_prompt = REASONING_USER_TEMPLATE.format(
            user_request=user_request or "(no free-text request)",
            dna_summary=dna_summary,
            title=track.get("title", "Unknown title"),
            artist=track.get("artist", "Unknown artist"),
            genre=track.get("genre", "?"),
            mood=track.get("mood", "?"),
            energy=track.get("energy", "?"),
            tempo=track.get("tempo_bpm", "?"),
            valence=track.get("valence", "?"),
            danceability=track.get("danceability", "?"),
            acousticness=track.get("acousticness", "?"),
            instrumentalness=track.get("instrumentalness", "?"),
            speechiness=track.get("speechiness", "?"),
            distance=track.get("distance", "?"),
            trivia=trivia.get(track.get("artist", ""), "n/a"),
        )
        try:
            explanation = _invoke_llm(llm, REASONING_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            explanation = (
                f"[LLM unavailable — falling back to feature summary] "
                f"{track.get('title','?')} by {track.get('artist','?')} matches "
                f"on energy={track.get('energy','?')} and genre={track.get('genre','?')}. "
                f"({exc})"
            )
        explanations.append(explanation)

    return {"explanations": explanations}


__all__ = [
    "ARTIST_TRIVIA",
    "REASONING_SYSTEM_PROMPT",
    "REASONING_USER_TEMPLATE",
    "ingestor_node",
    "researcher_node",
    "reasoning_node",
]
