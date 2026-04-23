"""
MCP (Model Context Protocol) server for GrooveGenius.

Exposes the recommender as 5 tools + 4 resources so external AI agents
(Claude Code, Claude Desktop, Cursor, custom agents) can call it.

Entry point: python -m src.mcp_server
"""

import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource

from .recommender import (
    load_songs, recommend_songs, score_song,
    DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED, RankingStrategy,
)
from .rag import load_knowledge
from .confidence import ConfidenceScorer
from .bias_auditor import BiasAuditor


STRATEGIES = {
    "default": DEFAULT,
    "genre_first": GENRE_FIRST,
    "mood_first": MOOD_FIRST,
    "energy_focused": ENERGY_FOCUSED,
}


# ── Handler functions (sync, testable without MCP transport) ──────────────

def _load_state():
    """Load songs and knowledge once. Cached at module level."""
    global _songs, _knowledge, _confidence_scorer
    if "_songs" not in globals() or _songs is None:
        _songs = load_songs("data/songs.csv")
        try:
            _knowledge = load_knowledge()
        except Exception:
            _knowledge = None
        _confidence_scorer = ConfidenceScorer(_songs)
    return _songs, _knowledge, _confidence_scorer


_songs = None
_knowledge = None
_confidence_scorer = None


def handle_recommend_manual(arguments: Dict) -> Dict:
    """Handle the recommend_manual tool call."""
    songs, knowledge, scorer = _load_state()

    profile = {
        "genre": arguments["genre"],
        "mood": arguments["mood"],
        "energy": arguments["energy"],
        "likes_acoustic": arguments["likes_acoustic"],
    }
    strategy = STRATEGIES.get(arguments.get("strategy", "default"), DEFAULT)
    k = arguments.get("k", 5)

    results = recommend_songs(profile, songs, k=k, strategy=strategy, knowledge=knowledge)
    confidence = scorer.compute(profile, results, knowledge)

    return {
        "recommendations": [
            {
                "title": s["title"],
                "artist": s["artist"],
                "genre": s["genre"],
                "mood": s["mood"],
                "energy": s["energy"],
                "score": round(sc, 4),
                "reasons": ex,
            }
            for s, sc, ex in results
        ],
        "confidence": {
            "score": confidence.overall_confidence,
            "label": confidence.confidence_label,
            "warnings": confidence.warnings,
        },
    }


def handle_explain_song(arguments: Dict) -> Dict:
    """Handle the explain_song tool call."""
    songs, knowledge, _ = _load_state()

    title = arguments["song_title"]
    song_dict = next(
        (s for s in songs if s["title"].lower() == title.lower()),
        None,
    )
    if not song_dict:
        return {"error": f"Song not found: {title}"}

    profile = {
        "genre": arguments["genre"],
        "mood": arguments["mood"],
        "energy": arguments["energy"],
        "likes_acoustic": arguments["likes_acoustic"],
    }
    score, reasons = score_song(profile, song_dict, knowledge=knowledge)

    return {
        "song": song_dict["title"],
        "artist": song_dict["artist"],
        "score": round(score, 4),
        "reasons": reasons,
    }


def handle_list_catalog(arguments: Dict) -> List[Dict]:
    """Handle the list_catalog tool call."""
    songs, _, _ = _load_state()

    filtered = songs
    if arguments.get("genre_filter"):
        filtered = [s for s in filtered if s["genre"] == arguments["genre_filter"]]
    if arguments.get("mood_filter"):
        filtered = [s for s in filtered if s["mood"] == arguments["mood_filter"]]

    return [
        {
            "title": s["title"],
            "artist": s["artist"],
            "genre": s["genre"],
            "mood": s["mood"],
            "energy": s["energy"],
            "popularity": s["popularity"],
            "release_year": s["release_year"],
        }
        for s in filtered
    ]


def handle_audit_bias(arguments: Dict) -> Dict:
    """Handle the audit_bias tool call."""
    songs, knowledge, _ = _load_state()

    strategy = STRATEGIES.get(arguments.get("strategy", "default"), DEFAULT)
    auditor = BiasAuditor(songs, strategy=strategy, knowledge=knowledge)
    report = auditor.run_audit()

    return {
        "strategy": report.strategy_name,
        "profiles_tested": report.profiles_tested,
        "biases": [
            {
                "name": b.name,
                "severity": b.severity,
                "affected": f"{b.affected_count}/{b.total_count}",
                "description": b.description,
                "suggestion": b.suggestion,
            }
            for b in report.biases
        ],
        "catalog_coverage": report.catalog_stats.get("catalog_coverage", 0),
    }


def handle_recommend(arguments: Dict) -> Dict:
    """Handle the recommend tool call (requires LLM)."""
    songs, knowledge, _ = _load_state()

    try:
        from .llm_provider import get_provider
        from .agent import AgentLoop

        llm = get_provider("ollama")
        agent = AgentLoop(llm=llm, songs=songs, knowledge=knowledge)
        result = agent.run(
            query=arguments["query"],
            k=arguments.get("k", 5),
            strategy=STRATEGIES.get(arguments.get("strategy", "default"), DEFAULT),
        )
        return result
    except ConnectionError:
        return {
            "error": "Ollama is not running. Use recommend_manual instead for offline recommendations.",
            "suggestion": "Install Ollama (https://ollama.com) and run: ollama pull llama3.1:8b",
        }


# ── EchoSphere-RAG handlers ───────────────────────────────────────────────

def _build_dna_from_args(arguments: Dict) -> Dict:
    """Shape an MCP arguments dict into an EchoSphere DNA profile."""
    from .echosphere.state import DEFAULT_DNA_PROFILE

    dna = dict(DEFAULT_DNA_PROFILE)
    supplied = arguments.get("dna_profile") or {}
    if isinstance(supplied, dict):
        dna.update({k: v for k, v in supplied.items() if v is not None})
    # Allow flat overrides so simple MCP clients don't have to nest.
    for key in (
        "genre", "mood", "energy", "tempo_bpm", "valence", "danceability",
        "acousticness", "instrumentalness", "speechiness", "likes_acoustic",
        "top_k",
    ):
        if key in arguments and arguments[key] is not None:
            dna[key] = arguments[key]
    if "k" in arguments:
        dna["top_k"] = arguments["k"]
    return dna


def handle_echosphere_recommend(arguments: Dict) -> Dict:
    """Run the EchoSphere-RAG LangGraph pipeline end to end."""
    from .echosphere import run_echosphere

    query = arguments.get("query", "")
    dna = _build_dna_from_args(arguments)
    try:
        state = run_echosphere(query, dna)
    except Exception as exc:
        return {
            "error": f"EchoSphere pipeline failed: {exc}",
            "suggestion": (
                "Ensure Ollama is running and ChromaDB is seeded via "
                "`python -m src.echosphere.vector_store`."
            ),
        }

    retrieved = state.get("retrieved_tracks") or []
    explanations = state.get("explanations") or []
    return {
        "query": query,
        "dna_profile": dna,
        "recommendations": [
            {
                "title": t.get("title"),
                "artist": t.get("artist"),
                "genre": t.get("genre"),
                "mood": t.get("mood"),
                "distance": t.get("distance"),
                "explanation": explanations[idx] if idx < len(explanations) else "",
            }
            for idx, t in enumerate(retrieved)
        ],
        "artist_trivia": state.get("artist_trivia") or {},
        "error": state.get("error"),
    }


def handle_echosphere_ingest(arguments: Dict) -> Dict:
    """Seed (or rebuild) the EchoSphere ChromaDB collection."""
    from .echosphere.vector_store import ingest_catalog, DEFAULT_CSV_PATH

    csv_path = arguments.get("csv_path", DEFAULT_CSV_PATH)
    try:
        summary = ingest_catalog(csv_path=csv_path)
    except Exception as exc:
        return {"error": f"Ingestion failed: {exc}"}
    return {"status": "ok", **summary}


def handle_echosphere_explain(arguments: Dict) -> Dict:
    """Run just the Reasoning node for a single track id + DNA profile."""
    from .echosphere.nodes import reasoning_node, researcher_node

    songs, _, _ = _load_state()
    track_id = str(arguments.get("track_id", ""))
    track = next((dict(s) for s in songs if str(s.get("id")) == track_id), None)
    if not track:
        return {"error": f"Track id not found: {track_id}"}
    track.setdefault("distance", 0.0)

    dna = _build_dna_from_args(arguments)
    state: Dict = {
        "user_request": arguments.get("query", ""),
        "dna_profile": dna,
        "retrieved_tracks": [track],
    }
    state.update(researcher_node(state) or {})
    state.update(reasoning_node(state) or {})
    return {
        "track": {
            "id": track.get("id"),
            "title": track.get("title"),
            "artist": track.get("artist"),
        },
        "explanation": (state.get("explanations") or [""])[0],
        "trivia": (state.get("artist_trivia") or {}).get(track.get("artist", ""), ""),
        "error": state.get("error"),
    }


def handle_resource_chroma_stats() -> str:
    """Return EchoSphere ChromaDB collection stats."""
    try:
        from .echosphere.vector_store import (
            COLLECTION_NAME,
            EMBEDDING_DIM,
            get_collection,
        )

        collection = get_collection()
        return json.dumps({
            "collection": COLLECTION_NAME,
            "count": collection.count(),
            "embedding_dim": EMBEDDING_DIM,
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "error": f"ChromaDB unavailable: {exc}",
            "suggestion": (
                "Run `python -m src.echosphere.vector_store` to seed the DB."
            ),
        })


# ── Resource handlers ─────────────────────────────────────────────────────

def handle_resource_catalog_songs() -> str:
    """Return full catalog as JSON."""
    songs, _, _ = _load_state()
    return json.dumps(handle_list_catalog({}), indent=2)


def handle_resource_catalog_stats() -> str:
    """Return catalog statistics."""
    songs, _, _ = _load_state()
    genre_counts = Counter(s["genre"] for s in songs)
    mood_counts = Counter(s["mood"] for s in songs)
    energies = [s["energy"] for s in songs]
    return json.dumps({
        "total_songs": len(songs),
        "unique_genres": len(genre_counts),
        "unique_moods": len(mood_counts),
        "genre_counts": dict(genre_counts),
        "mood_counts": dict(mood_counts),
        "energy_range": {"min": min(energies), "max": max(energies),
                         "mean": round(sum(energies) / len(energies), 3)},
    }, indent=2)


def handle_resource_strategies() -> str:
    """Return strategy definitions."""
    return json.dumps({
        name: {
            "genre_weight": s.genre_weight,
            "mood_weight": s.mood_weight,
            "energy_weight": s.energy_weight,
            "acoustic_weight": s.acoustic_weight,
        }
        for name, s in STRATEGIES.items()
    }, indent=2)


def handle_resource_audit_latest() -> str:
    """Return the most recent audit report from reports/ directory."""
    reports_dir = "reports"
    if not os.path.isdir(reports_dir):
        return json.dumps({"error": "No reports directory. Run --audit first."})

    files = sorted(
        [f for f in os.listdir(reports_dir) if f.startswith("bias_audit_") and f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return json.dumps({"error": "No audit reports found. Run: python -m src.main --audit"})

    with open(os.path.join(reports_dir, files[0]), encoding="utf-8") as f:
        return f.read()


# ── MCP Server wiring ────────────────────────────────────────────────────

app = Server("groovegenius")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="recommend",
            description=(
                "Get personalized music recommendations from a 30-song catalog using "
                "natural language. Uses an agentic loop (requires Ollama LLM). "
                "For offline use, prefer recommend_manual."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language music request"},
                    "k": {"type": "integer", "description": "Number of results (default: 5)", "default": 5},
                    "strategy": {
                        "type": "string",
                        "enum": ["default", "genre_first", "mood_first", "energy_focused"],
                        "default": "default",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="recommend_manual",
            description=(
                "Get music recommendations using explicit preferences. "
                "Works offline — no LLM required."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "Preferred genre (e.g., pop, lofi, rock)"},
                    "mood": {"type": "string", "description": "Preferred mood (e.g., happy, chill, angry)"},
                    "energy": {"type": "number", "description": "Target energy 0.0-1.0"},
                    "likes_acoustic": {"type": "boolean", "description": "Prefer acoustic sounds?"},
                    "k": {"type": "integer", "default": 5},
                    "strategy": {
                        "type": "string",
                        "enum": ["default", "genre_first", "mood_first", "energy_focused"],
                        "default": "default",
                    },
                },
                "required": ["genre", "mood", "energy", "likes_acoustic"],
            },
        ),
        Tool(
            name="explain_song",
            description=(
                "Get a detailed scoring breakdown for a specific song "
                "against a user preference profile."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "song_title": {"type": "string", "description": "Exact song title"},
                    "genre": {"type": "string"},
                    "mood": {"type": "string"},
                    "energy": {"type": "number"},
                    "likes_acoustic": {"type": "boolean"},
                },
                "required": ["song_title", "genre", "mood", "energy", "likes_acoustic"],
            },
        ),
        Tool(
            name="audit_bias",
            description=(
                "Run automated bias detection across synthetic user profiles. "
                "Returns detected biases with severity ratings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["default", "genre_first", "mood_first", "energy_focused"],
                        "default": "default",
                    },
                },
            },
        ),
        Tool(
            name="list_catalog",
            description="List songs in the catalog, optionally filtered by genre or mood.",
            inputSchema={
                "type": "object",
                "properties": {
                    "genre_filter": {"type": "string", "description": "Filter by genre"},
                    "mood_filter": {"type": "string", "description": "Filter by mood"},
                },
            },
        ),
        Tool(
            name="echosphere_recommend",
            description=(
                "Run the EchoSphere-RAG agentic pipeline: ChromaDB vector search "
                "(Ingestor) -> artist trivia (Researcher) -> DJ-style explanations "
                "(Reasoning, via local Ollama)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text listener request"},
                    "dna_profile": {
                        "type": "object",
                        "description": "Audio-feature taste DNA profile (partial overrides allowed)",
                    },
                    "k": {"type": "integer", "default": 5, "description": "Number of tracks"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="echosphere_ingest",
            description="(Re)seed the EchoSphere ChromaDB collection from data/songs.csv.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "Path to the catalog CSV (defaults to data/songs.csv)",
                    },
                },
            },
        ),
        Tool(
            name="echosphere_explain",
            description=(
                "Run only the Reasoning node on a single track id, given a DNA "
                "profile. Useful for per-track 'why was this recommended?' UX."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "string", "description": "Catalog id of the track"},
                    "query": {"type": "string", "description": "Optional listener request context"},
                    "dna_profile": {
                        "type": "object",
                        "description": "Audio-feature taste DNA profile",
                    },
                },
                "required": ["track_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handlers = {
        "recommend": handle_recommend,
        "recommend_manual": handle_recommend_manual,
        "explain_song": handle_explain_song,
        "audit_bias": handle_audit_bias,
        "list_catalog": handle_list_catalog,
        "echosphere_recommend": handle_echosphere_recommend,
        "echosphere_ingest": handle_echosphere_ingest,
        "echosphere_explain": handle_echosphere_explain,
    }

    handler = handlers.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    result = handler(arguments)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri="groovegenius://catalog/songs", name="Song Catalog",
                 description="Full 30-song catalog as JSON"),
        Resource(uri="groovegenius://catalog/stats", name="Catalog Statistics",
                 description="Genre counts, mood counts, energy distribution"),
        Resource(uri="groovegenius://strategies", name="Ranking Strategies",
                 description="Available strategies with weight configurations"),
        Resource(uri="groovegenius://audit/latest", name="Latest Audit Report",
                 description="Most recent bias audit report"),
        Resource(uri="groovegenius://chroma/stats", name="EchoSphere ChromaDB Stats",
                 description="Row count and embedding dim for the agentic vector store"),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    resource_handlers = {
        "groovegenius://catalog/songs": handle_resource_catalog_songs,
        "groovegenius://catalog/stats": handle_resource_catalog_stats,
        "groovegenius://strategies": handle_resource_strategies,
        "groovegenius://audit/latest": handle_resource_audit_latest,
        "groovegenius://chroma/stats": handle_resource_chroma_stats,
    }

    handler = resource_handlers.get(str(uri))
    if not handler:
        return json.dumps({"error": f"Unknown resource: {uri}"})

    return handler()


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
