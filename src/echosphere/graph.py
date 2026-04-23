"""
LangGraph wiring for the EchoSphere-RAG pipeline.

Edges strictly follow the design doc:

    User Input -> Ingestor Node -> Researcher Node -> Reasoning Node -> Output
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .nodes import ingestor_node, reasoning_node, researcher_node
from .state import DEFAULT_DNA_PROFILE, EchoState


def build_graph():
    """Compile the 3-node LangGraph and return the runnable graph.

    The import is local so that environments without ``langgraph`` installed
    can still import ``src.echosphere.nodes`` / ``state`` for unit tests that
    exercise individual nodes.
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(EchoState)
    graph.add_node("ingestor", ingestor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reasoning", reasoning_node)

    graph.add_edge(START, "ingestor")
    graph.add_edge("ingestor", "researcher")
    graph.add_edge("researcher", "reasoning")
    graph.add_edge("reasoning", END)

    return graph.compile()


def run_echosphere(
    user_request: str,
    dna_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke the compiled graph on a single query and return the final state."""
    app = build_graph()
    initial_state: EchoState = {
        "user_request": user_request,
        "dna_profile": dna_profile or dict(DEFAULT_DNA_PROFILE),
    }
    return app.invoke(initial_state)


def _main() -> None:
    """``python -m src.echosphere.graph`` — runs a sample query end to end."""
    sample_query = "I need a high-energy pop track"
    sample_dna = {
        **DEFAULT_DNA_PROFILE,
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "tempo_bpm": 125.0,
        "valence": 0.8,
        "danceability": 0.8,
        "acousticness": 0.15,
        "instrumentalness": 0.05,
        "speechiness": 0.1,
        "likes_acoustic": False,
        "top_k": 3,
    }
    final_state = run_echosphere(sample_query, sample_dna)
    print(json.dumps(final_state, indent=2, default=str))


if __name__ == "__main__":
    _main()


__all__ = ["build_graph", "run_echosphere"]
