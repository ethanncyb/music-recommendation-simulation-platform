"""
LangGraph wiring for the EchoSphere-RAG pipeline.

Edges strictly follow the design doc:

    User Input -> Ingestor Node -> Researcher Node -> Reasoning Node -> Output
"""

from __future__ import annotations

import functools
import json
from typing import Any, Dict, Optional

from .nodes import ingestor_node, reasoning_node, researcher_node, _build_llm
from .state import DEFAULT_DNA_PROFILE, EchoState


def build_graph(llm=None):
    """Compile the 3-node LangGraph and return the runnable graph.

    Args:
        llm: Optional LLM instance for the reasoning node.  Accepts a
             LangChain-compatible object **or** one of our ``LLMProvider``
             subclasses (``GeminiProvider``, ``AnthropicProvider``, etc.).
             When *None* the reasoning node falls back to local ``ChatOllama``.

    The import is local so that environments without ``langgraph`` installed
    can still import ``src.echosphere.nodes`` / ``state`` for unit tests that
    exercise individual nodes.
    """
    from langgraph.graph import END, START, StateGraph

    # Wrap our LLMProvider if needed, once, so _build_llm isn't called per track.
    resolved_llm = _build_llm(llm) if llm is not None else None

    # Bind the resolved LLM to the reasoning node so LangGraph can call it
    # with the standard single-arg (state) signature.
    def _reasoning_with_llm(state):
        return reasoning_node(state, llm=resolved_llm)

    graph = StateGraph(EchoState)
    graph.add_node("ingestor", ingestor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reasoning", _reasoning_with_llm)

    graph.add_edge(START, "ingestor")
    graph.add_edge("ingestor", "researcher")
    graph.add_edge("researcher", "reasoning")
    graph.add_edge("reasoning", END)

    return graph.compile()


def run_echosphere(
    user_request: str,
    dna_profile: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Dict[str, Any]:
    """Invoke the compiled graph on a single query and return the final state.

    Args:
        llm: Optional LLM for the reasoning node (``LLMProvider`` or
             LangChain-compatible). Defaults to local ChatOllama.
    """
    app = build_graph(llm=llm)
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
