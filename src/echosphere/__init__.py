"""
Project EchoSphere-RAG — an agentic music recommendation pipeline.

A LangGraph-orchestrated 3-node pipeline (Ingestor -> Researcher -> Reasoning)
over a local ChromaDB vector store and a local Ollama model.

Entry points:
- ``build_graph()`` — compiled LangGraph StateGraph.
- ``run_echosphere(user_request, dna_profile)`` — convenience wrapper.
- ``EchoState`` — the shared TypedDict that flows through the graph.
"""

from .state import EchoState, DEFAULT_DNA_PROFILE
from .graph import build_graph, run_echosphere

__all__ = [
    "EchoState",
    "DEFAULT_DNA_PROFILE",
    "build_graph",
    "run_echosphere",
]
