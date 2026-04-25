# System workflows and data flow

Canonical architecture reference for GrooveGenius 2.0. Diagrams match the code in `src/`, `frontend/`, and `src/echosphere/`.

## Major components and surfaces

```mermaid
flowchart TB
  subgraph users [Users_and_agents]
    Human[Human_browser_or_terminal]
    ExtAgent[External_AI_agents]
  end

  subgraph entry [Entry_points]
    Streamlit[Streamlit_frontend_app]
    CLI[src_main_py]
    MCP[src_mcp_server_py]
  end

  subgraph llm [LLM_abstraction]
    Provider[src_llm_provider_py]
  end

  subgraph fastPipe [Fast_pipeline]
    Rec[src_recommender_py]
    RAG[src_rag_py]
    Know[data_knowledge_json]
  end

  subgraph agentic [Agentic_EchoSphere]
    Graph[src_echosphere_graph_py]
    Nodes[src_echosphere_nodes_py]
    Chroma[ChromaDB_data_chroma]
  end

  subgraph reliability [Reliability_layer]
    Conf[src_confidence_py]
    Guard[src_guardrails_py]
    Audit[src_bias_auditor_py]
    Crit[src_self_critique_py]
  end

  Human --> Streamlit
  Human --> CLI
  ExtAgent --> MCP

  Streamlit --> Rec
  Streamlit --> Graph
  CLI --> Rec
  CLI --> Graph
  MCP --> Rec
  MCP --> Graph

  Rec --> RAG
  RAG --> Know
  Graph --> Nodes
  Nodes --> Chroma
  Nodes --> Provider

  CLI --> Provider
  MCP --> Provider
  Streamlit --> Provider

  Rec --> Conf
  Graph --> Conf
  Conf --> Guard
  Rec --> Crit
  CLI --> Audit
  Streamlit --> Audit
  MCP --> Audit
```

**Data flow (high level):** preferences or natural language → scoring or LangGraph state → ranked tracks + explanations → optional confidence, guardrails, self-critique, or bias audit JSON.

## Fast mode pipeline

Deterministic path: no LLM required for core ranking (LLM optional in interactive mode only).

| Step | Input | Processing | Output |
|------|--------|-------------|--------|
| 1 | `data/songs.json` | `load_songs()` in `src/recommender.py` | In-memory catalog |
| 2 | User prefs dict + optional knowledge | `score_song()` / `recommend_songs()`; optional `src/rag.py` for genre/mood similarity | Per-song score + reason strings |
| 3 | Scored list | Sort by score, diversity penalties in `recommend_songs` | Top-k tuples `(song, score, explanation)` |
| 4 | Results | `ConfidenceScorer` in `src/confidence.py`, `apply_guardrails` in `src/guardrails.py` | Confidence label + user-facing warnings |

ASCII summary:

```
songs.json + user_prefs [+ knowledge]
  -> load_songs
  -> recommend_songs (score_song per row, RAG optional)
  -> sorted top-k + explanations
  -> ConfidenceScorer + guardrails (+ self_critique when enabled)
```

## EchoSphere-RAG (LangGraph)

Compiled graph in `src/echosphere/graph.py`: `START → ingestor → researcher → reasoning → END`.

```mermaid
flowchart LR
  START([START])
  Ingest[ingestor_node]
  Research[researcher_node]
  Reason[reasoning_node]
  ENDN([END])

  START --> Ingest
  Ingest --> Research
  Research --> Reason
  Reason --> ENDN
```

| Node | Implementation | Role |
|------|----------------|------|
| `ingestor` | `ingestor_node` in `src/echosphere/nodes.py` | Builds 7-D query vector from DNA profile, queries ChromaDB, applies instrumentalness / speechiness / acousticness post-filters |
| `researcher` | `researcher_node` in `src/echosphere/nodes.py` | Attaches mock `ARTIST_TRIVIA` strings to retrieved artists |
| `reasoning` | `reasoning_node` in `src/echosphere/nodes.py` | LLM (`ChatOllama` or injected provider) produces DJ-style per-track explanations; state includes `retrieved_tracks`, `explanations`, `artist_trivia` |

**Observability:** final `EchoState` is JSON-printed by `python -m src.echosphere.graph` and surfaced in CLI batch agentic mode (`run_profile_agentic` in `src/main.py`) and Streamlit when agentic mode is selected.

## Reliability stack

| Component | Role | Primary module |
|-----------|------|----------------|
| Confidence scoring | Summarizes match strength and catalog coverage | `src/confidence.py` |
| Guardrails | Adds warnings / badges for weak matches; does not hard-block | `src/guardrails.py` |
| Self-critique | Extra narrative when confidence is very low (fast path) | `src/self_critique.py` |
| Bias auditor | Synthetic profiles, detectors, JSON reports | `src/bias_auditor.py` |
| Automated tests | Regression and behavior checks | `tests/test_*.py` |
