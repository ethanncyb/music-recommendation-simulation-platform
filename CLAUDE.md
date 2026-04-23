# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Mission

This project simulates how big-name music platforms like Spotify and TikTok predict what users will love next. You are building for a startup music platform that wants to understand the mechanics behind personalized recommendations — transforming raw song data and user "taste profiles" into ranked suggestions, with transparent explanations for every choice.

The goal is not just to make recommendations, but to make them *explainable*: every score is decomposed into signals (genre match, mood fit, energy proximity, etc.) so the system can show *why* a song was suggested, mirroring how production recommenders surface reasoning to users and product teams.

## Commands

```bash
# Run the application (fast mode — 4 profiles + strategy comparison)
python -m src.main

# Run the same 4 profiles through the new EchoSphere-RAG agentic pipeline
python -m src.main --batch --mode agentic

# Run bias audit (fast pipeline)
python -m src.main --audit

# Run interactive agent (requires Ollama with llama3.1:8b)
python -m src.main --interactive
python -m src.main --interactive --provider anthropic  # or use Claude API

# Launch Streamlit web UI (sidebar toggle selects fast vs agentic)
streamlit run frontend/app.py
python -m streamlit run frontend/app.py   # fallback if shebang is broken

# Start MCP server (exposes both fast and echosphere_* tools)
python -m src.mcp_server

# Regenerate fast-mode knowledge JSON graphs (requires Ollama)
python -m src.generate_knowledge

# EchoSphere-RAG: seed the ChromaDB and run the sample graph
python -m src.echosphere.vector_store
python -m src.echosphere.graph

# Run tests (98 tests)
pytest

# Run a single test
pytest tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score
```

Activate the virtual environment first if needed: `source .venv/bin/activate`

No linter is configured; the project uses manual code review.

## Architecture

This is a **content-based music recommender** that scores a 30-song catalog against a user's stated preferences and returns the top-k ranked songs with explanations. The repository contains **two pipelines** selectable via `--mode fast|agentic` (CLI), a sidebar toggle (Streamlit), or the `echosphere_*` MCP tools:

- **Fast mode** — `src/recommender.py` + optional `src/rag.py` (JSON knowledge graphs). Deterministic weighted scoring, no LLM, no vector DB. This is the original GrooveGenius 2.0 stack.
- **Agentic mode (EchoSphere-RAG)** — `src/echosphere/` — a LangGraph `StateGraph` (Ingestor → Researcher → Reasoning) over a persistent ChromaDB (`data/chroma/`, git-ignored) and `langchain_ollama.ChatOllama`. The Ingestor embeds each track with a 7-dim audio-feature vector (`energy, tempo_norm, valence, danceability, acousticness, instrumentalness, speechiness`) and post-filters results on instrumentalness / speechiness / acousticness thresholds. See [new_design.md](new_design.md) for the original spec and [src/echosphere/__init__.py](src/echosphere/__init__.py) for the module layout.

Shared peripherals (`ConfidenceScorer`, `apply_guardrails`, `BiasAuditor`, Streamlit UI, MCP server) accept either pipeline's output: `ConfidenceScorer.compute` auto-detects `EchoState` dicts, and `BiasAuditor.run_audit(pipeline="agentic")` runs every synthetic profile through EchoSphere-RAG.

### Data Flow

```
UserProfile (genre, mood, energy, likes_acoustic, optional: min_popularity, preferred_decade, preferred_tags)
  → load_songs() from data/songs.csv
  → score_song() for each song  [base signals + advanced bonuses → (score, reasons)]
  → recommend_songs()           [greedy diversity re-ranking]
  → Ranked table output
```

### Key Files

**Core (v1.0):**
- **`src/recommender.py`** — Core scoring: `Song`, `UserProfile`, `RankingStrategy` dataclasses; `load_songs()`, `score_song()`, `recommend_songs()` functions; `Recommender` OOP class. Now accepts optional `knowledge` param for RAG.
- **`src/main.py`** — CLI entry point with `--audit`, `--interactive`, `--provider`, `--model` flags.
- **`data/songs.csv`** — 30 songs x 17 attributes.

**RAG + Knowledge (Phase 1):**
- **`src/rag.py`** — `KnowledgeBase` class, `load_knowledge()` — similarity lookups for genre/mood.
- **`src/llm_provider.py`** — `OllamaProvider`, `AnthropicProvider`, `get_provider()` factory.
- **`src/generate_knowledge.py`** — CLI script to regenerate knowledge graphs via LLM.
- **`data/knowledge/genre_graph.json`** / **`mood_graph.json`** — Pre-shipped similarity matrices.

**Bias Detection (Phase 2):**
- **`src/bias_auditor.py`** — `BiasAuditor` with 6 bias detectors, synthetic profile generator.
- **`src/metrics.py`** — Evaluation metrics: precision@k, diversity, coverage, spread.

**Confidence + Self-Critique (Phase 3):**
- **`src/confidence.py`** — `ConfidenceScorer` with 5-signal weighted confidence.
- **`src/guardrails.py`** — Output guardrails based on confidence level.
- **`src/self_critique.py`** — LLM self-critique + offline fallback.

**Agentic Loop (Phase 4):**
- **`src/agent.py`** — `AgentLoop` with plan/execute/critique/refine/respond cycle.
- **`src/agent_tools.py`** — LLM-powered tools: `extract_profile()`, `select_strategy()`, `critique_results()`, `adjust_weights()`.
- **`src/conversation.py`** — `ConversationState` for multi-turn memory.
- **`src/agent_logger.py`** — Structured JSON logging to `logs/agent_runs/`.

**Frontend (Phase 5):**
- **`frontend/app.py`** — Streamlit entry point.
- **`frontend/pages/`** — 4 pages: Recommend, Explore, Audit, Agent Logs.

**MCP Server (Phase 6):**
- **`src/mcp_server.py`** — 5 tools + 4 resources for external AI agents.

**Tests (79 total):**
- **`tests/test_recommender.py`** — 2 original OOP tests.
- **`tests/test_rag.py`** — 14 RAG + scoring integration tests.
- **`tests/test_bias_auditor.py`** — 14 metrics + bias detection tests.
- **`tests/test_confidence.py`** — 14 confidence + guardrails + self-critique tests.
- **`tests/test_agent.py`** — 20 agent tools + loop tests (mock LLM).
- **`tests/test_mcp_server.py`** — 15 MCP handler tests.

### Scoring (`score_song`)

Returns a `(float, list[str])` tuple — score in ~[0, 1.24] and human-readable signal reasons.

**Base signals** (sum to 1.0 via `RankingStrategy` weights):
1. Genre match (binary 0/1)
2. Mood match (binary 0/1)
3. Energy proximity: `1 - |user.target_energy - song.energy|`
4. Acoustic fit: `song.acousticness` if `likes_acoustic` else `1 - song.acousticness`

**Advanced bonuses** (additive, up to ~0.24 total):
5. Popularity boost if `song.popularity >= user.min_popularity`
6. Release era bonus if song is within 5 years of `user.preferred_decade`
7. Mood tag overlap count from `user.preferred_tags` vs `song.detailed_moods`

### Weighting Strategies (`RankingStrategy`)

Four built-in constants in `src/recommender.py`:
- `DEFAULT`: genre 16%, mood 28%, energy 47%, acoustic 9%
- `GENRE_FIRST`: genre 50%, mood 25%, energy 20%, acoustic 5%
- `MOOD_FIRST`: genre 15%, mood 55%, energy 25%, acoustic 5%
- `ENERGY_FOCUSED`: genre 10%, mood 10%, energy 75%, acoustic 5%

### Diversity Re-ranking (`recommend_songs`)

Greedy loop over k iterations: score all remaining songs, apply artist repeat penalty (−0.30) and genre repeat penalty (−0.15) if already selected, pick the highest, remove it from the pool.

### OOP vs. Functional

The module exposes both:
- **Functional**: `recommend_songs(user, songs, k, strategy)` → `list[tuple[Song, float, list[str]]]`
- **OOP**: `Recommender(songs, strategy).recommend(user, k)` → `list[Song]`; `.explain_recommendation(user, song)` → `str`

Tests use the OOP interface; `main.py` uses the functional interface directly.

### LLM Provider Configuration

- Default provider is local Ollama (`--provider ollama`) and expects `llama3.1:8b` to be available.
- Anthropic is optional: install `anthropic` and set `ANTHROPIC_API_KEY`, then run with `--provider anthropic`.
