# music-recommendation-simulation-platform

## Original project and this extension

This repository extends the **AI110 music recommender simulation starter** (module show baseline). The original scope was a **content-based** recommender: load a fixed catalog from `data/songs.json`, score each song with explicit weighted signals (genre, mood, energy, acoustic fit), rank deterministically, and print human-readable explanations—no external services required.

**GrooveGenius 2.0** keeps that fast, reproducible core in `src/recommender.py` and adds integrated applied-AI features on top of the same catalog: **EchoSphere-RAG** (LangGraph + ChromaDB audio-feature retrieval + LLM explanations in `src/echosphere/`), **optional RAG** for partial genre/mood credit in fast mode (`src/rag.py`), **confidence scoring, guardrails, and offline self-critique** (`src/confidence.py`, `src/guardrails.py`, `src/self_critique.py`), a **bias auditing** pipeline with JSON reports (`src/bias_auditor.py`), a **Streamlit** front end (`frontend/`), **interactive agent** mode (`src/agent.py`, `src/main.py --interactive`), and an **MCP** server for external agents (`src/mcp_server.py`). Mode is selected per run (`--mode fast|agentic`), in the Streamlit sidebar, or via MCP tools.

**Documentation:** Architecture and data-flow diagrams → [docs/architecture/WORKFLOWS.md](docs/architecture/WORKFLOWS.md).

---

## Project summary

- **Fast mode** — deterministic weighted scoring (`src/recommender.py`). No LLM, no vector DB for core ranking.
- **Agentic mode (EchoSphere-RAG)** — LangGraph pipeline: **Ingestor** (ChromaDB), **Researcher** (mock trivia), **Reasoning** (LLM) → DJ-style explanations (`src/echosphere/`).

Pick the mode via `--mode fast|agentic` on the CLI, the Streamlit sidebar, or MCP (`echosphere_*` tools where applicable).

---

## Quick verification

- `pytest` — full automated harness.
- `python -m src.main` — fast batch: four profiles + strategy comparison tables.
- `python -m src.main --audit` — bias audit to console + `reports/bias_audit_*.json`.
- `python -m streamlit run frontend/app.py` — UI (recommend, explore, audit, logs); toggle fast / agentic in the sidebar.
- Agentic smoke: `ollama serve` → `python -m src.echosphere.vector_store` → `python -m src.echosphere.graph` (prints final JSON state) or `python -m src.main --batch --mode agentic`.

---

## Sample inputs and expected behavior

| Example | What you should see |
|---------|----------------------|
| `python -m src.main` | Four named profiles (e.g. High-Energy Pop, Chill Lofi); each prints a **tabulate** table (rank, title, artist, genre, score, reasons) plus confidence line and optional self-critique when confidence is very low. |
| `python -m src.main --audit` | Console summary of synthetic profiles tested and bias signatures; JSON written under `reports/`. |
| `python -m src.echosphere.graph` (after seeding Chroma, Ollama running) | JSON stdout with keys such as `user_request`, `retrieved_tracks`, `explanations`, `artist_trivia` (or `error` if DB/LLM missing). |

Additional copy-paste flow for agentic mode:

```bash
pip install -r requirements.txt
ollama pull llama3.2
# terminal 1: ollama serve
python -m src.echosphere.vector_store
python -m src.echosphere.graph
```

---

## Why this project matters (brief)

The starter optimized for transparent ranking; the extension adds **explainability under weak catalog coverage**, **auditable bias checks**, and an **agentic path** that still shares the same data and confidence layers. Reliability is addressed through tests, confidence labels, guardrail text—not by hiding low-quality matches.

---

## How the system works

Content-based focus: user preferences align to song features in `data/songs.json`. Full scoring tables, strategies, and diversity penalties are unchanged in spirit from the extended starter; see sections below.

### Song features

Each song stores attributes loaded from `data/songs.json` (genre, mood, energy, tempo, valence, danceability, acousticness, instrumentalness, speechiness, plus extended fields such as popularity and `detailed_moods` where implemented).

### User profile (fast mode)

Preferences drive scoring: genre, mood, target energy, likes acoustic, plus optional advanced fields (tags, decade, popularity) as implemented in `src/recommender.py`.

### Scoring and ranking

Weighted combination of signals; strategies (`DEFAULT`, `GENRE_FIRST`, `MOOD_FIRST`, `ENERGY_FOCUSED`) reweight the same functions. Greedy diversity penalties reduce repeated artist/genre in top-k.

### Data flow

Canonical diagrams (fast path, EchoSphere chain, reliability stack): **[docs/architecture/WORKFLOWS.md](docs/architecture/WORKFLOWS.md)**.

Fast mode in brief:

```
Input (user prefs + songs.json)
  → load_songs()
  → recommend_songs() / score_song() [optional RAG knowledge]
  → sort, top-k, explanations
  → ConfidenceScorer + guardrails (+ self-critique when enabled)
```

### Stress-test profiles

Four CLI profiles in `src/main.py` (`PROFILES`): High-Energy Pop, Chill Lofi, Deep Intense Rock, Conflicted Listener (adversarial high-energy classical).

### Known biases

Genre weight and catalog sparsity can dominate; mood cardinality limits matches; genre and mood are scored independently. Auditing helps surface systematic skew.

---

## Challenge implementations

Advanced attributes, multiple `RankingStrategy` presets, diversity penalties, and `tabulate` output are described in the original challenge specs—implementation lives in `src/recommender.py` and `src/main.py`.

---

## Getting started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure local or cloud models (optional):

   ```bash
   cp .env.example .env
   ```

   Edit `.env` for `OLLAMA_MODEL`, `ECHO_OLLAMA_MODEL`, or `OLLAMA_BASE_URL`. For cloud: `ONLINE_LLM_PROVIDER=gemini` + `GEMINI_API_KEY` (e.g. `gemini-2.5-flash`), or `anthropic` + `ANTHROPIC_API_KEY`.

4. Run the app:

   ```bash
   python -m src.main
   ```

### Ollama / cloud LLM (interactive + agentic)

Local:

```bash
ollama pull llama3.2
ollama serve
```

Cloud Anthropic (example):

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"
python -m src.main --interactive --provider anthropic
```

Cloud Gemini (example):

```bash
export ONLINE_LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.5-flash"
python -m streamlit run frontend/app.py
```

### Entry points

| Command | Description |
|---------|-------------|
| `python -m src.main` | Batch mode (four profiles + strategy comparison) — fast mode by default |
| `python -m src.main --batch --mode agentic` | Same profiles through EchoSphere-RAG (Ollama + seeded Chroma) |
| `python -m src.main --audit` | Bias audit: console + JSON report |
| `python -m src.main --interactive` | Conversational agent (fast pipeline + LLM tools) |
| `python -m streamlit run frontend/app.py` | Web UI with fast/agentic toggle |
| `python -m src.mcp_server` | MCP server (fast tools + `echosphere_*`) |
| `python -m src.echosphere.vector_store` | Seed or rebuild Chroma from `data/songs.json` |
| `python -m src.echosphere.graph` | Sample agentic query; prints final state JSON |

### EchoSphere-RAG first run

```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama serve
python -m src.echosphere.vector_store
python -m src.echosphere.graph
```

Pipeline edges match `build_graph()` in `src/echosphere/graph.py`: **ingestor** → **researcher** → **reasoning**. Node behavior: `src/echosphere/nodes.py` (`ingestor_node`, `researcher_node`, `reasoning_node`). System diagram: [docs/architecture/WORKFLOWS.md](docs/architecture/WORKFLOWS.md).

### Running tests

```bash
pytest
```

Focused:

```bash
pytest tests/test_recommender.py
```

### MCP integration

```bash
python -m src.mcp_server
```

Example MCP config:

```json
{
  "mcpServers": {
    "groovegenius": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/ai110-module3show-musicrecommendersimulation-starter"
    }
  }
}
```

Tools include `recommend_manual`, `recommend`, `explain_song`, `audit_bias`, `list_catalog`, and agentic helpers `echosphere_recommend`, `echosphere_ingest`, `echosphere_explain`.

---

## Experiments

Scoring weight experiments and profile comparisons can be reproduced from the CLI profiles and `compare_strategies()` in `src/main.py`.

---

## Limitations and risks

Sparse catalog coverage, no lyrics semantics in fast mode, and residual bias under weak genre/mood support. Confidence and audit outputs flag issues but do not replace richer offline metrics.

---

## Reflection (AI collaboration)

Narrative on AI-assisted development, helpful vs flawed suggestions, and future work: **[docs/analysis/ai_collaboration_reflection.md](docs/analysis/ai_collaboration_reflection.md)**.
