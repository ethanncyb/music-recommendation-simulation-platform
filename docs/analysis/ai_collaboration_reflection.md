# AI collaboration and system design reflection

How AI tools were used during development, one helpful and one flawed suggestion, and honest limitations.

## How AI was used

- **Exploration and scaffolding:** Large language models helped navigate LangGraph wiring, MCP handler shapes, and test layout faster than reading unrelated tutorials.
- **Debugging:** Stack traces and failing `pytest` cases were pasted into AI-assisted sessions to narrow hypotheses (imports, async MCP, Chroma path issues).
- **Documentation:** Drafts of README sections and traceability tables were iterated with AI assistance, then checked against the actual code paths.

## One helpful suggestion

Consolidating all LLM access behind `src/llm_provider.py` was reinforced early: it kept Ollama, Gemini, and Anthropic paths consistent for EchoSphere, interactive mode, and tests without scattering provider-specific imports.

## One flawed suggestion

An early suggestion to “always block” low-confidence recommendations would have broken the assignment’s goal of showing weak matches with transparent warnings. The better match for the codebase is **guardrails that annotate** (`src/guardrails.py`) rather than hard failure, preserving auditability while still signaling risk.

## Limitations and future work

- Catalog size and genre/mood cardinality still dominate edge behavior; auditing surfaces gaps but does not fix sparse coverage.
- Agentic quality depends on local or cloud LLM availability; CI-style runs lean on mocks where possible.
- Future work could add richer user feedback loops persisted across sessions and stronger offline evaluation metrics tied to human judgments.
