"""
Page 1: Recommend — Natural Language + Manual Profile Builder
"""

import streamlit as st

from frontend.components import (
    load_shared_state, render_results_table, render_confidence_badge,
    render_strategy_selector, render_mode_selector, render_agentic_results,
    run_agentic, check_ollama_available, get_online_llm, STRATEGIES,
)
from src.recommender import recommend_songs
from src.guardrails import apply_guardrails
from src.self_critique import self_critique_offline

load_shared_state()
mode = render_mode_selector(key="recommend_mode")

st.title("Recommend")
st.markdown("Get personalized music recommendations using natural language or manual preferences.")

if mode == "agentic":
    st.caption(
        "Agentic mode: EchoSphere-RAG LangGraph (Ingestor -> Researcher -> "
        "Reasoning) over ChromaDB + local Ollama."
    )
elif mode == "agentic_online":
    st.caption(
        "Agentic mode (online LLM): EchoSphere-RAG LangGraph (Ingestor -> "
        "Researcher -> Reasoning) over ChromaDB + online API "
        "(configured via ONLINE_LLM_PROVIDER in .env)."
    )

tab_nl, tab_manual = st.tabs(["Natural Language", "Manual Profile"])

# ── Tab A: Natural Language ──────────────────────────────────────────────

with tab_nl:
    # Determine which LLM is available for this mode
    if mode == "fast":
        online_llm = None
        llm_ready = False
        st.info(
            "Natural Language is disabled in Fast mode. "
            "Natural Language recommendations require an LLM. "
            "Switch to Agentic (local LLM) or Agentic (online LLM).",
            icon="ℹ️",
        )
    elif mode == "agentic_online":
        online_llm, online_err = get_online_llm()
        llm_ready = online_llm is not None
        if not llm_ready:
            st.warning(
                f"Online LLM not available: {online_err}. "
                "Configure ONLINE_LLM_PROVIDER plus matching API key/model in .env "
                "(for Gemini, prefer GEMINI_MODEL=gemini-2.5-flash).",
                icon="⚠️",
            )
    else:
        online_llm = None
        llm_ready = check_ollama_available()
        if not llm_ready:
            st.warning(
                "Ollama is not running. Natural language mode requires a local LLM. "
                "Install Ollama (https://ollama.com) and run: `ollama pull llama3.2`. "
                "Use the **Manual Profile** tab instead, or switch to **Agentic (online LLM)** mode.",
                icon="⚠️",
            )

    query = st.text_input(
        "What kind of music are you looking for?",
        placeholder="e.g., moody driving music for a night road trip",
        disabled=not llm_ready,
        key="nl_query",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        nl_k = st.number_input("Top-K", 1, 10, 5, key="nl_k")

    if st.button("Recommend", disabled=not llm_ready or not query, key="nl_btn"):
        if mode in ("agentic", "agentic_online"):
            llm_for_pipeline = online_llm if mode == "agentic_online" else None
            with st.spinner("Running EchoSphere-RAG pipeline..."):
                profile = {
                    "genre": None, "mood": None,
                    "energy": 0.6, "likes_acoustic": False, "top_k": nl_k,
                }
                state = run_agentic(profile, query=query, llm=llm_for_pipeline)
            render_agentic_results(state)
            if state.get("agent_log_path"):
                st.caption(f"Session log saved: `{state['agent_log_path']}`")
            st.stop()

        with st.spinner("Agent is thinking..."):
            from src.llm_provider import get_provider
            from src.agent import AgentLoop

            try:
                llm = get_provider("ollama")
                agent = AgentLoop(
                    llm=llm,
                    songs=st.session_state.songs,
                    knowledge=st.session_state.knowledge,
                )
                result = agent.run(query, k=nl_k)

                # Reasoning trace
                with st.expander("Agent Reasoning", expanded=False):
                    for step in result.get("reasoning_trace", []):
                        st.text(step)

                    profile = result.get("profile", {})
                    st.json(profile)

                # Results
                render_results_table(result.get("results", []))

                # Confidence
                from src.confidence import ConfidenceReport
                conf = result.get("confidence", {})
                report = ConfidenceReport(
                    overall_confidence=conf.get("score", 0),
                    confidence_label=conf.get("label", "?"),
                    signals=conf.get("signals", {}),
                    warnings=conf.get("warnings", []),
                    suggestion=None,
                )
                render_confidence_badge(report)

                # Guardrail / critique
                if result.get("guardrail"):
                    st.warning(result["guardrail"])
                if result.get("critique"):
                    with st.expander("Self-Critique", expanded=False):
                        st.write(result["critique"])

                # Save agent for refinement
                st.session_state["nl_agent"] = agent

            except Exception as e:
                st.error(f"Agent error: {e}")

    # Refinement
    if "nl_agent" in st.session_state:
        refine = st.text_input(
            "Refine your request:",
            placeholder="e.g., less electronic, more acoustic",
            key="nl_refine",
        )
        if st.button("Refine", key="nl_refine_btn") and refine:
            with st.spinner("Refining..."):
                try:
                    result = st.session_state["nl_agent"].chat(refine)
                    render_results_table(result.get("results", []))
                except Exception as e:
                    st.error(f"Refinement error: {e}")

# ── Tab B: Manual Profile ────────────────────────────────────────────────

with tab_manual:
    col1, col2 = st.columns(2)

    with col1:
        genre = st.selectbox("Genre", st.session_state.valid_genres, key="m_genre")
        mood = st.selectbox("Mood", st.session_state.valid_moods, key="m_mood")
        energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.05, key="m_energy")
        acoustic = st.toggle("Likes Acoustic", key="m_acoustic")

    with col2:
        strategy = render_strategy_selector(key="m_strategy")
        k = st.number_input("Top-K", 1, 10, 5, key="m_k")
        use_rag = st.toggle("Enable RAG", value=True, key="m_rag")

    with st.expander("Advanced Options"):
        min_pop = st.slider("Min Popularity", 0, 100, 0, key="m_pop")
        decade_options = [None, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
        decade_labels = ["None"] + [f"{d}s" for d in decade_options[1:]]
        decade_idx = st.selectbox("Preferred Decade", range(len(decade_labels)),
                                  format_func=lambda i: decade_labels[i], key="m_decade")
        preferred_decade = decade_options[decade_idx]

        all_tags = sorted(set(
            tag for s in st.session_state.songs
            for tag in s.get("detailed_moods", "").split("|") if tag
        ))
        preferred_tags = st.multiselect("Mood Tags", all_tags, key="m_tags")

    if st.button("Recommend", key="manual_btn"):
        profile = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "likes_acoustic": acoustic,
            "min_popularity": min_pop,
            "preferred_decade": preferred_decade,
            "preferred_tags": preferred_tags if preferred_tags else None,
            "top_k": k,
        }

        if mode in ("agentic", "agentic_online"):
            if mode == "agentic_online":
                online_llm_manual, online_err_manual = get_online_llm()
                if not online_llm_manual:
                    st.error(
                        "Online LLM not available: "
                        f"{online_err_manual}. "
                        "Set ONLINE_LLM_PROVIDER plus matching API key/model in .env, "
                        "then retry."
                    )
                    st.stop()
                llm_for_manual = online_llm_manual
            else:
                llm_for_manual = None

            with st.spinner("Running EchoSphere-RAG pipeline..."):
                state = run_agentic(profile, llm=llm_for_manual)
            render_agentic_results(state)
            if state.get("agent_log_path"):
                st.caption(f"Session log saved: `{state['agent_log_path']}`")

            scorer = st.session_state.confidence_scorer
            confidence = scorer.compute(profile, state)
            render_confidence_badge(confidence)

            guardrail = apply_guardrails(confidence, state.get("retrieved_tracks") or [])
            if guardrail["guardrail_message"]:
                st.warning(guardrail["guardrail_message"])
        else:
            knowledge = st.session_state.knowledge if use_rag else None
            results = recommend_songs(
                profile, st.session_state.songs, k=k,
                strategy=strategy, knowledge=knowledge,
            )

            render_results_table(results)

            scorer = st.session_state.confidence_scorer
            confidence = scorer.compute(profile, results, knowledge)
            render_confidence_badge(confidence)

            guardrail = apply_guardrails(confidence, results)
            if guardrail["guardrail_message"]:
                st.warning(guardrail["guardrail_message"])
            if guardrail["show_self_critique"]:
                critique = self_critique_offline(profile, results, confidence)
                with st.expander("Self-Critique", expanded=True):
                    st.write(critique)
