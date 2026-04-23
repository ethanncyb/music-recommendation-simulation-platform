"""
Page 4: Agent Logs — Session trace viewer
"""

import json
import os

import streamlit as st

from frontend.components import load_shared_state, render_mode_selector

load_shared_state()
mode = render_mode_selector(key="logs_mode")

load_dir = "logs/agent_runs"

st.title("Agent Logs")
st.markdown("Browse and inspect saved agent session traces.")
st.caption(
    "Note: both fast and agentic modes log to `logs/agent_runs/`. "
    "Use the sidebar toggle to change the default mode on other pages."
)

# List available logs
if not os.path.isdir(load_dir):
    st.info("No logs directory found. Run the interactive agent to generate logs.")
    st.stop()

log_files = sorted(
    [f for f in os.listdir(load_dir) if f.endswith(".json")],
    reverse=True,
)

if not log_files:
    st.info(
        "No agent logs yet. Use `python -m src.main --interactive` "
        "or the Natural Language tab on the Recommend page to generate sessions."
    )
    st.stop()

selected = st.selectbox("Select session", log_files, key="log_select")

# Load and display
path = os.path.join(load_dir, selected)
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Session summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Session ID", data.get("session_id", "?"))
with col2:
    st.metric("Steps", data.get("step_count", 0))
with col3:
    total_ms = data.get("total_duration_ms", 0)
    st.metric("Duration", f"{total_ms}ms")

# Timeline
steps = data.get("steps", [])
if steps:
    st.header("Step Timeline")

    timeline_parts = []
    for s in steps:
        name = s.get("step", "?").upper()
        dur = s.get("duration_ms", 0)
        timeline_parts.append(f"{name} ({dur}ms)")
    st.text(" -> ".join(timeline_parts))

    # Step details
    st.header("Step Details")
    for i, s in enumerate(steps):
        step_name = s.get("step", "?").upper()
        dur = s.get("duration_ms", 0)
        errors = s.get("errors", [])
        error_tag = " [ERRORS]" if errors else ""

        with st.expander(f"{step_name} — {dur}ms{error_tag}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input:**")
                st.json(s.get("input_data", {}))
            with col2:
                st.markdown("**Output:**")
                st.json(s.get("output_data", {}))

            if s.get("llm_reasoning"):
                st.markdown("**LLM Reasoning:**")
                st.text(s["llm_reasoning"])

            if errors:
                st.markdown("**Errors:**")
                for e in errors:
                    st.error(e)

            st.caption(f"Timestamp: {s.get('timestamp', '?')}")
