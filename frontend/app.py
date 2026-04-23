"""
GrooveGenius 2.0 — Streamlit Web UI

Entry point: streamlit run frontend/app.py
"""

import sys
import os

# Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from frontend.components import load_shared_state

st.set_page_config(
    page_title="GrooveGenius 2.0",
    page_icon="🎵",  # user explicitly built a music app with emoji in README
    layout="wide",
)

# Initialize shared state
load_shared_state()

# Sidebar
st.sidebar.title("GrooveGenius 2.0")
st.sidebar.markdown(
    "Self-critiquing music recommender with RAG, "
    "agentic refinement, and bias auditing."
)
st.sidebar.divider()
st.sidebar.markdown("**Navigation** — use the pages in the sidebar above.")
st.sidebar.markdown(
    f"Catalog: **{len(st.session_state.songs)}** songs | "
    f"**{len(st.session_state.valid_genres)}** genres | "
    f"**{len(st.session_state.valid_moods)}** moods"
)

# Landing page
st.title("GrooveGenius 2.0")
st.markdown("""
A self-critiquing music recommender that combines **content-based scoring**
with **RAG-enhanced knowledge**, **confidence estimation**, and **bias auditing**.

### Pages

- **Recommend** — Get personalized music recommendations via natural language or manual profile
- **Explore** — Browse the song catalog and compare ranking strategies
- **Audit** — Run automated bias detection across synthetic profiles
- **Agent Logs** — Review agent session traces and reasoning steps

### Quick Start

Use the **Recommend** page to try the manual profile builder (no LLM required),
or connect Ollama for the natural language agent mode.
""")

# Show system status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Songs", len(st.session_state.songs))
with col2:
    st.metric("Genres", len(st.session_state.valid_genres))
with col3:
    rag_status = "Enabled" if st.session_state.knowledge else "Disabled"
    st.metric("RAG Knowledge", rag_status)
