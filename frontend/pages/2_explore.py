"""
Page 2: Explore — Catalog Browser + Strategy Comparison
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from frontend.components import (
    load_shared_state, render_mode_selector, profile_to_dna, STRATEGIES,
)
from src.recommender import recommend_songs, score_song

load_shared_state()
render_mode_selector(key="explore_mode")

st.title("Explore")
st.markdown("Browse the song catalog and compare how different strategies rank songs.")

# ── Catalog Browser ──────────────────────────────────────────────────────

st.header("Song Catalog")

col1, col2, col3 = st.columns(3)
with col1:
    genre_filter = st.selectbox(
        "Filter by Genre",
        ["All"] + st.session_state.valid_genres,
        key="explore_genre",
    )
with col2:
    mood_filter = st.selectbox(
        "Filter by Mood",
        ["All"] + st.session_state.valid_moods,
        key="explore_mood",
    )
with col3:
    search = st.text_input("Search title/artist", key="explore_search")

songs = st.session_state.songs
filtered = songs
if genre_filter != "All":
    filtered = [s for s in filtered if s["genre"] == genre_filter]
if mood_filter != "All":
    filtered = [s for s in filtered if s["mood"] == mood_filter]
if search:
    search_lower = search.lower()
    filtered = [s for s in filtered if search_lower in s["title"].lower()
                or search_lower in s["artist"].lower()]

df = pd.DataFrame([
    {
        "Title": s["title"],
        "Artist": s["artist"],
        "Genre": s["genre"],
        "Mood": s["mood"],
        "Energy": s["energy"],
        "Acousticness": s["acousticness"],
        "Popularity": s["popularity"],
        "Year": s["release_year"],
    }
    for s in filtered
])

st.dataframe(df, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(filtered)} of {len(songs)} songs")

# ── Strategy Comparison ──────────────────────────────────────────────────

st.header("Strategy Comparison")
st.markdown("Compare how each strategy ranks songs for the same profile.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    cmp_genre = st.selectbox("Genre", st.session_state.valid_genres, key="cmp_genre")
with col2:
    cmp_mood = st.selectbox("Mood", st.session_state.valid_moods, key="cmp_mood")
with col3:
    cmp_energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.05, key="cmp_energy")
with col4:
    cmp_acoustic = st.toggle("Acoustic", key="cmp_acoustic")

profile = {
    "genre": cmp_genre,
    "mood": cmp_mood,
    "energy": cmp_energy,
    "likes_acoustic": cmp_acoustic,
}

if st.button("Compare Strategies", key="cmp_btn"):
    rows = []
    for name, strat in STRATEGIES.items():
        results = recommend_songs(profile, songs, k=1, strategy=strat)
        if results:
            song, score, explanation = results[0]
            rows.append({
                "Strategy": name,
                "#1 Title": song["title"],
                "Artist": song["artist"],
                "Genre": song["genre"],
                "Score": round(score, 4),
            })

    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Bar chart of scores
    fig = px.bar(
        cmp_df, x="Strategy", y="Score", color="Strategy",
        title="Top-1 Score by Strategy",
        text="Score",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.3])
    st.plotly_chart(fig, use_container_width=True)

# ── Score Distribution ───────────────────────────────────────────────────

st.header("Score Distribution")
st.markdown("See how all 30 songs score for your profile under the selected strategy.")

dist_strategy_name = st.selectbox(
    "Strategy", list(STRATEGIES.keys()), key="dist_strat"
)
dist_strategy = STRATEGIES[dist_strategy_name]

if st.button("Show Distribution", key="dist_btn"):
    all_scores = []
    knowledge = st.session_state.knowledge
    for song in songs:
        score, reasons = score_song(profile, song, dist_strategy, knowledge)
        all_scores.append({
            "Title": song["title"],
            "Artist": song["artist"],
            "Genre": song["genre"],
            "Score": round(score, 4),
        })

    score_df = pd.DataFrame(all_scores).sort_values("Score", ascending=False)

    fig = px.bar(
        score_df, x="Title", y="Score", color="Genre",
        title=f"All Song Scores — {dist_strategy_name} Strategy",
        hover_data=["Artist", "Genre"],
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(score_df, use_container_width=True, hide_index=True)

# ── Vector Nearest-Neighbors (EchoSphere ChromaDB) ───────────────────────

st.header("Vector Nearest-Neighbors (EchoSphere ChromaDB)")
st.markdown(
    "Query the agentic-mode vector store directly. Useful for debugging how "
    "the Ingestor node ranks tracks before any post-filtering or LLM call."
)

nn_k = st.slider("Neighbors to return", 1, 15, 5, key="nn_k")

if st.button("Find nearest neighbors", key="nn_btn"):
    try:
        from src.echosphere.vector_store import (
            build_query_vector, get_collection,
        )

        dna = profile_to_dna(profile)
        collection = get_collection()
        result = collection.query(
            query_embeddings=[build_query_vector(dna)],
            n_results=nn_k,
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows = []
        for idx, meta in enumerate(metadatas):
            if meta is None:
                continue
            rows.append({
                "#": idx + 1,
                "Title": meta.get("title"),
                "Artist": meta.get("artist"),
                "Genre": meta.get("genre"),
                "Mood": meta.get("mood"),
                "Energy": meta.get("energy"),
                "Distance": round(float(distances[idx]), 4) if idx < len(distances) else None,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(
            f"ChromaDB unavailable: {exc}. Seed it with "
            "`python -m src.echosphere.vector_store`."
        )
