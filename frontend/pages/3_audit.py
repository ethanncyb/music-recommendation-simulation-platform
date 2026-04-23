"""
Page 3: Audit — Bias Detection Dashboard
"""

import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.components import (
    load_shared_state, render_mode_selector, render_strategy_selector, STRATEGIES,
)
from src.bias_auditor import BiasAuditor, AuditReport, BiasSignature

load_shared_state()
mode = render_mode_selector(key="audit_mode")

st.title("Bias Audit")
st.markdown("Run automated bias detection across synthetic user profiles.")


def _load_saved_report(path: str) -> AuditReport:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    biases = [BiasSignature(**b) for b in data.get("biases", [])]
    return AuditReport(
        timestamp=data.get("timestamp", ""),
        strategy_name=data.get("strategy_name", "unknown"),
        profiles_tested=data.get("profiles_tested", 0),
        songs_in_catalog=data.get("songs_in_catalog", 0),
        biases=biases,
        catalog_stats=data.get("catalog_stats", {}),
        profile_summaries=data.get("profile_summaries", []),
        pipeline=data.get("pipeline", "fast"),
    )


def _latest_saved_report_path(report_dir: str = "reports") -> str | None:
    if not os.path.isdir(report_dir):
        return None

    files = sorted(
        [f for f in os.listdir(report_dir) if f.startswith("bias_audit_") and f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return None
    return os.path.join(report_dir, files[0])

# ── Run Audit ────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])
with col1:
    strategy = render_strategy_selector(key="audit_strategy")
with col2:
    use_rag = st.toggle("Enable RAG", value=False, key="audit_rag")

run_col, load_col = st.columns(2)
with run_col:
    run_audit_clicked = st.button("Run Audit", key="audit_btn")
with load_col:
    load_latest_clicked = st.button("Load Latest Saved Report", key="audit_load_latest")

if run_audit_clicked:
    spinner_msg = (
        "Running agentic bias audit (EchoSphere-RAG per profile — this is slow)..."
        if mode == "agentic"
        else "Running bias audit across synthetic profiles..."
    )
    with st.spinner(spinner_msg):
        knowledge = st.session_state.knowledge if use_rag else None
        auditor = BiasAuditor(st.session_state.songs, strategy=strategy, knowledge=knowledge)
        report = auditor.run_audit(pipeline=mode)
        st.session_state["audit_report"] = report
        auditor.save_report(report)

if load_latest_clicked:
    latest_path = _latest_saved_report_path()
    if latest_path is None:
        st.warning("No saved reports found in reports/.")
    else:
        st.session_state["audit_report"] = _load_saved_report(latest_path)
        st.success(f"Loaded saved report: {os.path.basename(latest_path)}")

if "audit_report" not in st.session_state:
    st.info("Click **Run Audit** to generate a bias report.")
    st.stop()

report = st.session_state["audit_report"]

# ── Summary Metrics ──────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Profiles Tested", report.profiles_tested)
with col2:
    st.metric("Biases Found", len(report.biases))
with col3:
    high_count = sum(1 for b in report.biases if b.severity == "high")
    st.metric("High Severity", high_count)
with col4:
    coverage = report.catalog_stats.get("catalog_coverage", 0)
    st.metric("Catalog Coverage", f"{coverage:.1%}")

# ── Bias Summary Table ───────────────────────────────────────────────────

st.header("Detected Biases")

if report.biases:
    bias_rows = []
    for b in report.biases:
        bias_rows.append({
            "Bias": b.name.replace("_", " ").title(),
            "Severity": b.severity.upper(),
            "Affected": f"{b.affected_count}/{b.total_count}",
            "Description": b.description[:120],
            "Suggestion": b.suggestion,
        })
    bias_df = pd.DataFrame(bias_rows)
    st.dataframe(bias_df, use_container_width=True, hide_index=True)
else:
    st.success("No biases detected.")

# ── Catalog Coverage Heatmap ─────────────────────────────────────────────

st.header("Catalog Coverage Heatmap")
st.markdown("Genre vs Mood — color shows number of songs. Most cells are empty (sparsity).")

songs = st.session_state.songs
genres = sorted(set(s["genre"] for s in songs))
moods = sorted(set(s["mood"] for s in songs))

# Build count matrix
counts = {}
for s in songs:
    key = (s["genre"], s["mood"])
    counts[key] = counts.get(key, 0) + 1

matrix = []
for g in genres:
    row = [counts.get((g, m), 0) for m in moods]
    matrix.append(row)

fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=moods,
    y=genres,
    colorscale="Blues",
    text=matrix,
    texttemplate="%{text}",
    hovertemplate="Genre: %{y}<br>Mood: %{x}<br>Songs: %{z}<extra></extra>",
))
fig.update_layout(
    height=600,
    xaxis_title="Mood",
    yaxis_title="Genre",
    xaxis_tickangle=-45,
)
st.plotly_chart(fig, use_container_width=True)

# ── Score Distribution by Energy Group ───────────────────────────────────

st.header("Score Distribution by Energy Level")

summaries = report.profile_summaries
if summaries:
    for s in summaries:
        e = s["energy"]
        if e <= 0.3:
            s["energy_group"] = "Low (<=0.3)"
        elif e <= 0.6:
            s["energy_group"] = "Mid (0.3-0.6)"
        else:
            s["energy_group"] = "High (>0.6)"

    sum_df = pd.DataFrame(summaries)

    fig = px.box(
        sum_df, x="energy_group", y="top1_score",
        title="Top-1 Score Distribution by Energy Group",
        labels={"top1_score": "Top-1 Score", "energy_group": "Energy Group"},
        color="energy_group",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Acoustic comparison
    sum_df["acoustic_label"] = sum_df["likes_acoustic"].map({True: "Acoustic", False: "Non-Acoustic"})
    fig2 = px.box(
        sum_df, x="acoustic_label", y="top1_score",
        title="Top-1 Score: Acoustic vs Non-Acoustic Preferences",
        labels={"top1_score": "Top-1 Score", "acoustic_label": "Preference"},
        color="acoustic_label",
    )
    st.plotly_chart(fig2, use_container_width=True)
