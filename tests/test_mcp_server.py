"""Tests for MCP server handler functions.

Tests the sync handler functions directly — no MCP transport or async needed.
"""

import json

from src.mcp_server import (
    handle_recommend_manual,
    handle_explain_song,
    handle_list_catalog,
    handle_audit_bias,
    handle_resource_catalog_songs,
    handle_resource_catalog_stats,
    handle_resource_strategies,
)


# ── recommend_manual ─────────────────────────────────────────────────────

def test_recommend_manual_returns_results():
    result = handle_recommend_manual({
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    })
    assert "recommendations" in result
    assert len(result["recommendations"]) == 5
    assert "confidence" in result
    assert result["confidence"]["label"] in ("high", "medium", "low", "very_low")


def test_recommend_manual_with_strategy():
    result = handle_recommend_manual({
        "genre": "rock",
        "mood": "angry",
        "energy": 0.9,
        "likes_acoustic": False,
        "strategy": "energy_focused",
        "k": 3,
    })
    assert len(result["recommendations"]) == 3


def test_recommend_manual_result_structure():
    result = handle_recommend_manual({
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.3,
        "likes_acoustic": True,
    })
    rec = result["recommendations"][0]
    assert "title" in rec
    assert "artist" in rec
    assert "genre" in rec
    assert "score" in rec
    assert "reasons" in rec
    assert isinstance(rec["score"], float)


# ── explain_song ─────────────────────────────────────────────────────────

def test_explain_song_known():
    result = handle_explain_song({
        "song_title": "Sunrise City",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    })
    assert "error" not in result
    assert result["song"] == "Sunrise City"
    assert result["artist"] == "Neon Echo"
    assert result["score"] > 0
    assert len(result["reasons"]) > 0


def test_explain_song_case_insensitive():
    result = handle_explain_song({
        "song_title": "sunrise city",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    })
    assert result["song"] == "Sunrise City"


def test_explain_song_unknown():
    result = handle_explain_song({
        "song_title": "Nonexistent Song",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    })
    assert "error" in result


# ── list_catalog ─────────────────────────────────────────────────────────

def test_list_catalog_all():
    result = handle_list_catalog({})
    assert len(result) == 30
    assert result[0]["title"]  # has title field


def test_list_catalog_genre_filter():
    result = handle_list_catalog({"genre_filter": "pop"})
    assert len(result) >= 1
    assert all(s["genre"] == "pop" for s in result)


def test_list_catalog_mood_filter():
    result = handle_list_catalog({"mood_filter": "chill"})
    assert len(result) >= 1
    assert all(s["mood"] == "chill" for s in result)


def test_list_catalog_combined_filter():
    result = handle_list_catalog({"genre_filter": "lofi", "mood_filter": "chill"})
    assert all(s["genre"] == "lofi" and s["mood"] == "chill" for s in result)


# ── audit_bias ───────────────────────────────────────────────────────────

def test_audit_bias_returns_report():
    result = handle_audit_bias({})
    assert "strategy" in result
    assert "profiles_tested" in result
    assert "biases" in result
    assert isinstance(result["biases"], list)
    assert result["profiles_tested"] >= 100


def test_audit_bias_with_strategy():
    result = handle_audit_bias({"strategy": "genre_first"})
    assert result["strategy"] == "Genre-First"


# ── Resources ────────────────────────────────────────────────────────────

def test_resource_catalog_songs():
    data = json.loads(handle_resource_catalog_songs())
    assert len(data) == 30


def test_resource_catalog_stats():
    data = json.loads(handle_resource_catalog_stats())
    assert data["total_songs"] == 30
    assert data["unique_genres"] == 27
    assert data["unique_moods"] == 25


def test_resource_strategies():
    data = json.loads(handle_resource_strategies())
    assert "default" in data
    assert "genre_first" in data
    # Weights should sum to ~1.0
    for name, weights in data.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"
