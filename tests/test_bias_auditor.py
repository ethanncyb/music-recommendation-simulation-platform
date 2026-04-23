"""Tests for the bias auditor and evaluation metrics."""

from collections import Counter

from src.recommender import load_songs, recommend_songs, DEFAULT
from src.bias_auditor import BiasAuditor, BiasSignature, AuditReport
from src.metrics import (
    top_k_score, score_spread, genre_hit_rate, mood_hit_rate,
    diversity_score, artist_diversity, catalog_coverage,
)


# ── Metrics unit tests ───────────────────────────────────────────────────

def _make_results(data):
    """Helper to create result tuples from simplified data."""
    return [
        ({"id": i, "genre": g, "mood": m, "artist": a}, score, "reason")
        for i, (g, m, a, score) in enumerate(data, start=1)
    ]


def test_top_k_score():
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("rock", "angry", "B", 0.7),
        ("jazz", "chill", "C", 0.5),
    ])
    assert top_k_score(results, 1) == 0.9
    assert top_k_score(results, 2) == 0.7
    assert top_k_score(results, 3) == 0.5
    assert top_k_score(results, 4) == 0.0  # out of bounds


def test_score_spread():
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("rock", "angry", "B", 0.3),
    ])
    assert abs(score_spread(results) - 0.6) < 0.001


def test_genre_hit_rate():
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("pop", "chill", "B", 0.8),
        ("rock", "angry", "C", 0.7),
    ])
    assert abs(genre_hit_rate(results, "pop") - 2/3) < 0.001
    assert abs(genre_hit_rate(results, "rock") - 1/3) < 0.001
    assert genre_hit_rate(results, "jazz") == 0.0


def test_mood_hit_rate():
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("rock", "happy", "B", 0.8),
        ("jazz", "chill", "C", 0.7),
    ])
    assert abs(mood_hit_rate(results, "happy") - 2/3) < 0.001


def test_diversity_score():
    # 3 unique genres out of 3 results = 1.0
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("rock", "angry", "B", 0.8),
        ("jazz", "chill", "C", 0.7),
    ])
    assert diversity_score(results) == 1.0

    # 1 unique genre out of 3 = 1/3
    results_same = _make_results([
        ("pop", "happy", "A", 0.9),
        ("pop", "chill", "B", 0.8),
        ("pop", "angry", "C", 0.7),
    ])
    assert abs(diversity_score(results_same) - 1/3) < 0.001


def test_artist_diversity():
    results = _make_results([
        ("pop", "happy", "A", 0.9),
        ("rock", "angry", "A", 0.8),  # same artist
        ("jazz", "chill", "B", 0.7),
    ])
    assert abs(artist_diversity(results) - 2/3) < 0.001


def test_catalog_coverage():
    all_results = [
        _make_results([("pop", "happy", "A", 0.9), ("rock", "angry", "B", 0.8)]),
        _make_results([("pop", "happy", "A", 0.9), ("jazz", "chill", "C", 0.7)]),
    ]
    # IDs: first list has 1,2; second has 1,2 (new IDs since enumerate restarts)
    # But the helper creates new IDs each time. Let's make it explicit:
    results1 = [
        ({"id": 1, "genre": "pop", "mood": "happy", "artist": "A"}, 0.9, "r"),
        ({"id": 2, "genre": "rock", "mood": "angry", "artist": "B"}, 0.8, "r"),
    ]
    results2 = [
        ({"id": 1, "genre": "pop", "mood": "happy", "artist": "A"}, 0.9, "r"),
        ({"id": 3, "genre": "jazz", "mood": "chill", "artist": "C"}, 0.7, "r"),
    ]
    assert catalog_coverage([results1, results2], 10) == 0.3  # 3 unique / 10


def test_empty_results():
    assert top_k_score([], 1) == 0.0
    assert score_spread([]) == 0.0
    assert genre_hit_rate([], "pop") == 0.0
    assert diversity_score([]) == 0.0
    assert artist_diversity([]) == 0.0


# ── Bias Auditor tests ───────────────────────────────────────────────────

def _load_auditor():
    songs = load_songs("data/songs.json")
    return BiasAuditor(songs, strategy=DEFAULT), songs


def test_generate_audit_profiles_count():
    auditor, _ = _load_auditor()
    profiles = auditor.generate_audit_profiles()
    # 27 genres x 3 energy x 2 acoustic = 162 natural + ~5 contradiction profiles
    assert len(profiles) >= 162
    assert len(profiles) <= 200


def test_genre_lockout_detected():
    auditor, songs = _load_auditor()
    biases = auditor._detect_genre_lockout()
    expected_locked_genres = sum(1 for count in Counter(s["genre"] for s in songs).values() if count == 1)
    assert len(biases) == 1
    assert biases[0].name == "genre_lockout"
    assert biases[0].severity == "high"
    assert biases[0].affected_count == expected_locked_genres


def test_mood_desert_detected():
    auditor, _ = _load_auditor()
    biases = auditor._detect_mood_desert()
    assert len(biases) == 1
    assert biases[0].name == "mood_desert"
    assert biases[0].severity == "medium"


def test_contradictions_detected():
    auditor, _ = _load_auditor()
    profiles = auditor.generate_audit_profiles()
    contradiction_profiles = [p for p in profiles if p["label"].startswith("CONTRADICTION")]
    assert len(contradiction_profiles) >= 3  # at least classical, ambient, lofi


def test_run_audit_returns_valid_report():
    auditor, songs = _load_auditor()
    report = auditor.run_audit()
    assert isinstance(report, AuditReport)
    assert report.strategy_name == "Default"
    assert report.profiles_tested >= 162
    assert report.songs_in_catalog == len(songs)
    assert len(report.biases) >= 1
    assert report.catalog_stats["catalog_coverage"] > 0
    assert len(report.profile_summaries) == report.profiles_tested


def test_audit_report_bias_structure():
    auditor, _ = _load_auditor()
    report = auditor.run_audit()
    for bias in report.biases:
        assert isinstance(bias, BiasSignature)
        assert bias.name
        assert bias.severity in ("high", "medium", "low")
        assert bias.affected_count >= 0
        assert bias.total_count > 0
        assert bias.description
        assert bias.suggestion
