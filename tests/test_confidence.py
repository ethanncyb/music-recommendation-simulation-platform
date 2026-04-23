"""Tests for confidence scoring, guardrails, and self-critique."""

from src.recommender import load_songs, recommend_songs, DEFAULT
from src.confidence import ConfidenceScorer, ConfidenceReport
from src.guardrails import apply_guardrails, format_confidence_badge
from src.self_critique import self_critique_offline


# ── Test helpers ─────────────────────────────────────────────────────────

def _load():
    songs = load_songs("data/songs.json")
    scorer = ConfidenceScorer(songs)
    return songs, scorer


def _run_profile(songs, prefs, scorer):
    results = recommend_songs(prefs, songs, k=5, strategy=DEFAULT)
    report = scorer.compute(prefs, results)
    return results, report


# ── ConfidenceScorer tests ───────────────────────────────────────────────

def test_high_energy_pop_is_high_confidence():
    songs, scorer = _load()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False}
    _, report = _run_profile(songs, prefs, scorer)
    assert report.confidence_label == "high"
    assert report.overall_confidence >= 0.75


def test_conflicted_listener_is_low_confidence():
    songs, scorer = _load()
    prefs = {"genre": "classical", "mood": "sad", "energy": 0.9, "likes_acoustic": True}
    _, report = _run_profile(songs, prefs, scorer)
    assert report.confidence_label in ("low", "very_low")
    assert report.overall_confidence < 0.50


def test_perfect_match_is_high_confidence():
    """A profile that matches the top song on all 4 signals should score high."""
    songs, scorer = _load()
    # Sunrise City: pop, happy, energy=0.82, acousticness=0.18
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.82, "likes_acoustic": False}
    _, report = _run_profile(songs, prefs, scorer)
    assert report.overall_confidence >= 0.70


def test_no_genre_or_mood_match_is_lower():
    """A genre/mood with no catalog match should score lower."""
    songs, scorer = _load()
    prefs = {"genre": "classical", "mood": "angry", "energy": 0.5, "likes_acoustic": False}
    _, report = _run_profile(songs, prefs, scorer)
    assert report.overall_confidence < 0.65


def test_confidence_signals_present():
    songs, scorer = _load()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    _, report = _run_profile(songs, prefs, scorer)
    assert "score_magnitude" in report.signals
    assert "score_gap" in report.signals
    assert "catalog_coverage" in report.signals
    assert "preference_coherence" in report.signals
    assert "signal_activation" in report.signals


def test_confidence_range():
    songs, scorer = _load()
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    _, report = _run_profile(songs, prefs, scorer)
    assert 0.0 <= report.overall_confidence <= 1.0
    for v in report.signals.values():
        assert 0.0 <= v <= 1.0


def test_empty_results():
    songs, scorer = _load()
    report = scorer.compute({"genre": "pop", "mood": "happy", "energy": 0.5}, [])
    assert report.confidence_label == "very_low"
    assert report.overall_confidence == 0.0


def test_coherence_detects_contradiction():
    songs, scorer = _load()
    # Classical + high energy = contradiction
    prefs = {"genre": "classical", "mood": "peaceful", "energy": 0.95, "likes_acoustic": True}
    _, report = _run_profile(songs, prefs, scorer)
    assert report.signals["preference_coherence"] < 0.7


# ── Guardrails tests ─────────────────────────────────────────────────────

def test_guardrails_high_confidence():
    report = ConfidenceReport(
        overall_confidence=0.85, confidence_label="high",
        signals={}, warnings=[], suggestion=None,
    )
    guardrail = apply_guardrails(report, [])
    assert guardrail["guardrail_message"] is None
    assert guardrail["show_self_critique"] is False


def test_guardrails_low_confidence():
    report = ConfidenceReport(
        overall_confidence=0.35, confidence_label="low",
        signals={}, warnings=["Weak match"], suggestion="Try a different genre.",
    )
    guardrail = apply_guardrails(report, [])
    assert guardrail["guardrail_message"] is not None
    assert "limited match" in guardrail["guardrail_message"].lower() or "Try" in guardrail["guardrail_message"]
    assert guardrail["show_self_critique"] is True


def test_guardrails_very_low_triggers_critique():
    report = ConfidenceReport(
        overall_confidence=0.15, confidence_label="very_low",
        signals={}, warnings=["No match"], suggestion=None,
    )
    guardrail = apply_guardrails(report, [])
    assert guardrail["show_self_critique"] is True


def test_format_confidence_badge():
    for label, expected_fragment in [
        ("high", "HIGH"),
        ("medium", "MEDIUM"),
        ("low", "LOW"),
        ("very_low", "VERY LOW"),
    ]:
        report = ConfidenceReport(
            overall_confidence=0.5, confidence_label=label,
            signals={}, warnings=[],
        )
        badge = format_confidence_badge(report)
        assert expected_fragment in badge


# ── Self-critique offline tests ──────────────────────────────────────────

def test_self_critique_offline_returns_string():
    report = ConfidenceReport(
        overall_confidence=0.35, confidence_label="low",
        signals={"signal_activation": 0.25, "preference_coherence": 0.4,
                 "catalog_coverage": 0.2},
        warnings=["Low coverage"],
        suggestion="Try broadening your preferences.",
    )
    profile = {"genre": "classical", "mood": "sad", "energy": 0.9, "likes_acoustic": True}
    results = [
        ({"title": "Hollow Rain", "genre": "folk", "mood": "sad",
          "artist": "Ember Frost", "energy": 0.30}, 0.547, "mood match"),
    ]
    critique = self_critique_offline(profile, results, report)
    assert isinstance(critique, str)
    assert len(critique) > 20
    assert "Hollow Rain" in critique


def test_self_critique_offline_mentions_conflicts():
    report = ConfidenceReport(
        overall_confidence=0.30, confidence_label="low",
        signals={"signal_activation": 0.25, "preference_coherence": 0.3,
                 "catalog_coverage": 0.1},
        warnings=["Preferences conflict"],
        suggestion="Adjust energy.",
    )
    profile = {"genre": "classical", "mood": "sad", "energy": 0.9}
    results = [
        ({"title": "Test", "genre": "classical", "mood": "peaceful",
          "artist": "A", "energy": 0.2}, 0.45, "genre match"),
    ]
    critique = self_critique_offline(profile, results, report)
    assert "conflict" in critique.lower()
