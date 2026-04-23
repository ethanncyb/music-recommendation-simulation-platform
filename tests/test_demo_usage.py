"""Demo-first tests for developers and users.

Purpose:
- Show common ways to use GrooveGenius APIs.
- Provide runnable examples that validate expected behavior.
- Serve as a quick onboarding file for both developers and project reviewers.

How to run:
- Run full demo suite:
    pytest tests/test_demo_usage.py

- Run with verbose output:
    pytest -v tests/test_demo_usage.py

- Run only one demo section:
    pytest -k audit tests/test_demo_usage.py
    pytest -k mcp tests/test_demo_usage.py

This file intentionally focuses on readable examples and stable assertions.
"""

from src.main import PROFILES, PROFILE_STRATEGIES
from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    recommend_songs,
    DEFAULT,
    GENRE_FIRST,
    MOOD_FIRST,
    ENERGY_FOCUSED,
)
from src.confidence import ConfidenceScorer
from src.bias_auditor import BiasAuditor
from src.mcp_server import handle_list_catalog, handle_recommend_manual, handle_explain_song


def _scores_are_desc_sorted(results):
    scores = [score for _, score, _ in results]
    return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_demo_load_catalog():
    """Demo: load song catalog from CSV."""
    songs = load_songs("data/songs.csv")
    assert len(songs) == 30
    assert {"title", "genre", "mood", "energy"}.issubset(songs[0].keys())


def test_demo_recommendation_cases_from_profiles():
    """Demo: run the four built-in profile scenarios from main.py."""
    songs = load_songs("data/songs.csv")

    for name, prefs in PROFILES.items():
        strategy = PROFILE_STRATEGIES.get(name, DEFAULT)
        results = recommend_songs(prefs, songs, k=5, strategy=strategy)

        assert len(results) == 5
        assert _scores_are_desc_sorted(results)

        top_song, top_score, top_explanation = results[0]
        assert isinstance(top_song["title"], str) and top_song["title"]
        assert isinstance(top_score, float)
        assert isinstance(top_explanation, str) and top_explanation.strip() != ""


def test_demo_strategy_comparison_changes_winner_for_edge_profile():
    """Demo: compare ranking strategies for the edge-case user profile."""
    songs = load_songs("data/songs.csv")
    user = PROFILES["Conflicted Listener"]

    top_titles = {}
    for strategy in [DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED]:
        result = recommend_songs(user, songs, k=1, strategy=strategy)
        top_titles[strategy.name] = result[0][0]["title"]

    # At least two strategies should produce different top choices.
    assert len(set(top_titles.values())) >= 2


def test_demo_oop_recommender_and_explanation_usage():
    """Demo: object-oriented API usage (Recommender + explanation)."""
    songs = [
        Song(
            id=1,
            title="Sample Pop",
            artist="Demo Artist",
            genre="pop",
            mood="happy",
            energy=0.85,
            tempo_bpm=122,
            valence=0.9,
            danceability=0.8,
            acousticness=0.15,
        ),
        Song(
            id=2,
            title="Sample Lofi",
            artist="Demo Artist",
            genre="lofi",
            mood="chill",
            energy=0.3,
            tempo_bpm=80,
            valence=0.5,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]

    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )

    rec = Recommender(songs)
    top = rec.recommend(user, k=2)
    explanation = rec.explain_recommendation(user, top[0])

    assert len(top) == 2
    assert top[0].title == "Sample Pop"
    assert isinstance(explanation, str) and explanation.strip() != ""


def test_demo_confidence_easy_vs_hard_profile():
    """Demo: confidence scoring for easy and hard preference combinations."""
    songs = load_songs("data/songs.csv")
    scorer = ConfidenceScorer(songs)

    easy = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "likes_acoustic": False,
    }
    hard = {
        "genre": "classical",
        "mood": "sad",
        "energy": 0.9,
        "likes_acoustic": True,
    }

    easy_results = recommend_songs(easy, songs, k=5)
    hard_results = recommend_songs(hard, songs, k=5)

    easy_conf = scorer.compute(easy, easy_results)
    hard_conf = scorer.compute(hard, hard_results)

    assert easy_conf.overall_confidence >= hard_conf.overall_confidence
    assert easy_conf.confidence_label in {"high", "medium"}


def test_demo_bias_audit_core_signals():
    """Demo: run bias audit and verify core catalog-level findings are reported."""
    songs = load_songs("data/songs.csv")
    auditor = BiasAuditor(songs)
    report = auditor.run_audit()

    names = {b.name for b in report.biases}
    assert "genre_lockout" in names
    assert "mood_desert" in names
    assert "energy_skew" in names


def test_demo_mcp_offline_handlers():
    """Demo: call MCP handlers directly without running an MCP client."""
    catalog = handle_list_catalog({})
    assert len(catalog) == 30

    manual = handle_recommend_manual(
        {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "likes_acoustic": False,
            "k": 3,
            "strategy": "default",
        }
    )
    assert len(manual["recommendations"]) == 3
    assert "confidence" in manual

    explain = handle_explain_song(
        {
            "song_title": "Sunrise City",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "likes_acoustic": False,
        }
    )
    assert "error" not in explain
    assert explain["song"] == "Sunrise City"
