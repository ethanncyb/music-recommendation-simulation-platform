"""Tests for the RAG knowledge base and its integration with score_song()."""

from src.rag import KnowledgeBase, load_knowledge
from src.recommender import Song, UserProfile, Recommender, score_song


# ── Knowledge Base unit tests ────────────────────────────────────────────────

def test_genre_self_similarity():
    kb = KnowledgeBase()
    assert kb.genre_similarity("pop", "pop") == 1.0
    assert kb.genre_similarity("rock", "rock") == 1.0
    assert kb.genre_similarity("lofi", "lofi") == 1.0


def test_mood_self_similarity():
    kb = KnowledgeBase()
    assert kb.mood_similarity("happy", "happy") == 1.0
    assert kb.mood_similarity("chill", "chill") == 1.0
    assert kb.mood_similarity("angry", "angry") == 1.0


def test_genre_symmetry():
    kb = KnowledgeBase()
    assert kb.genre_similarity("pop", "rock") == kb.genre_similarity("rock", "pop")
    assert kb.genre_similarity("lofi", "ambient") == kb.genre_similarity("ambient", "lofi")
    assert kb.genre_similarity("metal", "grunge") == kb.genre_similarity("grunge", "metal")


def test_mood_symmetry():
    kb = KnowledgeBase()
    assert kb.mood_similarity("happy", "sad") == kb.mood_similarity("sad", "happy")
    assert kb.mood_similarity("chill", "relaxed") == kb.mood_similarity("relaxed", "chill")


def test_similar_genres_score_higher_than_distant():
    kb = KnowledgeBase()
    # synth-pop should be more similar to pop than metal is
    assert kb.genre_similarity("pop", "synth-pop") > kb.genre_similarity("pop", "metal")
    # grunge should be more similar to rock than to classical
    assert kb.genre_similarity("rock", "grunge") > kb.genre_similarity("rock", "classical")
    # lofi should be more similar to ambient than to metal
    assert kb.genre_similarity("lofi", "ambient") > kb.genre_similarity("lofi", "metal")


def test_similar_moods_score_higher_than_distant():
    kb = KnowledgeBase()
    # angry should be more similar to intense than to happy
    assert kb.mood_similarity("angry", "intense") > kb.mood_similarity("angry", "happy")
    # chill should be more similar to relaxed than to angry
    assert kb.mood_similarity("chill", "relaxed") > kb.mood_similarity("chill", "angry")
    # happy should be more similar to upbeat than to sad
    assert kb.mood_similarity("happy", "upbeat") > kb.mood_similarity("happy", "sad")


def test_unknown_genre_returns_zero():
    kb = KnowledgeBase()
    assert kb.genre_similarity("pop", "nonexistent_genre") == 0.0
    assert kb.genre_similarity("nonexistent", "also_nonexistent") == 0.0


def test_unknown_mood_returns_zero():
    kb = KnowledgeBase()
    assert kb.mood_similarity("happy", "nonexistent_mood") == 0.0


# ── load_knowledge() tests ───────────────────────────────────────────────────

def test_load_knowledge_returns_callables():
    knowledge = load_knowledge()
    assert callable(knowledge["genre_similarity"])
    assert callable(knowledge["mood_similarity"])
    assert knowledge["genre_similarity"]("pop", "pop") == 1.0
    assert knowledge["mood_similarity"]("happy", "happy") == 1.0


# ── score_song integration tests ─────────────────────────────────────────────

def _make_song_dict(**overrides):
    defaults = {
        "id": 1, "title": "Test Song", "artist": "Test Artist",
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "tempo_bpm": 120, "valence": 0.9, "danceability": 0.8,
        "acousticness": 0.2, "instrumentalness": 0.02, "speechiness": 0.05,
        "popularity": 50, "release_year": 2023,
        "key_signature": "C Major", "time_signature": 4,
        "detailed_moods": "happy|upbeat|bright",
    }
    defaults.update(overrides)
    return defaults


def _make_prefs(**overrides):
    defaults = {
        "genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False,
    }
    defaults.update(overrides)
    return defaults


def test_score_song_without_knowledge_unchanged():
    """score_song with knowledge=None produces identical results to omitting it."""
    song = _make_song_dict()
    prefs = _make_prefs()
    score_with_none, reasons_with_none = score_song(prefs, song, None, None)
    score_default, reasons_default = score_song(prefs, song, None)
    assert score_with_none == score_default
    assert reasons_with_none == reasons_default


def test_score_song_with_knowledge_gives_partial_genre_credit():
    """A related genre gets partial credit instead of 0.0."""
    knowledge = {
        "genre_similarity": lambda a, b: (
            1.0 if a == b else 0.7 if {a, b} == {"pop", "synth-pop"} else 0.0
        ),
        "mood_similarity": lambda a, b: 1.0 if a == b else 0.0,
    }
    song = _make_song_dict(genre="synth-pop")
    prefs = _make_prefs(genre="pop")

    score_with_knowledge, reasons_with = score_song(prefs, song, None, knowledge)
    score_without_knowledge, reasons_without = score_song(prefs, song, None, None)

    # With knowledge: genre_score = 0.7, without: genre_score = 0.0
    assert score_with_knowledge > score_without_knowledge
    assert any("Similar genre" in r for r in reasons_with)


def test_score_song_with_knowledge_gives_partial_mood_credit():
    """A related mood gets partial credit instead of 0.0."""
    knowledge = {
        "genre_similarity": lambda a, b: 1.0 if a == b else 0.0,
        "mood_similarity": lambda a, b: (
            1.0 if a == b else 0.8 if {a, b} == {"angry", "intense"} else 0.0
        ),
    }
    song = _make_song_dict(mood="intense")
    prefs = _make_prefs(mood="angry")

    score, reasons = score_song(prefs, song, None, knowledge)
    assert any("Similar mood" in r for r in reasons)


def test_score_song_exact_match_still_works_with_knowledge():
    """Exact genre/mood match still returns 1.0 and correct reason with knowledge."""
    knowledge = load_knowledge()
    song = _make_song_dict(genre="pop", mood="happy")
    prefs = _make_prefs(genre="pop", mood="happy")

    score, reasons = score_song(prefs, song, None, knowledge)
    assert any("Matches your favorite genre" in r for r in reasons)
    assert any("Matches your favorite mood" in r for r in reasons)


# ── Recommender OOP integration test ─────────────────────────────────────────

def test_recommender_with_knowledge():
    """Recommender class works with knowledge injected."""
    songs = [
        Song(id=1, title="Pop Hit", artist="A", genre="pop", mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Synth Track", artist="B", genre="synth-pop", mood="upbeat",
             energy=0.8, tempo_bpm=125, valence=0.85, danceability=0.75, acousticness=0.15),
    ]
    knowledge = load_knowledge()
    user = UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)

    rec_with = Recommender(songs, knowledge=knowledge)
    rec_without = Recommender(songs)

    results_with = rec_with.recommend(user, k=2)
    results_without = rec_without.recommend(user, k=2)

    # Both should return Pop Hit first (exact match beats partial)
    assert results_with[0].title == "Pop Hit"
    assert results_without[0].title == "Pop Hit"

    # With knowledge, the explanation for Synth Track should mention similarity
    explanation = rec_with.explain_recommendation(user, songs[1])
    assert "Similar genre" in explanation or "Similar mood" in explanation
