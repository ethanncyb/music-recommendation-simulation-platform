# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Mission

This project simulates how big-name music platforms like Spotify and TikTok predict what users will love next. You are building for a startup music platform that wants to understand the mechanics behind personalized recommendations — transforming raw song data and user "taste profiles" into ranked suggestions, with transparent explanations for every choice.

The goal is not just to make recommendations, but to make them *explainable*: every score is decomposed into signals (genre match, mood fit, energy proximity, etc.) so the system can show *why* a song was suggested, mirroring how production recommenders surface reasoning to users and product teams.

## Commands

```bash
# Run the application (all 4 user profiles + strategy comparison)
python -m src.main

# Run tests
pytest

# Run a single test
pytest tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score
```

Activate the virtual environment first if needed: `source .venv/bin/activate`

No linter is configured; the project uses manual code review.

## Architecture

This is a **content-based music recommender** that scores a 30-song catalog against a user's stated preferences and returns the top-k ranked songs with explanations.

### Data Flow

```
UserProfile (genre, mood, energy, likes_acoustic, optional: min_popularity, preferred_decade, preferred_tags)
  → load_songs() from data/songs.csv
  → score_song() for each song  [base signals + advanced bonuses → (score, reasons)]
  → recommend_songs()           [greedy diversity re-ranking]
  → Ranked table output
```

### Key Files

- **`src/recommender.py`** — All core logic: `Song`, `UserProfile`, `RankingStrategy` dataclasses; `load_songs()`, `score_song()`, `recommend_songs()` functions; and the `Recommender` OOP class used by tests.
- **`src/main.py`** — CLI entry point. Defines 4 user profiles and runs them via `run_profile()`, `explain_top_song()`, and `compare_strategies()`.
- **`data/songs.csv`** — 30 songs × 17 attributes (genre, mood, energy, acousticness, popularity, release_year, detailed_moods pipe-separated, etc.)
- **`tests/test_recommender.py`** — 2 unit tests using a 2-song fixture via the `Recommender` class.

### Scoring (`score_song`)

Returns a `(float, list[str])` tuple — score in ~[0, 1.24] and human-readable signal reasons.

**Base signals** (sum to 1.0 via `RankingStrategy` weights):
1. Genre match (binary 0/1)
2. Mood match (binary 0/1)
3. Energy proximity: `1 - |user.target_energy - song.energy|`
4. Acoustic fit: `song.acousticness` if `likes_acoustic` else `1 - song.acousticness`

**Advanced bonuses** (additive, up to ~0.24 total):
5. Popularity boost if `song.popularity >= user.min_popularity`
6. Release era bonus if song is within 5 years of `user.preferred_decade`
7. Mood tag overlap count from `user.preferred_tags` vs `song.detailed_moods`

### Weighting Strategies (`RankingStrategy`)

Four built-in constants in `src/recommender.py`:
- `DEFAULT`: genre 16%, mood 28%, energy 47%, acoustic 9%
- `GENRE_FIRST`: genre 50%, mood 25%, energy 20%, acoustic 5%
- `MOOD_FIRST`: genre 15%, mood 55%, energy 25%, acoustic 5%
- `ENERGY_FOCUSED`: genre 10%, mood 10%, energy 75%, acoustic 5%

### Diversity Re-ranking (`recommend_songs`)

Greedy loop over k iterations: score all remaining songs, apply artist repeat penalty (−0.30) and genre repeat penalty (−0.15) if already selected, pick the highest, remove it from the pool.

### OOP vs. Functional

The module exposes both:
- **Functional**: `recommend_songs(user, songs, k, strategy)` → `list[tuple[Song, float, list[str]]]`
- **OOP**: `Recommender(songs, strategy).recommend(user, k)` → `list[Song]`; `.explain_recommendation(user, song)` → `str`

Tests use the OOP interface; `main.py` uses the functional interface directly.
