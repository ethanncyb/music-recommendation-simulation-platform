"""
LLM-powered tool functions for the agentic recommendation loop.

Each function wraps an LLM call with structured output parsing and validation.
All tools include fallback logic for malformed LLM responses.

Usage:
    from src.agent_tools import extract_profile, select_strategy

    profile = extract_profile("moody night drive music", llm, valid_genres, valid_moods)
    strategy = select_strategy(profile, llm)
"""

import json
from typing import Dict, List, Optional, Tuple

from .recommender import (
    RankingStrategy, DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED,
)
from .llm_provider import LLMProvider


VALID_STRATEGIES = {
    "default": DEFAULT,
    "genre_first": GENRE_FIRST,
    "genre-first": GENRE_FIRST,
    "mood_first": MOOD_FIRST,
    "mood-first": MOOD_FIRST,
    "energy_focused": ENERGY_FOCUSED,
    "energy-focused": ENERGY_FOCUSED,
}

# ── Profile Extraction ────────────────────────────────────────────────────

EXTRACT_PROFILE_PROMPT = """Extract music preferences from this user request.

User request: "{query}"

Valid genres (pick EXACTLY one): {genres}
Valid moods (pick EXACTLY one): {moods}

Return ONLY valid JSON with these fields:
{{
  "genre": "one genre from the list above",
  "mood": "one mood from the list above",
  "energy": 0.0 to 1.0 (how energetic/intense),
  "likes_acoustic": true or false,
  "preferred_tags": ["tag1", "tag2", "tag3"] (mood descriptors)
}}

Pick the genre and mood that BEST match the user's intent, even if not explicitly stated.
Energy: 0.0=very calm, 0.5=moderate, 1.0=very intense."""


def extract_profile(query: str, llm: LLMProvider,
                    valid_genres: List[str], valid_moods: List[str]) -> Dict:
    """Parse a natural language query into a structured user profile.

    Falls back to sensible defaults if LLM output is unparseable.
    """
    prompt = EXTRACT_PROFILE_PROMPT.format(
        query=query,
        genres=", ".join(valid_genres),
        moods=", ".join(valid_moods),
    )

    try:
        data = llm.generate_json(
            prompt,
            system="You are a music preference extractor. Return ONLY valid JSON.",
        )
        return _validate_profile(data, valid_genres, valid_moods)
    except Exception:
        # Fallback: best-effort extraction from keywords
        return _fallback_profile(query, valid_genres, valid_moods)


def _validate_profile(data: Dict, valid_genres: List[str],
                       valid_moods: List[str]) -> Dict:
    """Validate and normalize an LLM-extracted profile."""
    genre = str(data.get("genre", "pop")).lower().strip()
    mood = str(data.get("mood", "happy")).lower().strip()

    # Snap to valid values
    if genre not in valid_genres:
        genre = _closest_match(genre, valid_genres)
    if mood not in valid_moods:
        mood = _closest_match(mood, valid_moods)

    energy = data.get("energy", 0.5)
    try:
        energy = max(0.0, min(1.0, float(energy)))
    except (ValueError, TypeError):
        energy = 0.5

    likes_acoustic = bool(data.get("likes_acoustic", False))

    tags = data.get("preferred_tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).lower().strip() for t in tags[:5]]

    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
        "preferred_tags": tags,
    }


def _closest_match(value: str, valid: List[str]) -> str:
    """Find the closest match by substring or stripped-punctuation match."""
    value_lower = value.lower()
    value_stripped = value_lower.replace("-", "").replace(" ", "").replace("_", "")

    for v in valid:
        v_stripped = v.replace("-", "").replace(" ", "").replace("_", "")
        # Exact match after stripping (lo-fi → lofi)
        if value_stripped == v_stripped:
            return v
        # Substring containment
        if value_lower in v or v in value_lower:
            return v
    return valid[0] if valid else value


def _fallback_profile(query: str, valid_genres: List[str],
                       valid_moods: List[str]) -> Dict:
    """Keyword-based fallback when LLM is unavailable."""
    query_lower = query.lower()

    genre = "pop"
    for g in valid_genres:
        if g in query_lower:
            genre = g
            break

    mood = "happy"
    for m in valid_moods:
        if m in query_lower:
            mood = m
            break

    # Energy keywords
    energy = 0.5
    high_words = {"energetic", "intense", "pump", "workout", "party", "fast", "hype", "driving"}
    low_words = {"chill", "calm", "relax", "sleep", "study", "focus", "quiet", "soft", "peaceful"}
    if any(w in query_lower for w in high_words):
        energy = 0.8
    elif any(w in query_lower for w in low_words):
        energy = 0.25

    # Acoustic keywords
    acoustic_words = {"acoustic", "unplugged", "folk", "guitar", "piano", "instrumental"}
    likes_acoustic = any(w in query_lower for w in acoustic_words)

    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
        "preferred_tags": [],
    }


# ── Strategy Selection ────────────────────────────────────────────────────

SELECT_STRATEGY_PROMPT = """Given this user's music preferences, pick the BEST ranking strategy.

User preferences:
  Genre: {genre}
  Mood: {mood}
  Energy: {energy}
  Acoustic: {acoustic}

Available strategies:
  - "default": Balanced (genre 16%, mood 28%, energy 47%, acoustic 9%)
  - "genre_first": Genre-heavy (genre 50%, mood 25%, energy 20%, acoustic 5%)
  - "mood_first": Mood-heavy (genre 15%, mood 55%, energy 25%, acoustic 5%)
  - "energy_focused": Energy-heavy (genre 10%, mood 10%, energy 75%, acoustic 5%)

Return ONLY the strategy name as a JSON string, e.g.: {{"strategy": "default"}}

Guidelines:
- If the user cares most about a specific genre → genre_first
- If the user describes a mood/feeling/emotion → mood_first
- If the user wants a specific energy level (workout, chill, focus) → energy_focused
- If unclear or mixed → default"""


def select_strategy(profile: Dict, llm: LLMProvider) -> RankingStrategy:
    """Ask the LLM to select the best ranking strategy for a profile."""
    prompt = SELECT_STRATEGY_PROMPT.format(
        genre=profile.get("genre", "?"),
        mood=profile.get("mood", "?"),
        energy=profile.get("energy", "?"),
        acoustic=profile.get("likes_acoustic", "?"),
    )

    try:
        data = llm.generate_json(
            prompt,
            system="You are a recommendation strategy selector. Return ONLY valid JSON.",
        )
        name = str(data.get("strategy", "default")).lower().strip()
        return VALID_STRATEGIES.get(name, DEFAULT)
    except Exception:
        return DEFAULT


# ── Result Critique ───────────────────────────────────────────────────────

CRITIQUE_PROMPT = """Review these music recommendations. Did they match the user's intent?

User's original request: "{query}"
Extracted profile: genre={genre}, mood={mood}, energy={energy}, acoustic={acoustic}

Top results:
{results_text}

Return ONLY valid JSON:
{{
  "approved": true or false,
  "issues": ["issue1", "issue2"],
  "adjustments": {{"field": "new_value"}}
}}

Set approved=true if the results reasonably match the request.
Set approved=false ONLY if there are clear mismatches. Suggest adjustments to fix them.
Valid adjustment fields: genre, mood, energy (float), likes_acoustic (bool)."""


def critique_results(query: str, profile: Dict, results: List,
                     llm: LLMProvider) -> Dict:
    """Ask the LLM to review whether results match the user's intent."""
    results_text = "\n".join(
        f"  #{i}: \"{s.get('title', '?')}\" [{s.get('genre', '?')}/{s.get('mood', '?')}, "
        f"energy={s.get('energy', '?')}] — score {sc:.3f}"
        for i, (s, sc, _) in enumerate(results[:5], 1)
    )

    prompt = CRITIQUE_PROMPT.format(
        query=query,
        genre=profile.get("genre", "?"),
        mood=profile.get("mood", "?"),
        energy=profile.get("energy", "?"),
        acoustic=profile.get("likes_acoustic", "?"),
        results_text=results_text,
    )

    try:
        data = llm.generate_json(
            prompt,
            system="You are a recommendation quality reviewer. Return ONLY valid JSON.",
        )
        return {
            "approved": bool(data.get("approved", True)),
            "issues": list(data.get("issues", [])),
            "adjustments": dict(data.get("adjustments", {})),
        }
    except Exception:
        return {"approved": True, "issues": [], "adjustments": {}}


# ── Weight Adjustment ─────────────────────────────────────────────────────

ADJUST_WEIGHTS_PROMPT = """The user wants to adjust their music recommendations.

User feedback: "{feedback}"

Current strategy weights:
  genre: {genre_w}, mood: {mood_w}, energy: {energy_w}, acoustic: {acoustic_w}

Return adjusted weights as JSON. Weights MUST sum to 1.0, each between 0.05 and 0.80:
{{"genre": 0.XX, "mood": 0.XX, "energy": 0.XX, "acoustic": 0.XX}}

Interpret the feedback:
- "less electronic" / "more acoustic" → increase acoustic weight
- "more variety" / "different genres" → decrease genre weight
- "match the mood better" → increase mood weight
- "match the energy better" → increase energy weight"""


def adjust_weights(feedback: str, current: RankingStrategy,
                   llm: LLMProvider) -> RankingStrategy:
    """Adjust strategy weights based on user feedback."""
    prompt = ADJUST_WEIGHTS_PROMPT.format(
        feedback=feedback,
        genre_w=current.genre_weight,
        mood_w=current.mood_weight,
        energy_w=current.energy_weight,
        acoustic_w=current.acoustic_weight,
    )

    try:
        data = llm.generate_json(
            prompt,
            system="You are a weight adjuster. Return ONLY valid JSON with 4 weight values.",
        )
        return _validate_weights(data, current)
    except Exception:
        return _fallback_adjust(feedback, current)


def _validate_weights(data: Dict, current: RankingStrategy) -> RankingStrategy:
    """Validate and normalize LLM-adjusted weights."""
    try:
        g = max(0.05, min(0.80, float(data.get("genre", current.genre_weight))))
        m = max(0.05, min(0.80, float(data.get("mood", current.mood_weight))))
        e = max(0.05, min(0.80, float(data.get("energy", current.energy_weight))))
        a = max(0.05, min(0.80, float(data.get("acoustic", current.acoustic_weight))))
    except (ValueError, TypeError):
        return current

    # Normalize to sum to 1.0
    total = g + m + e + a
    if total == 0:
        return current

    return RankingStrategy(
        name="Adjusted",
        genre_weight=round(g / total, 3),
        mood_weight=round(m / total, 3),
        energy_weight=round(e / total, 3),
        acoustic_weight=round(a / total, 3),
    )


def _fallback_adjust(feedback: str, current: RankingStrategy) -> RankingStrategy:
    """Keyword-based weight adjustment when LLM is unavailable."""
    g, m, e, a = (current.genre_weight, current.mood_weight,
                   current.energy_weight, current.acoustic_weight)
    fb = feedback.lower()

    if "acoustic" in fb or "less electronic" in fb:
        a = min(0.30, a + 0.10)
    if "variety" in fb or "different" in fb or "diverse" in fb:
        g = max(0.05, g - 0.10)
    if "mood" in fb or "feeling" in fb or "vibe" in fb:
        m = min(0.60, m + 0.10)
    if "energy" in fb or "intense" in fb or "calm" in fb:
        e = min(0.80, e + 0.10)

    total = g + m + e + a
    return RankingStrategy(
        name="Adjusted",
        genre_weight=round(g / total, 3),
        mood_weight=round(m / total, 3),
        energy_weight=round(e / total, 3),
        acoustic_weight=round(a / total, 3),
    )
