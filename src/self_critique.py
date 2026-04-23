"""
LLM-powered self-critique of recommendation quality.

Asks the LLM to honestly assess whether the results match the user's intent,
with a template-based offline fallback when no LLM is available.

Usage:
    from src.self_critique import self_critique, self_critique_offline

    # With LLM (requires Ollama or Anthropic)
    critique = self_critique(query, profile, results, confidence, llm)

    # Without LLM (always works)
    critique = self_critique_offline(profile, results, confidence)
"""

from typing import Dict, List, Optional, Tuple

from .confidence import ConfidenceReport


CRITIQUE_PROMPT = """You are reviewing music recommendations. Be honest about limitations.

User wanted: {query}
Extracted profile: genre={genre}, mood={mood}, energy={energy}, acoustic={acoustic}

Top results:
{results_text}

Confidence score: {confidence_score} ({confidence_label})
Warnings: {warnings}

Answer these questions concisely (3-5 sentences total):
1. Do these results actually match what the user asked for?
2. Are there obvious gaps (e.g., user wants X but no result has X)?
3. What would make these recommendations better?

Be specific and actionable. Do not sugarcoat poor results."""


def self_critique(query: str, profile: Dict, results: List[Tuple],
                  confidence: ConfidenceReport, llm) -> str:
    """Ask the LLM to review the recommendation quality.

    Args:
        query: original user query (natural language)
        profile: extracted user preferences dict
        results: recommendation results from recommend_songs()
        confidence: ConfidenceReport from the scorer
        llm: LLMProvider instance

    Returns:
        LLM-generated critique string, or offline fallback on failure.
    """
    results_text = _format_results_for_prompt(results)
    warnings_text = "; ".join(confidence.warnings) if confidence.warnings else "None"

    prompt = CRITIQUE_PROMPT.format(
        query=query,
        genre=profile.get("genre", "?"),
        mood=profile.get("mood", "?"),
        energy=profile.get("energy", "?"),
        acoustic=profile.get("likes_acoustic", "?"),
        results_text=results_text,
        confidence_score=confidence.overall_confidence,
        confidence_label=confidence.confidence_label,
        warnings=warnings_text,
    )

    try:
        response = llm.generate(
            prompt,
            system="You are a music recommendation quality reviewer. Be concise and honest.",
        )
        return response.strip()
    except Exception:
        return self_critique_offline(profile, results, confidence)


def self_critique_offline(profile: Dict, results: List[Tuple],
                          confidence: ConfidenceReport) -> str:
    """Template-based critique when no LLM is available.

    Always works — no external dependencies.
    """
    parts = []

    # Score assessment
    if results:
        top_score = results[0][1]
        top_song = results[0][0]
        parts.append(
            f"Your top result \"{top_song.get('title', '?')}\" "
            f"({top_song.get('genre', '?')}) scored {top_score:.3f} — "
            f"rated {confidence.confidence_label.upper()} confidence."
        )

    # Signal activation
    activation = confidence.signals.get("signal_activation", 0)
    fired = int(activation * 4)
    parts.append(f"{fired}/4 preference signals matched for the top result.")

    # Coherence
    coherence = confidence.signals.get("preference_coherence", 1.0)
    if coherence < 0.5:
        parts.append(
            f"Your preferences appear to conflict — "
            f"{profile.get('genre', '?')} is not typically associated with "
            f"energy={profile.get('energy', '?')}."
        )

    # Coverage
    coverage = confidence.signals.get("catalog_coverage", 1.0)
    if coverage < 0.3:
        parts.append(
            f"Very few songs in the catalog match your genre "
            f"({profile.get('genre', '?')}) or mood ({profile.get('mood', '?')})."
        )

    # Warnings
    if confidence.warnings:
        parts.append("Issues: " + "; ".join(confidence.warnings))

    # Suggestion
    if confidence.suggestion:
        parts.append("Suggestion: " + confidence.suggestion)

    return " ".join(parts)


def _format_results_for_prompt(results: List[Tuple], max_results: int = 5) -> str:
    """Format results into a readable string for the LLM prompt."""
    lines = []
    for i, (song, score, explanation) in enumerate(results[:max_results], 1):
        title = song.get("title", "?")
        artist = song.get("artist", "?")
        genre = song.get("genre", "?")
        mood = song.get("mood", "?")
        energy = song.get("energy", "?")
        lines.append(
            f"  #{i}: \"{title}\" by {artist} "
            f"[{genre}/{mood}, energy={energy}] — score {score:.3f}"
        )
    return "\n".join(lines)
