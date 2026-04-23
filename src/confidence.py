"""
Confidence scoring for recommendation quality.

Evaluates how much the system should trust its own output by analyzing
5 signals: score magnitude, score gap, catalog coverage, preference
coherence, and signal activation.

Usage:
    from src.confidence import ConfidenceScorer

    scorer = ConfidenceScorer(songs)
    report = scorer.compute(user_prefs, results)
    print(report.confidence_label, report.overall_confidence)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


# Theoretical max score: base 1.0 + popularity 0.08 + era 0.06 + tags 0.10
THEORETICAL_MAX = 1.24

# Genre/energy contradiction heuristics (used when knowledge is unavailable)
# Genres that are typically low-energy
LOW_ENERGY_GENRES = {"classical", "ambient", "lofi", "folk", "bossa nova", "jazz"}
# Genres that are typically high-energy
HIGH_ENERGY_GENRES = {"metal", "edm", "grunge", "punk", "trap"}


@dataclass
class ConfidenceReport:
    """Result of confidence scoring."""
    overall_confidence: float          # 0.0 to 1.0
    confidence_label: str              # "high", "medium", "low", "very_low"
    signals: Dict[str, float]          # individual confidence components
    warnings: List[str]                # human-readable issues
    suggestion: Optional[str] = None   # what the user could change


class ConfidenceScorer:
    """Computes confidence scores for recommendation results."""

    # Signal weights (sum to 1.0)
    WEIGHT_SCORE_MAGNITUDE = 0.30
    WEIGHT_SCORE_GAP = 0.20
    WEIGHT_CATALOG_COVERAGE = 0.20
    WEIGHT_COHERENCE = 0.15
    WEIGHT_SIGNAL_ACTIVATION = 0.15

    def __init__(self, songs: List[Dict]):
        self.songs = songs
        self.genres = set(s["genre"] for s in songs)
        self.moods = set(s["mood"] for s in songs)
        self._genre_counts = {}
        self._mood_counts = {}
        for s in songs:
            self._genre_counts[s["genre"]] = self._genre_counts.get(s["genre"], 0) + 1
            self._mood_counts[s["mood"]] = self._mood_counts.get(s["mood"], 0) + 1

    def compute(
        self,
        user_prefs: Dict,
        results: Union[List[Tuple], Mapping[str, Any]],
        knowledge: Optional[Dict] = None,
    ) -> ConfidenceReport:
        """Compute confidence for a set of recommendation results.

        Args:
            user_prefs: user preference dict (genre, mood, energy, likes_acoustic)
            results: either the fast-mode shape ``List[Tuple[Dict, float, str]]``
                produced by ``recommend_songs()``, **or** the EchoSphere-RAG
                agentic state (a mapping with ``retrieved_tracks`` +
                ``explanations``). The agentic shape is coerced into the
                tuple form by converting Chroma distances into a
                similarity-style score (``1 - distance`` clamped to [0, 1]).
            knowledge: optional knowledge dict with genre/mood similarity functions
        """
        results = self._coerce_results(results)
        if not results:
            return ConfidenceReport(
                overall_confidence=0.0,
                confidence_label="very_low",
                signals={},
                warnings=["No results returned."],
                suggestion="Check that the song catalog is loaded.",
            )

        warnings = []
        signals = {}

        # Signal 1: Score magnitude (30%)
        top1_score = results[0][1]
        score_mag = min(top1_score / THEORETICAL_MAX, 1.0)
        signals["score_magnitude"] = round(score_mag, 3)
        if score_mag < 0.45:
            warnings.append(
                f"Top result scored only {top1_score:.3f} "
                f"({score_mag:.0%} of theoretical max {THEORETICAL_MAX})."
            )

        # Signal 2: Score gap (20%)
        if len(results) >= 2:
            gap = results[0][1] - results[1][1]
            # Normalize: a gap of 0.2+ is "very clear", 0.0 is a toss-up
            score_gap = min(gap / 0.2, 1.0)
        else:
            score_gap = 0.5  # only 1 result — neutral
        signals["score_gap"] = round(score_gap, 3)
        if score_gap < 0.15:
            warnings.append("Top results are virtually tied — no clear best match.")

        # Signal 3: Catalog coverage (20%)
        user_genre = user_prefs.get("genre", "")
        user_mood = user_prefs.get("mood", "")
        matching_songs = sum(
            1 for s in self.songs
            if s["genre"] == user_genre or s["mood"] == user_mood
        )
        coverage = matching_songs / len(self.songs) if self.songs else 0
        # Scale: 3+ matches out of 30 = good (0.1 coverage → 1.0 signal)
        coverage_signal = min(coverage * 10, 1.0)
        signals["catalog_coverage"] = round(coverage_signal, 3)
        if matching_songs <= 1:
            warnings.append(
                f"Only {matching_songs} song(s) match your genre ({user_genre}) "
                f"or mood ({user_mood})."
            )

        # Signal 4: Preference coherence (15%)
        coherence = self._check_coherence(user_prefs, knowledge)
        signals["preference_coherence"] = round(coherence, 3)
        if coherence < 0.5:
            warnings.append(
                "Your preferences may conflict — for example, "
                f"{user_genre} is typically "
                f"{'low' if user_genre in LOW_ENERGY_GENRES else 'high'}-energy, "
                f"but you requested energy={user_prefs.get('energy', '?')}."
            )

        # Signal 5: Signal activation (15%)
        top_reasons = results[0][2] if isinstance(results[0][2], str) else ""
        activation = self._count_signal_activation(top_reasons)
        signals["signal_activation"] = round(activation, 3)
        if activation < 0.5:
            warnings.append(
                "Most scoring signals did not fire for the top result — "
                "limited overlap with your preferences."
            )

        # Weighted sum
        overall = (
            self.WEIGHT_SCORE_MAGNITUDE * score_mag
            + self.WEIGHT_SCORE_GAP * score_gap
            + self.WEIGHT_CATALOG_COVERAGE * coverage_signal
            + self.WEIGHT_COHERENCE * coherence
            + self.WEIGHT_SIGNAL_ACTIVATION * activation
        )
        overall = round(min(max(overall, 0.0), 1.0), 3)

        label = self._label(overall)
        suggestion = self._make_suggestion(user_prefs, signals, warnings)

        return ConfidenceReport(
            overall_confidence=overall,
            confidence_label=label,
            signals=signals,
            warnings=warnings,
            suggestion=suggestion,
        )

    @staticmethod
    def _coerce_results(
        results: Union[List[Tuple], Mapping[str, Any], None],
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """Accept fast-mode tuples or an EchoState mapping and normalise.

        Fast-mode results (``List[Tuple[song, score, explanation]]``) pass
        through unchanged. For an agentic EchoState, each retrieved track is
        paired with a cosine-similarity score derived from the Chroma distance
        and the matching Reasoning-node explanation.
        """
        if not results:
            return []
        if isinstance(results, Mapping):
            retrieved = list(results.get("retrieved_tracks") or [])
            explanations = list(results.get("explanations") or [])
            coerced: List[Tuple[Dict[str, Any], float, str]] = []
            for idx, track in enumerate(retrieved):
                distance = track.get("distance") if isinstance(track, dict) else None
                try:
                    similarity = 1.0 - float(distance) if distance is not None else 0.5
                except (TypeError, ValueError):
                    similarity = 0.5
                similarity = max(0.0, min(1.0, similarity))
                explanation = explanations[idx] if idx < len(explanations) else ""
                coerced.append((track, similarity, explanation))
            return coerced
        # Already a list (assumed to be the tuple shape).
        return list(results)

    def _check_coherence(self, user_prefs: Dict,
                         knowledge: Optional[Dict] = None) -> float:
        """Check whether user preferences are internally consistent.

        Returns 1.0 for coherent, lower for contradictory.
        """
        genre = user_prefs.get("genre", "")
        energy = user_prefs.get("energy", 0.5)
        acoustic = user_prefs.get("likes_acoustic", False)
        score = 1.0

        # Energy vs genre contradiction
        if genre in LOW_ENERGY_GENRES and energy >= 0.8:
            score -= 0.4  # classical + high energy = contradiction
        elif genre in HIGH_ENERGY_GENRES and energy <= 0.25:
            score -= 0.3  # metal + very low energy = contradiction

        # Acoustic vs genre contradiction
        if acoustic and genre in HIGH_ENERGY_GENRES:
            score -= 0.2  # acoustic metal is rare
        if not acoustic and genre in {"folk", "classical", "bossa nova"}:
            score -= 0.15  # these genres are naturally acoustic

        return max(score, 0.0)

    def _count_signal_activation(self, reasons: str) -> float:
        """Count what fraction of the 4 base signals fired, based on reason text.

        Looks for keywords in the explanation string.
        """
        signals_fired = 0
        total_signals = 4

        if "genre" in reasons.lower():
            signals_fired += 1
        if "mood" in reasons.lower():
            signals_fired += 1
        if "energy" in reasons.lower():
            signals_fired += 1
        if "acoustic" in reasons.lower():
            signals_fired += 1

        return signals_fired / total_signals

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "medium"
        elif score >= 0.25:
            return "low"
        else:
            return "very_low"

    @staticmethod
    def _make_suggestion(user_prefs: Dict, signals: Dict,
                         warnings: List[str]) -> Optional[str]:
        """Generate a suggestion based on the weakest signals."""
        if not warnings:
            return None

        suggestions = []
        if signals.get("catalog_coverage", 1) < 0.3:
            genre = user_prefs.get("genre", "")
            suggestions.append(
                f"Try a more common genre (the catalog has limited {genre} songs)."
            )
        if signals.get("preference_coherence", 1) < 0.5:
            suggestions.append(
                "Your preferences may conflict — consider adjusting energy level "
                "to better match your chosen genre."
            )
        if signals.get("score_magnitude", 1) < 0.45:
            suggestions.append(
                "No strong matches found. Try broadening your genre or mood preference."
            )

        return " ".join(suggestions) if suggestions else None
