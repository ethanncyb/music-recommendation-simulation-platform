"""
Automated bias detection and evaluation pipeline.

Generates synthetic user profiles, runs them through the recommender,
detects systematic biases, and produces structured reports.

Usage:
    from src.bias_auditor import BiasAuditor
    from src.recommender import load_songs, DEFAULT

    songs = load_songs("data/songs.json")
    auditor = BiasAuditor(songs, strategy=DEFAULT)
    report = auditor.run_audit()
    auditor.print_report(report)
    auditor.save_report(report)
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .recommender import recommend_songs, RankingStrategy, DEFAULT
from .metrics import (
    top_k_score, score_spread, genre_hit_rate, mood_hit_rate,
    diversity_score, artist_diversity, catalog_coverage,
)


@dataclass
class BiasSignature:
    """A detected pattern of systematic unfairness."""
    name: str
    severity: str          # "low", "medium", "high"
    affected_count: int    # number of profiles affected
    total_count: int       # total profiles tested
    description: str
    evidence: Dict
    suggestion: str


@dataclass
class AuditReport:
    """Full bias audit report."""
    timestamp: str
    strategy_name: str
    profiles_tested: int
    songs_in_catalog: int
    biases: List[BiasSignature]
    catalog_stats: Dict
    profile_summaries: List[Dict] = field(default_factory=list)
    # "fast" (weighted scoring) or "agentic" (EchoSphere-RAG LangGraph).
    pipeline: str = "fast"


class BiasAuditor:
    """Runs bias audits across synthetic user profiles."""

    def __init__(self, songs: List[Dict], strategy: Optional[RankingStrategy] = None,
                 knowledge: Optional[Dict] = None):
        self.songs = songs
        self.strategy = strategy or DEFAULT
        self.knowledge = knowledge
        self._compute_catalog_stats()

    def _compute_catalog_stats(self):
        """Pre-compute catalog-level statistics."""
        self.genre_counts = Counter(s["genre"] for s in self.songs)
        self.mood_counts = Counter(s["mood"] for s in self.songs)
        self.unique_genres = sorted(self.genre_counts.keys())
        self.unique_moods = sorted(self.mood_counts.keys())
        energies = [s["energy"] for s in self.songs]
        self.energy_stats = {
            "min": min(energies),
            "max": max(energies),
            "mean": sum(energies) / len(energies),
            "low_count": sum(1 for e in energies if e < 0.4),
            "mid_count": sum(1 for e in energies if 0.4 <= e < 0.7),
            "high_count": sum(1 for e in energies if e >= 0.7),
        }
        acoustic_vals = [s["acousticness"] for s in self.songs]
        self.acoustic_stats = {
            "mean": sum(acoustic_vals) / len(acoustic_vals),
            "high_count": sum(1 for a in acoustic_vals if a >= 0.5),
            "low_count": sum(1 for a in acoustic_vals if a < 0.5),
        }

    def generate_audit_profiles(self) -> List[Dict]:
        """Generate synthetic profiles covering the preference space.

        Strategy: for each genre, pick the most common mood in the catalog
        as the 'natural' pairing, then test 3 energy levels x 2 acoustic settings.
        Also add cross-genre profiles to test contradictions.
        """
        profiles = []

        # Natural profiles: each genre x 3 energy levels x 2 acoustic
        for genre in self.unique_genres:
            for energy in [0.2, 0.5, 0.8]:
                for acoustic in [True, False]:
                    profiles.append({
                        "genre": genre,
                        "mood": self._pick_natural_mood(genre),
                        "energy": energy,
                        "likes_acoustic": acoustic,
                        "label": f"{genre}/e={energy}/ac={acoustic}",
                    })

        # Contradiction profiles: high energy + typically calm genres
        calm_genres = ["classical", "ambient", "lofi", "folk", "bossa nova"]
        for genre in calm_genres:
            if genre in self.unique_genres:
                profiles.append({
                    "genre": genre,
                    "mood": "sad",
                    "energy": 0.9,
                    "likes_acoustic": True,
                    "label": f"CONTRADICTION:{genre}/sad/e=0.9/ac=True",
                })

        return profiles

    def _pick_natural_mood(self, genre: str) -> str:
        """Pick a mood that naturally pairs with this genre.

        Uses the first song of this genre in the catalog, or falls back to
        the most common mood overall.
        """
        for song in self.songs:
            if song["genre"] == genre:
                return song["mood"]
        # Fallback: most common mood
        return self.mood_counts.most_common(1)[0][0]

    def run_audit(self, k: int = 5, pipeline: str = "fast") -> AuditReport:
        """Run the full audit pipeline and return a report.

        Args:
            k: number of top recommendations to analyse per profile.
            pipeline: ``"fast"`` runs the deterministic weighted scorer;
                ``"agentic"`` routes every synthetic profile through the
                EchoSphere-RAG LangGraph pipeline and evaluates the retrieved
                tracks. Agentic mode requires Ollama + a seeded ChromaDB.
        """
        if pipeline not in {"fast", "agentic"}:
            raise ValueError(f"pipeline must be 'fast' or 'agentic', got {pipeline!r}")

        profiles = self.generate_audit_profiles()
        profile_results = []
        all_results_for_coverage = []

        for profile in profiles:
            if pipeline == "agentic":
                results = self._run_agentic_profile(profile, k=k)
            else:
                results = recommend_songs(
                    profile, self.songs, k=k,
                    strategy=self.strategy, knowledge=self.knowledge,
                )
            all_results_for_coverage.append(results)

            summary = {
                "label": profile["label"],
                "genre": profile["genre"],
                "mood": profile["mood"],
                "energy": profile["energy"],
                "likes_acoustic": profile["likes_acoustic"],
                "top1_score": top_k_score(results, 1),
                "top1_genre": results[0][0]["genre"] if results else None,
                "top1_mood": results[0][0]["mood"] if results else None,
                "score_spread": score_spread(results),
                "genre_hit_rate": genre_hit_rate(results, profile["genre"]),
                "mood_hit_rate": mood_hit_rate(results, profile["mood"]),
                "diversity": diversity_score(results),
                "artist_diversity": artist_diversity(results),
            }
            profile_results.append(summary)

        # Detect biases
        biases = []
        biases.extend(self._detect_genre_lockout())
        biases.extend(self._detect_mood_desert())
        biases.extend(self._detect_energy_skew(profile_results))
        biases.extend(self._detect_acoustic_penalty(profile_results))
        biases.extend(self._detect_diversity_failure(profile_results))
        biases.extend(self._detect_contradictions(profile_results))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        biases.sort(key=lambda b: severity_order.get(b.severity, 3))

        coverage = catalog_coverage(all_results_for_coverage, len(self.songs))

        strategy_name = (
            "echosphere-rag" if pipeline == "agentic" else self.strategy.name
        )

        return AuditReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
            profiles_tested=len(profiles),
            songs_in_catalog=len(self.songs),
            biases=biases,
            catalog_stats={
                "genre_counts": dict(self.genre_counts),
                "mood_counts": dict(self.mood_counts),
                "energy_stats": self.energy_stats,
                "acoustic_stats": self.acoustic_stats,
                "catalog_coverage": round(coverage, 3),
            },
            profile_summaries=profile_results,
            pipeline=pipeline,
        )

    def _run_agentic_profile(self, profile: Dict, k: int = 5) -> List[Tuple]:
        """Run one synthetic profile through the EchoSphere-RAG pipeline.

        Returns the fast-mode tuple shape ``List[Tuple[song_dict, score, str]]``
        so the existing detectors / metrics keep working. The score is a
        Chroma cosine-similarity (``1 - distance``) and the "reason" string is
        the Reasoning-node explanation (or an empty string on LLM failure).
        """
        from .echosphere import run_echosphere
        from .echosphere.state import DEFAULT_DNA_PROFILE

        dna = dict(DEFAULT_DNA_PROFILE)
        energy = float(profile.get("energy", dna["energy"]))
        likes_acoustic = bool(profile.get("likes_acoustic", False))
        dna.update({
            "genre": profile.get("genre"),
            "mood": profile.get("mood"),
            "energy": energy,
            "likes_acoustic": likes_acoustic,
            "tempo_bpm": 70.0 + energy * 80.0,
            "acousticness": 0.75 if likes_acoustic else max(0.0, 0.35 - energy * 0.3),
            "instrumentalness": 0.6 if likes_acoustic else 0.2,
            "top_k": k,
        })
        user_request = (
            f"Audit profile {profile.get('label', '')}: "
            f"genre={dna['genre']}, mood={dna['mood']}, energy={energy}, "
            f"likes_acoustic={likes_acoustic}"
        )
        try:
            state = run_echosphere(user_request, dna)
        except Exception as exc:
            # Degrade gracefully so one bad profile doesn't break the audit.
            state = {"retrieved_tracks": [], "explanations": [], "error": str(exc)}

        retrieved = state.get("retrieved_tracks") or []
        explanations = state.get("explanations") or []
        results: List[Tuple] = []
        for idx, track in enumerate(retrieved):
            distance = track.get("distance") if isinstance(track, dict) else None
            try:
                similarity = 1.0 - float(distance) if distance is not None else 0.5
            except (TypeError, ValueError):
                similarity = 0.5
            similarity = max(0.0, min(1.0, similarity))
            explanation = explanations[idx] if idx < len(explanations) else ""
            results.append((track, similarity, explanation))
        return results

    # ── Bias Detectors ────────────────────────────────────────────────────

    def _detect_genre_lockout(self) -> List[BiasSignature]:
        """Genres with only 1 song — top-1 is predetermined."""
        locked = {g: c for g, c in self.genre_counts.items() if c == 1}
        if not locked:
            return []
        return [BiasSignature(
            name="genre_lockout",
            severity="high",
            affected_count=len(locked),
            total_count=len(self.unique_genres),
            description=(
                f"{len(locked)} of {len(self.unique_genres)} genres have only 1 song in the catalog. "
                f"For these genres, the #1 recommendation is predetermined regardless of other preferences."
            ),
            evidence={
                "locked_genres": sorted(locked.keys()),
                "multi_song_genres": sorted(g for g, c in self.genre_counts.items() if c > 1),
            },
            suggestion="Add 3-5 songs per genre so scoring signals can meaningfully differentiate.",
        )]

    def _detect_mood_desert(self) -> List[BiasSignature]:
        """Moods appearing in 0-1 songs — mood signal is wasted."""
        sparse = {m: c for m, c in self.mood_counts.items() if c <= 1}
        if not sparse:
            return []
        return [BiasSignature(
            name="mood_desert",
            severity="medium",
            affected_count=len(sparse),
            total_count=len(self.unique_moods),
            description=(
                f"{len(sparse)} of {len(self.unique_moods)} moods appear in only 0-1 songs. "
                f"The mood signal rarely fires twice, reducing its influence on ranking."
            ),
            evidence={
                "sparse_moods": sorted(sparse.keys()),
                "moods_with_multiple": sorted(m for m, c in self.mood_counts.items() if c > 1),
            },
            suggestion="Consolidate similar moods (e.g., 'chill' and 'relaxed') or add more songs per mood.",
        )]

    def _detect_energy_skew(self, summaries: List[Dict]) -> List[BiasSignature]:
        """Low-energy profiles score significantly lower than high-energy ones."""
        low_scores = [s["top1_score"] for s in summaries
                      if s["energy"] <= 0.3 and not s["label"].startswith("CONTRADICTION")]
        high_scores = [s["top1_score"] for s in summaries
                       if s["energy"] >= 0.7 and not s["label"].startswith("CONTRADICTION")]

        if not low_scores or not high_scores:
            return []

        avg_low = sum(low_scores) / len(low_scores)
        avg_high = sum(high_scores) / len(high_scores)
        gap = avg_high - avg_low

        if gap < 0.10:
            return []

        return [BiasSignature(
            name="energy_skew",
            severity="medium",
            affected_count=len(low_scores),
            total_count=len(low_scores) + len(high_scores),
            description=(
                f"Low-energy profiles (energy<=0.3) average a top-1 score of {avg_low:.3f} "
                f"vs {avg_high:.3f} for high-energy profiles — a gap of {gap:.3f}. "
                f"The catalog skews high-energy ({self.energy_stats['high_count']}/{len(self.songs)} "
                f"songs have energy >= 0.7)."
            ),
            evidence={
                "avg_low_energy_score": round(avg_low, 3),
                "avg_high_energy_score": round(avg_high, 3),
                "gap": round(gap, 3),
                "catalog_energy_distribution": self.energy_stats,
            },
            suggestion="Add more low-energy songs to balance the catalog energy distribution.",
        )]

    def _detect_acoustic_penalty(self, summaries: List[Dict]) -> List[BiasSignature]:
        """Acoustic-preferring profiles consistently score lower."""
        acoustic_scores = [s["top1_score"] for s in summaries
                          if s["likes_acoustic"] and not s["label"].startswith("CONTRADICTION")]
        non_acoustic_scores = [s["top1_score"] for s in summaries
                               if not s["likes_acoustic"] and not s["label"].startswith("CONTRADICTION")]

        if not acoustic_scores or not non_acoustic_scores:
            return []

        avg_acoustic = sum(acoustic_scores) / len(acoustic_scores)
        avg_non = sum(non_acoustic_scores) / len(non_acoustic_scores)
        gap = avg_non - avg_acoustic

        if gap < 0.05:
            return []

        return [BiasSignature(
            name="acoustic_penalty",
            severity="medium",
            affected_count=len(acoustic_scores),
            total_count=len(acoustic_scores) + len(non_acoustic_scores),
            description=(
                f"Acoustic-preferring profiles average {avg_acoustic:.3f} vs "
                f"{avg_non:.3f} for non-acoustic — a gap of {gap:.3f}. "
                f"The catalog has {self.acoustic_stats['low_count']} low-acoustic and "
                f"{self.acoustic_stats['high_count']} high-acoustic songs."
            ),
            evidence={
                "avg_acoustic_score": round(avg_acoustic, 3),
                "avg_non_acoustic_score": round(avg_non, 3),
                "gap": round(gap, 3),
                "catalog_acoustic_stats": self.acoustic_stats,
            },
            suggestion="Balance the catalog's acoustic distribution or reduce acoustic weight in scoring.",
        )]

    def _detect_diversity_failure(self, summaries: List[Dict]) -> List[BiasSignature]:
        """Profiles where top-5 has fewer than 3 unique genres despite penalty."""
        failures = [s for s in summaries if s["diversity"] < 0.6]  # < 3/5 unique genres

        if not failures:
            return []

        return [BiasSignature(
            name="diversity_failure",
            severity="low",
            affected_count=len(failures),
            total_count=len(summaries),
            description=(
                f"{len(failures)} of {len(summaries)} profiles produced top-5 results "
                f"with fewer than 3 unique genres, despite the diversity penalty."
            ),
            evidence={
                "affected_profiles": [f["label"] for f in failures[:10]],
                "avg_diversity": round(
                    sum(s["diversity"] for s in summaries) / len(summaries), 3
                ),
            },
            suggestion="Increase the genre repeat penalty (currently -0.15) or add catalog diversity.",
        )]

    def _detect_contradictions(self, summaries: List[Dict]) -> List[BiasSignature]:
        """Profiles where the best score is below 0.50 — catalog can't satisfy."""
        weak = [s for s in summaries if s["top1_score"] < 0.50]

        if not weak:
            return []

        return [BiasSignature(
            name="contradictory_preferences",
            severity="high",
            affected_count=len(weak),
            total_count=len(summaries),
            description=(
                f"{len(weak)} profiles scored below 0.50 on their top result — "
                f"the catalog cannot adequately satisfy these preference combinations. "
                f"These may involve contradictory preferences or catalog gaps."
            ),
            evidence={
                "affected_profiles": [
                    {"label": s["label"], "top1_score": round(s["top1_score"], 3)}
                    for s in weak[:10]
                ],
            },
            suggestion=(
                "Warn users when their preferences conflict, or expand the catalog "
                "to cover more genre/mood/energy combinations."
            ),
        )]

    # ── Output ────────────────────────────────────────────────────────────

    def print_report(self, report: AuditReport):
        """Print the audit report to console using tabulate."""
        from tabulate import tabulate

        print(f"\n{'='*60}")
        print(f"  Bias Audit Report")
        print(f"  Strategy: {report.strategy_name}")
        print(f"  Profiles tested: {report.profiles_tested}")
        print(f"  Songs in catalog: {report.songs_in_catalog}")
        print(f"  Catalog coverage: {report.catalog_stats['catalog_coverage']:.1%}")
        print(f"{'='*60}\n")

        if not report.biases:
            print("  No biases detected.\n")
            return

        rows = []
        for b in report.biases:
            rows.append([
                b.name.replace("_", " ").title(),
                b.severity.upper(),
                f"{b.affected_count}/{b.total_count}",
                b.description[:80] + ("..." if len(b.description) > 80 else ""),
            ])

        print(tabulate(
            rows,
            headers=["Bias", "Severity", "Affected", "Description"],
            tablefmt="rounded_outline",
        ))
        print()

        # Print suggestions
        print("Suggestions:")
        for b in report.biases:
            print(f"  [{b.severity.upper()}] {b.name}: {b.suggestion}")
        print()

    def save_report(self, report: AuditReport, output_dir: str = "reports"):
        """Save the audit report as a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"bias_audit_{timestamp}.json")

        data = {
            "timestamp": report.timestamp,
            "strategy_name": report.strategy_name,
            "pipeline": report.pipeline,
            "profiles_tested": report.profiles_tested,
            "songs_in_catalog": report.songs_in_catalog,
            "catalog_stats": report.catalog_stats,
            "biases": [asdict(b) for b in report.biases],
            "profile_summaries": report.profile_summaries,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Report saved to {path}")
        return path
