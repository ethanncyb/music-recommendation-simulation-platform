"""
Evaluation metrics for recommendation quality.

All functions accept the result format from recommend_songs():
    List[Tuple[Dict, float, str]]  →  [(song_dict, score, explanation), ...]
"""

from typing import Dict, List, Tuple


# Type alias for recommendation results
Result = Tuple[Dict, float, str]


def top_k_score(results: List[Result], k: int = 1) -> float:
    """Return the score of the k-th result (1-indexed). Returns 0.0 if not enough results."""
    if k < 1 or k > len(results):
        return 0.0
    return results[k - 1][1]


def score_spread(results: List[Result]) -> float:
    """Return the difference between the highest and lowest scores in results."""
    if len(results) < 2:
        return 0.0
    scores = [score for _, score, _ in results]
    return max(scores) - min(scores)


def genre_hit_rate(results: List[Result], target_genre: str) -> float:
    """Return the fraction of results matching the target genre."""
    if not results:
        return 0.0
    hits = sum(1 for song, _, _ in results if song["genre"] == target_genre)
    return hits / len(results)


def mood_hit_rate(results: List[Result], target_mood: str) -> float:
    """Return the fraction of results matching the target mood."""
    if not results:
        return 0.0
    hits = sum(1 for song, _, _ in results if song["mood"] == target_mood)
    return hits / len(results)


def diversity_score(results: List[Result]) -> float:
    """Return the fraction of unique genres in results."""
    if not results:
        return 0.0
    genres = set(song["genre"] for song, _, _ in results)
    return len(genres) / len(results)


def artist_diversity(results: List[Result]) -> float:
    """Return the fraction of unique artists in results."""
    if not results:
        return 0.0
    artists = set(song["artist"] for song, _, _ in results)
    return len(artists) / len(results)


def catalog_coverage(all_results: List[List[Result]], total_songs: int) -> float:
    """Return the fraction of unique songs recommended across all profile runs.

    Args:
        all_results: list of result lists, one per profile
        total_songs: total number of songs in the catalog
    """
    if total_songs == 0:
        return 0.0
    seen_ids = set()
    for results in all_results:
        for song, _, _ in results:
            seen_ids.add(song["id"])
    return len(seen_ids) / total_songs
