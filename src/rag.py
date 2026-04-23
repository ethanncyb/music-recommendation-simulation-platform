"""
RAG (Retrieval-Augmented Generation) module for music knowledge retrieval.

Loads pre-generated genre and mood similarity graphs and provides
lookup functions that can be injected into score_song() to replace
binary matching with similarity-based scoring.

Usage:
    from src.rag import load_knowledge

    knowledge = load_knowledge()  # loads from data/knowledge/
    score, reasons = score_song(user_prefs, song, strategy, knowledge)
"""

import json
import os
from typing import Dict, Optional


_DEFAULT_KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge"
)


class KnowledgeBase:
    """Loads and queries genre/mood similarity graphs."""

    def __init__(self, knowledge_dir: Optional[str] = None):
        kdir = knowledge_dir or _DEFAULT_KNOWLEDGE_DIR
        genre_path = os.path.join(kdir, "genre_graph.json")
        mood_path = os.path.join(kdir, "mood_graph.json")

        self.genre_graph = self._load(genre_path)
        self.mood_graph = self._load(mood_path)

    @staticmethod
    def _load(path: str) -> Dict:
        """Load a similarity graph from JSON, returning the similarities dict."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("similarities", data)

    def genre_similarity(self, genre_a: str, genre_b: str) -> float:
        """Return similarity between two genres (0.0 to 1.0).

        - Exact match always returns 1.0
        - Unknown genre returns 0.0
        """
        if genre_a == genre_b:
            return 1.0
        return self.genre_graph.get(genre_a, {}).get(genre_b, 0.0)

    def mood_similarity(self, mood_a: str, mood_b: str) -> float:
        """Return similarity between two moods (0.0 to 1.0).

        - Exact match always returns 1.0
        - Unknown mood returns 0.0
        """
        if mood_a == mood_b:
            return 1.0
        return self.mood_graph.get(mood_a, {}).get(mood_b, 0.0)


def load_knowledge(knowledge_dir: Optional[str] = None) -> Dict:
    """Load knowledge base and return a dict suitable for score_song().

    Returns:
        {"genre_similarity": callable, "mood_similarity": callable}
    """
    kb = KnowledgeBase(knowledge_dir)
    return {
        "genre_similarity": kb.genre_similarity,
        "mood_similarity": kb.mood_similarity,
    }
