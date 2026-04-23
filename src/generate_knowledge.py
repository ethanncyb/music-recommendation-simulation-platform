"""
Generate genre and mood similarity knowledge graphs using an LLM.

This is a one-time script. The generated JSON files are committed to the repo
so the system works without a running LLM. Re-run to regenerate or customize.

Usage:
    python -m src.generate_knowledge
    python -m src.generate_knowledge --backend ollama --model llama3.2
    python -m src.generate_knowledge --backend anthropic
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

from .recommender import load_songs
from .llm_provider import get_provider, LLMProvider
from .env_config import load_dotenv


def extract_unique_values(songs: List[Dict], field: str) -> List[str]:
    """Extract sorted unique values for a field from the song catalog."""
    return sorted(set(song[field] for song in songs))


def build_prompt(items: List[str], item_type: str) -> str:
    """Build the LLM prompt for pairwise similarity rating."""
    items_str = ", ".join(items)
    return f"""Rate the musical similarity between every pair of these {len(items)} {item_type}s on a 0.0-1.0 scale.
Scale: 1.0 = identical, 0.8+ = very similar, 0.5 = moderately related, 0.2 = loosely related, 0.0 = unrelated.

{item_type.title()}s: {items_str}

Return ONLY valid JSON with this exact structure (no other text):
{{"similarities": {{"{items[0]}": {{"{items[0]}": 1.0, "{items[1]}": 0.X, ...}}, ...}}}}

Rules:
- Include ALL {len(items)}x{len(items)} pairs
- Self-similarity must be 1.0
- Consider instrumentation, tempo patterns, cultural context, and listener overlap"""


SYSTEM_PROMPT = "You are a music theory expert. Return ONLY valid JSON, no markdown fences, no explanations."


def validate_graph(data: dict, expected_items: List[str]) -> Dict:
    """Validate and normalize an LLM-generated similarity graph.

    Enforces: values in [0,1], self-similarity=1.0, symmetry, all items present.
    """
    raw = data.get("similarities", data)

    similarities = {}
    for item in expected_items:
        similarities[item] = {}
        for other in expected_items:
            if item == other:
                similarities[item][other] = 1.0
                continue
            # Get score from both directions and average for symmetry
            score_ab = _get_score(raw, item, other)
            score_ba = _get_score(raw, other, item)
            if score_ab is not None and score_ba is not None:
                score = (score_ab + score_ba) / 2.0
            elif score_ab is not None:
                score = score_ab
            elif score_ba is not None:
                score = score_ba
            else:
                score = 0.0
            similarities[item][other] = round(max(0.0, min(1.0, score)), 2)

    return similarities


def _get_score(raw: dict, key_a: str, key_b: str):
    """Safely extract a score from nested dict, returning None if missing."""
    if key_a in raw and isinstance(raw[key_a], dict):
        val = raw[key_a].get(key_b)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
    return None


def generate_graph(llm: LLMProvider, items: List[str], item_type: str,
                   max_retries: int = 2) -> Dict:
    """Generate a similarity graph by prompting the LLM with retries."""
    prompt = build_prompt(items, item_type)

    for attempt in range(max_retries + 1):
        try:
            print(f"  Generating {item_type} graph (attempt {attempt + 1})...")
            data = llm.generate_json(prompt, system=SYSTEM_PROMPT)
            similarities = validate_graph(data, items)
            print(f"  Validated {len(items)}x{len(items)} {item_type} similarity matrix.")
            return similarities
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to generate valid {item_type} graph after {max_retries + 1} attempts. "
                    "Check your LLM or use the pre-generated defaults in data/knowledge/."
                )


def save_graph(similarities: Dict, items: List[str], item_type: str,
               output_dir: str, backend: str, model: str):
    """Save the similarity graph as a JSON file with metadata."""
    output = {
        "meta": {
            "version": "1.0",
            "generated_by": f"{backend}/{model}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "description": f"{item_type.title()} similarity scores (0.0-1.0)",
            "item_count": len(items),
        },
        "items": items,
        "similarities": similarities,
    }
    path = os.path.join(output_dir, f"{item_type}_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate genre/mood similarity knowledge graphs using an LLM."
    )
    parser.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: llama3.2 for ollama, claude-sonnet-4-20250514 for anthropic)")
    parser.add_argument("--output-dir", default="data/knowledge",
                        help="Output directory for JSON files (default: data/knowledge)")
    parser.add_argument("--catalog-path", default="data/songs.json",
                        help="Path to songs JSON catalog (default: data/songs.json)")
    args = parser.parse_args()

    load_dotenv()
    # Set default model per backend
    if args.model is None:
        args.model = "llama3.2" if args.backend == "ollama" else "claude-sonnet-4-20250514"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading songs from {args.catalog_path}...")
    songs = load_songs(args.catalog_path)

    genres = extract_unique_values(songs, "genre")
    moods = extract_unique_values(songs, "mood")
    print(f"Found {len(genres)} unique genres and {len(moods)} unique moods.\n")

    print(f"Connecting to {args.backend} ({args.model})...")
    kwargs = {"model": args.model}
    llm = get_provider(args.backend, **kwargs)

    print("\n--- Genre Similarity Graph ---")
    genre_similarities = generate_graph(llm, genres, "genre")
    save_graph(genre_similarities, genres, "genre", args.output_dir, args.backend, args.model)

    print("\n--- Mood Similarity Graph ---")
    mood_similarities = generate_graph(llm, moods, "mood")
    save_graph(mood_similarities, moods, "mood", args.output_dir, args.backend, args.model)

    print("\nDone! Knowledge base generated successfully.")


if __name__ == "__main__":
    main()
