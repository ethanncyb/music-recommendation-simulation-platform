"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import (
    load_songs, recommend_songs, score_song,
    DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED, RankingStrategy,
)
from tabulate import tabulate
import subprocess
import sys


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "likes_acoustic": False,
        "min_popularity": 70,                          # only boost popular tracks
        "preferred_tags": ["upbeat", "energetic", "bright", "happy"],
    },
    "Chill Lofi": {
        "genre": "lofi",   # dataset uses "lofi", not "lo-fi"
        "mood": "chill",   # dataset uses "chill", not "calm"
        "energy": 0.2,
        "likes_acoustic": True,
        "preferred_tags": ["chill", "focused", "ambient", "peaceful"],
        "preferred_decade": 2020,                      # prefers recent releases
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "angry",
        "energy": 0.95,
        "likes_acoustic": False,
        "preferred_decade": 1990,                      # classic rock era
        "preferred_tags": ["aggressive", "raw", "rebellious", "heavy", "intense"],
    },
    # Edge-case / adversarial: conflicting preferences
    "Conflicted Listener": {
        "genre": "classical",
        "mood": "sad",
        "energy": 0.9,   # high energy but sad mood — intentionally contradictory
        "likes_acoustic": True,
        "preferred_tags": ["melancholy", "introspective", "elegant", "sad"],
        "preferred_decade": 2010,
    },
}

# Weights mirror DEFAULT strategy — used by explain_top_song() for the signal breakdown
WEIGHTS = {
    "genre":   DEFAULT.genre_weight,
    "mood":    DEFAULT.mood_weight,
    "energy":  DEFAULT.energy_weight,
    "acoustic": DEFAULT.acoustic_weight,
}

# Per-profile strategy assignments — change any value to switch that profile's ranking mode
PROFILE_STRATEGIES = {
    "High-Energy Pop":    DEFAULT,
    "Chill Lofi":         MOOD_FIRST,
    "Deep Intense Rock":  ENERGY_FOCUSED,
    "Conflicted Listener": GENRE_FIRST,
}


def explain_top_song(user_prefs: dict, song: dict) -> None:
    """Prints a signal-by-signal score breakdown for why a song ranked #1."""
    genre_match = song["genre"] == user_prefs["genre"]
    mood_match  = song["mood"]  == user_prefs["mood"]
    energy_score = 1.0 - abs(song["energy"] - user_prefs["energy"])
    if user_prefs.get("likes_acoustic"):
        acoustic_score = song["acousticness"]
    else:
        acoustic_score = 1.0 - song["acousticness"]

    total = (WEIGHTS["genre"]   * (1.0 if genre_match else 0.0)
           + WEIGHTS["mood"]    * (1.0 if mood_match  else 0.0)
           + WEIGHTS["energy"]  * energy_score
           + WEIGHTS["acoustic"] * acoustic_score)

    print(f"  >> Why '{song['title']}' ranked #1:")
    print(f"     Genre match  ({WEIGHTS['genre']:.0%} weight): {'YES' if genre_match else 'NO ':3}  "
          f"({song['genre']} == {user_prefs['genre']}?)  → {WEIGHTS['genre'] * (1.0 if genre_match else 0.0):.3f}")
    print(f"     Mood match   ({WEIGHTS['mood']:.0%} weight): {'YES' if mood_match  else 'NO ':3}  "
          f"({song['mood']} == {user_prefs['mood']}?)  → {WEIGHTS['mood'] * (1.0 if mood_match else 0.0):.3f}")
    print(f"     Energy prox  ({WEIGHTS['energy']:.0%} weight): {energy_score:.2f}  "
          f"(1 - |{song['energy']:.2f} - {user_prefs['energy']:.2f}|)  → {WEIGHTS['energy'] * energy_score:.3f}")
    print(f"     Acoustic fit ({WEIGHTS['acoustic']:.0%} weight): {acoustic_score:.2f}  "
          f"(acousticness={song['acousticness']:.2f})  → {WEIGHTS['acoustic'] * acoustic_score:.3f}")
    print(f"     {'─'*40}")
    print(f"     Total score: {total:.4f}")
    print()


def compare_strategies(user_prefs: dict, songs: list, profile_name: str) -> None:
    """Runs the same profile through every strategy and shows the #1 winner for each."""
    print(f"\n{'#'*60}")
    print(f"  Strategy Comparison — Profile: {profile_name}")
    print(f"{'#'*60}\n")
    rows = []
    for strat in [DEFAULT, GENRE_FIRST, MOOD_FIRST, ENERGY_FOCUSED]:
        results = recommend_songs(user_prefs, songs, k=1, strategy=strat)
        if results:
            song, score, _ = results[0]
            rows.append([strat.name, song["title"], song["artist"], song["genre"], f"{score:.4f}"])
    print(tabulate(
        rows,
        headers=["Strategy", "#1 Title", "Artist", "Genre", "Score"],
        tablefmt="rounded_outline",
    ))
    print()


def profile_to_dna(user_prefs: dict) -> dict:
    """Map a fast-mode preferences dict onto an EchoSphere DNA profile.

    Fast-mode profiles express taste as a small set of categorical fields plus
    a target energy; the agentic pipeline needs a full audio-feature vector.
    This helper fills in the audio-feature targets heuristically so the same
    profile dict can drive either pipeline.
    """
    from .echosphere.state import DEFAULT_DNA_PROFILE

    dna = dict(DEFAULT_DNA_PROFILE)
    energy = float(user_prefs.get("energy", dna["energy"]))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    dna["genre"] = user_prefs.get("genre")
    dna["mood"] = user_prefs.get("mood")
    dna["energy"] = energy
    dna["likes_acoustic"] = likes_acoustic
    dna["tempo_bpm"] = 70.0 + energy * 80.0  # 70 BPM at 0 energy -> 150 at 1.0
    dna["valence"] = max(0.0, min(1.0, 0.3 + energy * 0.6))
    dna["danceability"] = max(0.0, min(1.0, 0.4 + energy * 0.5))
    dna["acousticness"] = 0.75 if likes_acoustic else max(0.0, 0.35 - energy * 0.3)
    dna["instrumentalness"] = 0.6 if likes_acoustic else 0.2
    dna["speechiness"] = 0.08
    dna["top_k"] = 5
    return dna


def run_profile_agentic(name: str, user_prefs: dict, songs: list) -> None:
    """EchoSphere-RAG path: LangGraph + ChromaDB + ChatOllama."""
    from .echosphere import run_echosphere

    dna = profile_to_dna(user_prefs)
    user_request = (
        f"Recommend tracks for profile '{name}': genre={dna['genre']}, "
        f"mood={dna['mood']}, energy={dna['energy']}, "
        f"likes_acoustic={dna['likes_acoustic']}."
    )

    print(f"\n{'='*60}")
    print(f"  Profile: {name}  [Mode: agentic — EchoSphere-RAG]")
    print(f"  genre={dna['genre']} | mood={dna['mood']} | "
          f"energy={dna['energy']} | likes_acoustic={dna['likes_acoustic']}")
    print("=" * 60)

    state = run_echosphere(user_request, dna)

    if state.get("error"):
        print(f"\n  [Pipeline error] {state['error']}\n")

    retrieved = state.get("retrieved_tracks") or []
    explanations = state.get("explanations") or []
    trivia = state.get("artist_trivia") or {}

    print(f"\nTop {len(retrieved)} Recommendations:\n")
    table_rows = []
    for rank, track in enumerate(retrieved, start=1):
        reason = explanations[rank - 1] if rank - 1 < len(explanations) else ""
        table_rows.append([
            rank,
            track.get("title", "?"),
            track.get("artist", "?"),
            track.get("genre", "?"),
            f"{(track.get('distance') or 0):.4f}",
            reason,
        ])
    if table_rows:
        print(tabulate(
            table_rows,
            headers=["#", "Title", "Artist", "Genre", "Distance", "Why"],
            tablefmt="rounded_outline",
            colalign=("center", "left", "left", "left", "center", "left"),
        ))
    else:
        print("  (no tracks retrieved — is the ChromaDB seeded? "
              "Run: python -m src.echosphere.vector_store)\n")
    print()

    if trivia:
        print("  Artist trivia:")
        for artist, fact in trivia.items():
            print(f"   - {artist}: {fact}")
        print()

    # Confidence scoring works for agentic output too (see ConfidenceScorer).
    from .confidence import ConfidenceScorer
    from .guardrails import apply_guardrails, format_confidence_badge

    scorer = ConfidenceScorer(songs)
    confidence = scorer.compute(user_prefs, state)
    guardrail = apply_guardrails(confidence, retrieved)
    print(f"  Confidence: {format_confidence_badge(confidence)}  "
          f"({confidence.overall_confidence:.2f})")
    if guardrail["guardrail_message"]:
        print(f"  >> {guardrail['guardrail_message']}")
    print()


def run_profile(name: str, user_prefs: dict, songs: list,
                strategy: RankingStrategy = DEFAULT, mode: str = "fast") -> None:
    if mode == "agentic":
        run_profile_agentic(name, user_prefs, songs)
        return

    print(f"\n{'='*60}")
    print(f"  Profile: {name}  [Strategy: {strategy.name}]")
    print(f"  genre={user_prefs['genre']} | mood={user_prefs['mood']} | "
          f"energy={user_prefs['energy']} | likes_acoustic={user_prefs['likes_acoustic']}")
    print("=" * 60)

    recommendations = recommend_songs(user_prefs, songs, k=5, strategy=strategy)

    print(f"\nTop {len(recommendations)} Recommendations:\n")
    table_rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        reasons_formatted = "\n".join(f"• {r}" for r in explanation.split("; "))
        table_rows.append([
            rank,
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.4f}",
            reasons_formatted,
        ])
    print(tabulate(
        table_rows,
        headers=["#", "Title", "Artist", "Genre", "Score", "Reasons"],
        tablefmt="rounded_outline",
        colalign=("center", "left", "left", "left", "center", "left"),
    ))
    print()

    # Step 2: confidence scoring
    from .confidence import ConfidenceScorer
    from .guardrails import apply_guardrails, format_confidence_badge
    from .self_critique import self_critique_offline

    scorer = ConfidenceScorer(songs)
    confidence = scorer.compute(user_prefs, recommendations)
    guardrail = apply_guardrails(confidence, recommendations)

    print(f"  Confidence: {format_confidence_badge(confidence)}  "
          f"({confidence.overall_confidence:.2f})")
    if guardrail["guardrail_message"]:
        print(f"  >> {guardrail['guardrail_message']}")
    if guardrail["show_self_critique"]:
        critique = self_critique_offline(user_prefs, recommendations, confidence)
        print(f"\n  Self-Critique:\n  {critique}")
    print()

    # Step 3: explain why the #1 song ranked first
    if recommendations:
        explain_top_song(user_prefs, recommendations[0][0])


def run_interactive(songs: list, provider: str = "ollama", model: str = None) -> None:
    """Launch the conversational agent in interactive mode."""
    from .llm_provider import get_provider
    from .rag import load_knowledge
    from .agent import AgentLoop
    from .guardrails import format_confidence_badge

    # Default model per provider
    if model is None:
        model = "llama3.1:8b" if provider == "ollama" else "claude-sonnet-4-20250514"

    print(f"\nGrooveGenius 2.0 — Interactive Mode")
    print(f"Using: {provider}/{model}")
    print(f"Type your request, or 'quit' to exit.\n")

    try:
        llm = get_provider(provider, model=model)
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Install Ollama (https://ollama.com) and run: ollama pull llama3.1:8b")
        return

    knowledge = load_knowledge()
    agent = AgentLoop(llm=llm, songs=songs, knowledge=knowledge)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = agent.chat(user_input)

        # Show reasoning trace
        print()
        for step in result.get("reasoning_trace", []):
            print(f"  [{step}]")

        # Show results table
        recommendations = result.get("results", [])
        if recommendations:
            print(f"\nTop {len(recommendations)} Recommendations:\n")
            table_rows = []
            for rank, (song, score, explanation) in enumerate(recommendations, start=1):
                reasons_short = "; ".join(explanation.split("; ")[:3])
                table_rows.append([
                    rank, song["title"], song["artist"],
                    song["genre"], f"{score:.4f}", reasons_short,
                ])
            print(tabulate(
                table_rows,
                headers=["#", "Title", "Artist", "Genre", "Score", "Reasons"],
                tablefmt="rounded_outline",
                colalign=("center", "left", "left", "left", "center", "left"),
            ))

        # Show confidence
        conf = result.get("confidence", {})
        from .confidence import ConfidenceReport
        report = ConfidenceReport(
            overall_confidence=conf.get("score", 0),
            confidence_label=conf.get("label", "?"),
            signals=conf.get("signals", {}),
            warnings=conf.get("warnings", []),
        )
        print(f"\n  {format_confidence_badge(report)} ({report.overall_confidence:.2f})")

        if result.get("guardrail"):
            print(f"  >> {result['guardrail']}")
        if result.get("critique"):
            print(f"\n  Self-Critique: {result['critique']}")
        print()

    # Save session log
    path = agent.save_session()
    print(f"Session saved to {path}")


def run_audit(songs: list) -> None:
    """Run the bias auditor and print/save the report."""
    from .bias_auditor import BiasAuditor
    auditor = BiasAuditor(songs, strategy=DEFAULT)
    report = auditor.run_audit()
    auditor.print_report(report)
    auditor.save_report(report)


def run_batch(songs: list, mode: str = "fast") -> None:
    """Run the default batch profile demo.

    ``mode='fast'`` uses the deterministic weighted scorer (original GrooveGenius
    pipeline). ``mode='agentic'`` routes every profile through the new
    EchoSphere-RAG LangGraph pipeline (requires Ollama + a seeded ChromaDB).
    """
    for name, prefs in PROFILES.items():
        run_profile(name, prefs, songs, PROFILE_STRATEGIES.get(name, DEFAULT), mode=mode)
    if mode == "fast":
        compare_strategies(PROFILES["Conflicted Listener"], songs, "Conflicted Listener")


def run_tests() -> None:
    """Run project tests in-process via current Python executable."""
    print("\nRunning test suite with pytest...\n")
    result = subprocess.run([sys.executable, "-m", "pytest"])
    if result.returncode == 0:
        print("\nTests completed successfully.")
    else:
        print(f"\nTests finished with failures (exit code {result.returncode}).")


def run_menu(songs: list) -> None:
    """Default interactive menu for local usage."""
    while True:
        print("\nGrooveGenius 2.0")
        print("1) Run profile demo (batch, fast mode)")
        print("2) Run profile demo (batch, agentic EchoSphere-RAG)")
        print("3) Run bias audit")
        print("4) Start chatbot mode")
        print("5) Run tests")
        print("6) Exit")

        choice = input("\nSelect an option [1-6]: ").strip()

        if choice == "1":
            run_batch(songs, mode="fast")
        elif choice == "2":
            run_batch(songs, mode="agentic")
        elif choice == "3":
            run_audit(songs)
        elif choice == "4":
            provider = input("Provider [ollama/anthropic] (default: ollama): ").strip().lower() or "ollama"
            if provider not in {"ollama", "anthropic"}:
                print("Invalid provider. Use 'ollama' or 'anthropic'.")
                continue
            model = input("Model override (press Enter for default): ").strip() or None
            run_interactive(songs, provider=provider, model=model)
        elif choice == "5":
            run_tests()
        elif choice == "6":
            print("Goodbye.")
            break
        else:
            print("Invalid selection. Please choose 1-6.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GrooveGenius Music Recommender")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch recommendations directly (no menu)")
    parser.add_argument("--audit", action="store_true",
                        help="Run bias auditor instead of batch recommendations")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch conversational agent (requires Ollama or --provider anthropic)")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"],
                        help="LLM provider for interactive mode (default: ollama)")
    parser.add_argument("--model", default=None,
                        help="Model name override for the LLM provider")
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "agentic"],
        help="Pipeline mode: 'fast' (deterministic weighted scoring) or "
             "'agentic' (EchoSphere-RAG LangGraph + ChromaDB + Ollama).",
    )
    args = parser.parse_args()

    songs = load_songs("data/songs.csv")

    if args.interactive:
        run_interactive(songs, provider=args.provider, model=args.model)
    elif args.audit:
        run_audit(songs)
    elif args.batch:
        run_batch(songs, mode=args.mode)
    else:
        # In non-interactive environments (e.g., CI), keep deterministic batch behavior.
        if sys.stdin.isatty():
            run_menu(songs)
        else:
            run_batch(songs, mode=args.mode)


if __name__ == "__main__":
    main()
