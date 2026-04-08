"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs, score_song


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",   # dataset uses "lofi", not "lo-fi"
        "mood": "chill",   # dataset uses "chill", not "calm"
        "energy": 0.2,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "angry",
        "energy": 0.95,
        "likes_acoustic": False,
    },
    # Edge-case / adversarial: conflicting preferences
    "Conflicted Listener": {
        "genre": "classical",
        "mood": "sad",
        "energy": 0.9,   # high energy but sad mood — intentionally contradictory
        "likes_acoustic": True,
    },
}

WEIGHTS = {"genre": 0.16, "mood": 0.28, "energy": 0.47, "acoustic": 0.09}


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


def run_profile(name: str, user_prefs: dict, songs: list) -> None:
    print(f"\n{'='*60}")
    print(f"  Profile: {name}")
    print(f"  genre={user_prefs['genre']} | mood={user_prefs['mood']} | "
          f"energy={user_prefs['energy']} | likes_acoustic={user_prefs['likes_acoustic']}")
    print("=" * 60)

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(f"\nTop {len(recommendations)} Recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank}. {song['title']} by {song['artist']}")
        print(f"     Score: {score:.4f}")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
        print()

    # Step 2: explain why the #1 song ranked first
    if recommendations:
        explain_top_song(user_prefs, recommendations[0][0])


def main() -> None:
    songs = load_songs("data/songs.csv")
    for name, prefs in PROFILES.items():
        run_profile(name, prefs, songs)


if __name__ == "__main__":
    main()
