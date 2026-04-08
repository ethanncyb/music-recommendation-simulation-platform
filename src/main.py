"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}

    print(f"\nUser Profile: genre={user_prefs['genre']} | mood={user_prefs['mood']} | "
          f"energy={user_prefs['energy']} | likes_acoustic={user_prefs['likes_acoustic']}")
    print("-" * 60)

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(f"\nTop {len(recommendations)} Recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"  {rank}. {song['title']} by {song['artist']}")
        print(f"     Score: {score:.4f}")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
        print()


if __name__ == "__main__":
    main()
