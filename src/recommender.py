from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Advanced features (default to neutral values so existing tests keep working)
    popularity: int = 50
    release_year: int = 2000
    key_signature: str = ""
    time_signature: int = 4
    detailed_moods: str = ""

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Advanced preference fields (all optional — default to "no preference")
    min_popularity: int = 0          # 0-100; only songs >= this score are boosted
    preferred_decade: Optional[int] = None  # e.g. 1990 for "1990s"
    preferred_tags: Optional[List[str]] = None  # e.g. ["upbeat", "energetic"]

@dataclass
class RankingStrategy:
    """
    Strategy pattern: holds the four scoring weights.
    Swap this object to change how songs are ranked without touching any
    other logic — weights are injected into score_song() at call time.
    Weights should sum to 1.0 so scores stay in the [0, 1] range.
    """
    name: str
    genre_weight: float
    mood_weight: float
    energy_weight: float
    acoustic_weight: float

# Built-in strategy constants — import these in main.py to switch modes
DEFAULT        = RankingStrategy("Default",        0.16, 0.28, 0.47, 0.09)
GENRE_FIRST    = RankingStrategy("Genre-First",    0.50, 0.25, 0.20, 0.05)
MOOD_FIRST     = RankingStrategy("Mood-First",     0.15, 0.55, 0.25, 0.05)
ENERGY_FOCUSED = RankingStrategy("Energy-Focused", 0.10, 0.10, 0.75, 0.05)

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], strategy: Optional[RankingStrategy] = None,
                 knowledge: Optional[Dict] = None):
        self.songs = songs
        self.strategy = strategy if strategy is not None else DEFAULT
        self.knowledge = knowledge

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs sorted by score descending, with diversity penalty."""
        ARTIST_PENALTY = 0.30
        GENRE_PENALTY = 0.15

        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "min_popularity": user.min_popularity,
            "preferred_decade": user.preferred_decade,
            "preferred_tags": user.preferred_tags,
        }

        remaining = [(song, score_song(user_prefs, song.__dict__, self.strategy, self.knowledge)[0]) for song in self.songs]

        selected: List[Song] = []
        selected_artists: set = set()
        selected_genres: set = set()

        while len(selected) < k and remaining:
            adjusted = []
            for song, base_score in remaining:
                score = base_score
                if song.artist in selected_artists:
                    score -= ARTIST_PENALTY
                if song.genre in selected_genres:
                    score -= GENRE_PENALTY
                adjusted.append((song, score))

            adjusted.sort(key=lambda x: x[1], reverse=True)
            best_song, _ = adjusted[0]

            selected.append(best_song)
            selected_artists.add(best_song.artist)
            selected_genres.add(best_song.genre)
            remaining = [(s, sc) for s, sc in remaining if s.id != best_song.id]

        return selected

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable explanation of why the song was recommended."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "min_popularity": user.min_popularity,
            "preferred_decade": user.preferred_decade,
            "preferred_tags": user.preferred_tags,
        }
        _, reasons = score_song(user_prefs, song.__dict__, self.strategy, self.knowledge)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs from a CSV file and returns a list of dicts with typed fields."""
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":               int(row["id"]),
                "title":            row["title"],
                "artist":           row["artist"],
                "genre":            row["genre"],
                "mood":             row["mood"],
                "energy":           float(row["energy"]),
                "tempo_bpm":        float(row["tempo_bpm"]),
                "valence":          float(row["valence"]),
                "danceability":     float(row["danceability"]),
                "acousticness":     float(row["acousticness"]),
                "instrumentalness": float(row["instrumentalness"]),
                "speechiness":      float(row["speechiness"]),
                # Advanced features
                "popularity":       int(row["popularity"]),
                "release_year":     int(row["release_year"]),
                "key_signature":    row["key_signature"],
                "time_signature":   int(row["time_signature"]),
                "detailed_moods":   row["detailed_moods"],
            })
    print(f"Loaded songs: {len(songs)}")
    return songs

def score_song(user_prefs: Dict, song: Dict, strategy: Optional[RankingStrategy] = None, knowledge: Optional[Dict] = None) -> Tuple[float, List[str]]:
    """Scores a single song against user preferences; returns (score, reasons).

    When knowledge is provided, genre and mood scoring uses similarity lookups
    instead of binary matching, giving partial credit for related genres/moods.
    When knowledge is None, behavior is identical to the original binary matching.
    """
    if strategy is None:
        strategy = DEFAULT
    reasons = []

    # Signal 1: Genre match (similarity-aware when knowledge is available)
    if knowledge and "genre_similarity" in knowledge:
        genre_score = knowledge["genre_similarity"](user_prefs["genre"], song["genre"])
        if genre_score >= 1.0:
            reasons.append(f"Matches your favorite genre: {song['genre']}")
        elif genre_score >= 0.6:
            reasons.append(f"Similar genre: {song['genre']} ({genre_score:.0%} match to {user_prefs['genre']})")
        elif genre_score >= 0.3:
            reasons.append(f"Somewhat related genre: {song['genre']} ({genre_score:.0%} match to {user_prefs['genre']})")
    else:
        if song["genre"] == user_prefs["genre"]:
            genre_score = 1.0
            reasons.append(f"Matches your favorite genre: {song['genre']}")
        else:
            genre_score = 0.0

    # Signal 2: Mood match (similarity-aware when knowledge is available)
    if knowledge and "mood_similarity" in knowledge:
        mood_score = knowledge["mood_similarity"](user_prefs["mood"], song["mood"])
        if mood_score >= 1.0:
            reasons.append(f"Matches your favorite mood: {song['mood']}")
        elif mood_score >= 0.6:
            reasons.append(f"Similar mood: {song['mood']} ({mood_score:.0%} match to {user_prefs['mood']})")
        elif mood_score >= 0.3:
            reasons.append(f"Somewhat related mood: {song['mood']} ({mood_score:.0%} match to {user_prefs['mood']})")
    else:
        if song["mood"] == user_prefs["mood"]:
            mood_score = 1.0
            reasons.append(f"Matches your favorite mood: {song['mood']}")
        else:
            mood_score = 0.0

    # Signal 3: Energy proximity
    energy_score = 1.0 - abs(song["energy"] - user_prefs["energy"])
    if energy_score >= 0.80:
        reasons.append(
            f"Close energy match ({song['energy']:.2f} vs your target {user_prefs['energy']:.2f})"
        )

    # Signal 4: Acoustic fit
    if user_prefs.get("likes_acoustic", False):
        acoustic_score = song["acousticness"]
        if acoustic_score >= 0.70:
            reasons.append("Strong acoustic character fits your preference")
    else:
        acoustic_score = 1.0 - song["acousticness"]
        if acoustic_score >= 0.70:
            reasons.append("Low acoustic level fits your preference")

    # Fallback (checked before advanced signals so it fires only when base signals all miss)
    if not reasons:
        reasons.append("Limited match to your preferences")

    # ── Advanced feature bonuses ─────────────────────────────────────────────
    # These add on top of the base weighted score. Each caps at a small value
    # so they influence ranking without overriding strong genre/mood matches.

    # Signal 5: Popularity boost (always active)
    # Songs scoring >= min_popularity get a bonus proportional to popularity/100.
    popularity = song.get("popularity", 50)
    min_pop = user_prefs.get("min_popularity", 0)
    if popularity >= min_pop:
        popularity_bonus = (popularity / 100.0) * 0.08
        if popularity >= 80:
            reasons.append(f"Highly popular track ({popularity}/100)")
        elif popularity >= 65:
            reasons.append(f"Popular track ({popularity}/100)")
    else:
        popularity_bonus = 0.0  # below user's minimum threshold — no bonus

    # Signal 6: Release era match (active only when user sets preferred_decade)
    # Score drops by 0.25 per decade away from the preferred decade, floored at 0.
    era_bonus = 0.0
    preferred_decade = user_prefs.get("preferred_decade")
    if preferred_decade is not None:
        release_year = song.get("release_year", 2000)
        song_decade = (release_year // 10) * 10
        decades_off = abs(song_decade - preferred_decade) // 10
        era_score = max(0.0, 1.0 - decades_off * 0.25)
        era_bonus = era_score * 0.06
        if song_decade == preferred_decade:
            reasons.append(f"Released in your preferred era ({song_decade}s)")
        elif era_score > 0.5:
            reasons.append(f"From a nearby era ({song_decade}s ≈ {preferred_decade}s)")

    # Signal 7: Detailed mood tag overlap (active only when user sets preferred_tags)
    # Score = |intersection| / |user_tags|, rewarding songs that cover more of the user's tags.
    tag_bonus = 0.0
    preferred_tags = user_prefs.get("preferred_tags")
    if preferred_tags and song.get("detailed_moods"):
        song_tags = set(song["detailed_moods"].split("|"))
        user_tags = set(preferred_tags)
        overlap = song_tags & user_tags
        if overlap:
            tag_score = len(overlap) / max(len(user_tags), 1)
            tag_bonus = tag_score * 0.10
            reasons.append(f"Mood tags match: {', '.join(sorted(overlap))}")

    score = (strategy.genre_weight    * genre_score
           + strategy.mood_weight     * mood_score
           + strategy.energy_weight   * energy_score
           + strategy.acoustic_weight * acoustic_score
           + popularity_bonus
           + era_bonus
           + tag_bonus)

    return (score, reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, strategy: Optional[RankingStrategy] = None, knowledge: Optional[Dict] = None) -> List[Tuple[Dict, float, str]]:
    """Scores all songs and returns the top-k using greedy diversity-aware re-ranking.

    A diversity penalty is applied before each selection to de-prioritize songs
    whose artist or genre is already represented in the current results:
      - Artist repeat: -0.30
      - Genre repeat:  -0.15

    When knowledge is provided, it is forwarded to score_song() for similarity-based
    genre/mood matching instead of binary matching.
    """
    if strategy is None:
        strategy = DEFAULT
    ARTIST_PENALTY = 0.30
    GENRE_PENALTY = 0.15

    remaining = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, strategy, knowledge)
        explanation = "; ".join(reasons)
        remaining.append((song, score, explanation))

    selected: List[Tuple[Dict, float, str]] = []
    selected_artists: set = set()
    selected_genres: set = set()

    while len(selected) < k and remaining:
        adjusted = []
        for song, base_score, explanation in remaining:
            score = base_score
            penalties = []
            if song["artist"] in selected_artists:
                score -= ARTIST_PENALTY
                penalties.append(f"artist repeat \u2212{ARTIST_PENALTY}")
            if song["genre"] in selected_genres:
                score -= GENRE_PENALTY
                penalties.append(f"genre repeat \u2212{GENRE_PENALTY}")
            note = ("; Diversity penalty: " + ", ".join(penalties)) if penalties else ""
            adjusted.append((song, score, explanation + note))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        best_song, best_score, best_expl = adjusted[0]

        selected.append((best_song, best_score, best_expl))
        selected_artists.add(best_song["artist"])
        selected_genres.add(best_song["genre"])
        remaining = [(s, sc, ex) for s, sc, ex in remaining if s["id"] != best_song["id"]]

    return selected
