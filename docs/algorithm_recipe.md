# Algorithm Recipe: Content-Based Scoring for the Music Recommender

**Scope:** Design document for `score_song()` and `Recommender.recommend()` in `src/recommender.py`  
**Goal:** Produce a single [0, 1] score per song that measures how well it fits a specific user's taste profile.

---

## 1. Overview

This recipe describes a **content-based scoring algorithm**. It answers: *given what I know about a user's taste, how well does this specific song fit?*

The algorithm computes a score in [0, 1] for every song in the catalog by comparing the song's features to the user's stated preferences. Songs are then ranked by score and the top-k are returned as recommendations.

Content-based filtering works entirely from the song's own properties — genre, mood, energy, acousticness — without needing any data from other users. This makes it:
- **Transparent** — you can always explain *why* a song was ranked highly
- **Cold-start-safe** — works on day one with a brand-new user
- **Interpretable** — each scoring dimension maps directly to a user preference field

---

## 2. Signal Decomposition

The score is built from **four independent signals**, each producing a value in [0, 1]. Each signal maps directly to one field in `UserProfile` (OOP) or the `user_prefs` dict (functional).

| Signal | User field | Song field | Type |
|--------|------------|------------|------|
| `genre_score` | `favorite_genre` / `"genre"` | `genre` | Binary |
| `mood_score` | `favorite_mood` / `"mood"` | `mood` | Binary |
| `energy_score` | `target_energy` / `"energy"` | `energy` | **Proximity** |
| `acoustic_score` | `likes_acoustic` | `acousticness` | Boolean → continuous |

---

## 3. Signal Formulas

### Signal 1: Genre Match (Binary)

Genre is the strongest single predictor of taste. It is a nominal category (no meaningful numeric distance between "pop" and "ambient"), so the match is binary.

```
genre_score = 1.0   if  song.genre == user.genre
genre_score = 0.0   otherwise
```

### Signal 2: Mood Match (Binary)

Mood captures the emotional purpose of a listening session. Like genre, it is a nominal label — "happy" is not numerically "between" "chill" and "intense."

```
mood_score = 1.0   if  song.mood == user.mood
mood_score = 0.0   otherwise
```

### Signal 3: Energy Proximity (Continuous — the key math)

This is where the core insight lives. Energy is a float in [0, 1]. The user does not want "high energy" or "low energy" in the abstract — they want energy *close to their target*.

**The wrong approach:** reward higher energy always → a user who wants 0.5 would always get the most intense songs.

**The right approach:** reward closeness to the target using absolute difference:

```
energy_score = 1 - |song.energy - user.target_energy|
```

**Why this formula works:**
- Both `song.energy` and `user.target_energy` are in [0, 1], so `|delta|` is always in [0, 1]
- No clamping is needed — the result is guaranteed to be in [0, 1]
- **Perfect match** (delta = 0) → score = 1.0
- **Maximum mismatch** (delta = 1.0) → score = 0.0
- **Symmetric:** being 0.10 *above* target penalizes equally to being 0.10 *below* target
- **Partial credit:** songs that are "somewhat close" get a non-zero score — the algorithm never hard-cuts a song to zero just because it isn't a perfect match

**Worked example for `target_energy = 0.80`** (against real songs from `data/songs.csv`):

| Song | Energy | Delta | `energy_score` |
|------|--------|-------|----------------|
| Sunrise City | 0.82 | 0.02 | **0.98** |
| Night Drive Loop | 0.75 | 0.05 | **0.95** |
| Gym Hero | 0.93 | 0.13 | 0.87 |
| Midnight Coding | 0.42 | 0.38 | 0.62 |
| Spacewalk Thoughts | 0.28 | 0.52 | 0.48 |

Notice that Midnight Coding (energy 0.42) is not a zero — it earns 0.62 as partial credit. But it is clearly weaker than Sunrise City (0.98) because the gap is much larger. This graceful degradation is more realistic than a hard cutoff like "only recommend songs with energy within ±0.1 of the target."

### Signal 4: Acoustic Fit (Boolean → Continuous)

`likes_acoustic` is a boolean. `acousticness` is a float in [0, 1] (1.0 = fully acoustic/real instruments, 0.0 = fully electronic/produced). The conversion:

```
if user.likes_acoustic is True:
    acoustic_score = song.acousticness

if user.likes_acoustic is False:
    acoustic_score = 1 - song.acousticness
```

When the user *likes* acoustic music, a highly acoustic song (0.9) scores 0.9 — good. A produced song (0.1) scores 0.1 — poor fit.

When the user does *not* like acoustic music, the scale flips: `1 - 0.9 = 0.1` (bad), `1 - 0.1 = 0.9` (good). The output is always in [0, 1].

**Example for `likes_acoustic = False`:**

| Song | Acousticness | `acoustic_score` |
|------|-------------|-----------------|
| Gym Hero | 0.05 | **0.95** |
| Sunrise City | 0.18 | **0.82** |
| Night Drive Loop | 0.22 | 0.78 |
| Midnight Coding | 0.71 | 0.29 |
| Spacewalk Thoughts | 0.92 | 0.08 |

---

## 4. Weighted Combination Formula

The four signals are combined into a single score via a weighted linear sum:

```
score = (0.35 × genre_score)
      + (0.30 × mood_score)
      + (0.25 × energy_score)
      + (0.10 × acoustic_score)
```

**Weight justifications:**

| Weight | Signal | Reason |
|--------|--------|--------|
| **0.35** | genre | Dominant taste signal; users self-identify strongly by genre |
| **0.30** | mood | Session-level intent; nearly as important as genre |
| **0.25** | energy | Always produces a non-zero value (partial credit on every song), so a smaller weight prevents it from overriding the categorical signals |
| **0.10** | acoustic | A strong preference for some listeners but a secondary tiebreaker for most |

**Verification — weights sum to 1.0:**
```
0.35 + 0.30 + 0.25 + 0.10 = 1.00
```

Because all weights are non-negative and sum to 1, and each signal is in [0, 1], the total score is **guaranteed to be in [0, 1]**.

---

## 5. Worked End-to-End Example

**User:**
```
genre:          "pop"
mood:           "happy"
target_energy:  0.80
likes_acoustic: False
```

**Song A — Sunrise City** (genre: pop, mood: happy, energy: 0.82, acousticness: 0.18):

```
genre_score    = 1.0                          (pop == pop)
mood_score     = 1.0                          (happy == happy)
energy_score   = 1 - |0.82 - 0.80| = 0.98
acoustic_score = 1 - 0.18 = 0.82             (likes_acoustic=False)

score = (0.35 × 1.00) + (0.30 × 1.00) + (0.25 × 0.98) + (0.10 × 0.82)
      = 0.3500 + 0.3000 + 0.2450 + 0.0820
      = 0.9770
```

**Song B — Midnight Coding** (genre: lofi, mood: chill, energy: 0.42, acousticness: 0.71):

```
genre_score    = 0.0                          (lofi != pop)
mood_score     = 0.0                          (chill != happy)
energy_score   = 1 - |0.42 - 0.80| = 0.62
acoustic_score = 1 - 0.71 = 0.29             (likes_acoustic=False)

score = (0.35 × 0.00) + (0.30 × 0.00) + (0.25 × 0.62) + (0.10 × 0.29)
      = 0.0000 + 0.0000 + 0.1550 + 0.0290
      = 0.1840
```

**Result:** Sunrise City (0.977) ranks far above Midnight Coding (0.184). Midnight Coding is not zero — it earns partial credit from energy and acoustic signals — but the genre and mood mismatches eliminate 0.65 points of potential score.

**Full catalog ranking for this user:**

| Rank | Song | Score | Why |
|------|------|-------|-----|
| 1 | Sunrise City | 0.977 | Genre + mood + near-perfect energy + low acoustic |
| 2 | Rooftop Lights | 0.693 | Mood match (happy), close energy (0.76), low acoustic |
| 3 | Gym Hero | 0.663 | Genre match (pop), intense mood mismatch, strong energy |
| 4 | Night Drive Loop | 0.316 | Close energy only + low acoustic |
| 5 | Storm Runner | 0.313 | Decent energy proximity + low acoustic |
| 6 | Focus Flow | 0.219 | Moderate energy proximity |
| 7 | Midnight Coding | 0.184 | Partial energy + poor acoustic fit |
| 8 | Coffee Shop Stories | 0.171 | Poor energy, very acoustic |
| 9 | Library Rain | 0.158 | Poor energy, very acoustic |
| 10 | Spacewalk Thoughts | 0.128 | Worst energy match + most acoustic |

---

## 6. Reason-Generation Logic

`score_song()` returns not just a score but a list of human-readable reason strings. These are collected during scoring based on thresholds:

| Signal | Threshold | Reason string |
|--------|-----------|--------------|
| Genre | exact match | `"Matches your favorite genre: {genre}"` |
| Mood | exact match | `"Matches your favorite mood: {mood}"` |
| Energy | `energy_score >= 0.80` (delta ≤ 0.20) | `"Close energy match ({song_energy:.2f} vs your target {target:.2f})"` |
| Acoustic (likes=True) | `acoustic_score >= 0.70` (acousticness ≥ 0.70) | `"Strong acoustic character fits your preference"` |
| Acoustic (likes=False) | `acoustic_score >= 0.70` (acousticness ≤ 0.30) | `"Low acoustic level fits your preference"` |
| Fallback | no reasons collected | `"Limited match to your preferences"` |

The energy threshold of 0.20 (score ≥ 0.80) means "within 20% of the 0–1 scale" — a musically meaningful neighborhood. Songs beyond this sound noticeably different from what the user is seeking.

The fallback reason ensures the list is never empty, which satisfies `test_explain_recommendation_returns_non_empty_string`.

**Sunrise City reasons (same user as above):**
```
["Matches your favorite genre: pop",
 "Matches your favorite mood: happy",
 "Close energy match (0.82 vs your target 0.80)",
 "Low acoustic level fits your preference"]
```

**Midnight Coding reasons:**
```
["Limited match to your preferences"]
```

---

## 7. Explanation Assembly

The `List[str]` reasons is joined into a single explanation string:

```python
explanation = "; ".join(reasons)
```

**Examples:**
- Sunrise City: `"Matches your favorite genre: pop; Matches your favorite mood: happy; Close energy match (0.82 vs your target 0.80); Low acoustic level fits your preference"`
- Midnight Coding: `"Limited match to your preferences"`

The `explain_recommendation()` method should call the same `score_song()` function and join its reasons — this guarantees the explanation is always logically consistent with the score.

---

## 8. Ranking Rule (for a list of songs)

The **Scoring Rule** (Sections 3–4) answers: *how good is this one song?*  
The **Ranking Rule** answers: *given all songs, which k should we recommend?*

```
function recommend_songs(user_prefs, songs, k):
    1. scored = []
    2. For each song in songs:
           (score, reasons) = score_song(user_prefs, song)
           explanation = "; ".join(reasons)
           scored.append( (song, score, explanation) )
    3. Sort scored descending by score
    4. Return the first k entries
```

**Three properties this rule enforces:**

| Property | How |
|----------|-----|
| **Completeness** | Every song is scored — no pre-filtering that could accidentally exclude a good match |
| **Total ordering** | Descending sort produces an unambiguous rank 1 → rank N, no ties matter because floats rarely collide |
| **Top-k selection** | Slicing after sorting means the returned list is always the globally best k songs, not just the first k encountered |

**Mapping to code:**

| Rule | Functional interface | OOP interface |
|------|---------------------|---------------|
| Scoring Rule | `score_song(user_prefs, song)` → `(float, List[str])` | same, called internally |
| Ranking Rule | `recommend_songs(user_prefs, songs, k)` → `List[(song, score, explanation)]` | `Recommender.recommend(user, k)` → `List[Song]` |

---

## 9. Pseudocode for `score_song()`

```
function score_song(user_prefs, song):
    reasons = []

    # --- Signal 1: Genre ---
    if song["genre"] == user_prefs["genre"]:
        genre_score = 1.0
        reasons.append("Matches your favorite genre: " + song["genre"])
    else:
        genre_score = 0.0

    # --- Signal 2: Mood ---
    if song["mood"] == user_prefs["mood"]:
        mood_score = 1.0
        reasons.append("Matches your favorite mood: " + song["mood"])
    else:
        mood_score = 0.0

    # --- Signal 3: Energy Proximity ---
    energy_score = 1.0 - abs(song["energy"] - user_prefs["energy"])
    if energy_score >= 0.80:
        reasons.append("Close energy match ({:.2f} vs your target {:.2f})"
                       .format(song["energy"], user_prefs["energy"]))

    # --- Signal 4: Acoustic Fit ---
    if user_prefs["likes_acoustic"] is True:
        acoustic_score = song["acousticness"]
        if acoustic_score >= 0.70:
            reasons.append("Strong acoustic character fits your preference")
    else:
        acoustic_score = 1.0 - song["acousticness"]
        if acoustic_score >= 0.70:
            reasons.append("Low acoustic level fits your preference")

    # --- Fallback ---
    if len(reasons) == 0:
        reasons.append("Limited match to your preferences")

    # --- Weighted Total ---
    score = (0.35 * genre_score)
          + (0.30 * mood_score)
          + (0.25 * energy_score)
          + (0.10 * acoustic_score)

    return (score, reasons)
```

---

## 9. Connection to Real Systems

This algorithm is a simplified but structurally honest model of the content-based filtering layer inside Spotify's production recommendation pipeline. Spotify extracts audio features — including energy, valence, danceability, acousticness, and tempo — from every track using Convolutional Neural Networks (CNNs) applied to mel spectrograms, a technique pioneered by Sander Dieleman in 2014. These features play the same semantic roles as the features in this simulation: `energy` measures intensity, `valence` measures positiveness (analogous to mood), and `acousticness` measures production style. The key architectural principle this simulation captures is that production recommendation systems do not score songs on an absolute scale ("high energy is always good") — they measure *fit to a specific user's position in feature space*, which is exactly what the proximity formula `1 - |song.energy - target_energy|` formalizes. In Spotify's system this generalization goes further: rather than a hand-coded weighted sum, matrix factorization learns user and song vectors in a high-dimensional latent space, and the dot product of those vectors plays the same role as the weighted combination formula here. The BaRT bandit system then uses those scores as the "exploitation" baseline while occasionally surfacing surprises to prevent filter bubbles — a refinement not present in this simulation, but one that flows directly from the same content-based scoring foundation this recipe describes.

---

## 10. Summary Tables

**All formulas at a glance:**

| Formula | Expression | Range |
|---------|-----------|-------|
| Genre score | `1.0 if song.genre == user.genre else 0.0` | {0, 1} |
| Mood score | `1.0 if song.mood == user.mood else 0.0` | {0, 1} |
| Energy proximity | `1 - abs(song.energy - user.energy)` | [0, 1] |
| Acoustic fit (likes=True) | `song.acousticness` | [0, 1] |
| Acoustic fit (likes=False) | `1 - song.acousticness` | [0, 1] |
| **Total score** | `0.35×genre + 0.30×mood + 0.25×energy + 0.10×acoustic` | [0, 1] |

**Reason thresholds at a glance:**

| Signal | Fire reason when | Reason string |
|--------|-----------------|---------------|
| Genre | `genre_score == 1.0` | `"Matches your favorite genre: {genre}"` |
| Mood | `mood_score == 1.0` | `"Matches your favorite mood: {mood}"` |
| Energy | `energy_score >= 0.80` | `"Close energy match (X vs your target Y)"` |
| Acoustic (likes) | `acousticness >= 0.70` | `"Strong acoustic character fits your preference"` |
| Acoustic (not likes) | `acousticness <= 0.30` | `"Low acoustic level fits your preference"` |
| Fallback | `len(reasons) == 0` | `"Limited match to your preferences"` |
