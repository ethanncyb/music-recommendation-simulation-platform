# Weight Shift Experiment

## Objective
Test system sensitivity by doubling energy's importance and halving genre's importance, then evaluate whether the new weights produce more accurate recommendations.

---

## Weight Configuration

| Signal   | Baseline | Shifted (normalized) | Change         |
|----------|----------|----------------------|----------------|
| Genre    | 0.35     | **0.16**             | −54% (halved)  |
| Mood     | 0.30     | **0.28**             | −7% (minor)    |
| Energy   | 0.25     | **0.47**             | +88% (doubled) |
| Acoustic | 0.10     | **0.09**             | −10% (minor)   |
| **Total**| **1.00** | **1.00**             | ✓ valid        |

**Math note:** Raw changes (genre 0.175, energy 0.50) summed to 1.075, so both were normalized by dividing by 1.075. All weights remain in [0, 1] and sum to exactly 1.00. All individual signal scores remain bounded [0, 1], so final scores stay in [0, 1]. ✓

---

## Results

### Profile 1: High-Energy Pop
`genre=pop | mood=happy | energy=0.9 | likes_acoustic=False`

| Rank | Baseline                           | Shifted                            |
|------|------------------------------------|------------------------------------|
| #1   | Sunrise City (0.9620)              | Sunrise City (0.9462)              |
| #2   | Gym Hero (0.6875)                  | **Rooftop Lights** (0.7427)        |
| #3   | Rooftop Lights (0.5800)            | **Gym Hero** (0.7014)              |
| #4   | Ddu-Du Ddu-Du (0.3430)             | Ddu-Du Ddu-Du (0.5488)             |
| #5   | Storm Runner (0.3375)              | Storm Runner (0.5463)              |

**Notable change:** Rooftop Lights jumped from #3 to #2, swapping with Gym Hero.  
Rooftop Lights (energy=0.76, mood=happy, non-pop genre) now beats Gym Hero (energy=0.93, pop genre) because the large energy weight advantage Gym Hero had is reduced, and Rooftop Lights' mood match carries more relative weight.

**More accurate?** Debatable. Gym Hero being a pop song with high energy *and* close energy match (0.93 vs 0.9) arguably deserves #2. The shift slightly demotes genre-loyal picks in favor of cross-genre mood matches.

![High-Energy Pop results with shifted weights](../pics/weightshift_profiles_1.png)

---

### Profile 2: Chill Lofi
`genre=lofi | mood=chill | energy=0.2 | likes_acoustic=True`

| Rank | Baseline                              | Shifted                               |
|------|---------------------------------------|---------------------------------------|
| #1   | Library Rain (0.9485)                 | Library Rain (0.9169)                 |
| #2   | Midnight Coding (0.9160)              | Midnight Coding (0.8705)              |
| #3   | Focus Flow (0.6280)                   | **Spacewalk Thoughts** (0.7952)       |
| #4   | Spacewalk Thoughts (0.6220)           | **Focus Flow** (0.6062)               |
| #5   | Moonlit Sonata (0.3420)               | Moonlit Sonata (0.5479)               |

**Notable change:** Spacewalk Thoughts (mood=chill, energy=0.28, non-lofi) overtook Focus Flow (genre=lofi, energy=0.40).  
With energy now dominant, Spacewalk Thoughts' energy of 0.28 (only 0.08 from target 0.2) beats Focus Flow's 0.40 (0.20 from target), and the mood match compensates for the missing genre tag.

**More accurate?** Yes — Spacewalk Thoughts has both a chill mood match AND closer energy (0.28 vs 0.40). A chill lofi listener cares deeply about low energy and feel; prioritizing energy proximity here is more meaningful than strict genre gating.

![Chill Lofi results with shifted weights](../pics/weightshift_profiles_2.png)

---

### Profile 3: Deep Intense Rock
`genre=rock | mood=angry | energy=0.95 | likes_acoustic=False`

| Rank | Baseline                              | Shifted                               |
|------|---------------------------------------|---------------------------------------|
| #1   | **Storm Runner** (rock, 0.6800)       | **Iron Curtain** (metal, 0.8346)      |
| #2   | Iron Curtain (0.6440)                 | Storm Runner (0.6922)                 |
| #3   | Smells Like Teen Spirit (0.3490)      | Smells Like Teen Spirit (0.5591)      |
| #4   | Overdrive (0.3445)                    | Overdrive (0.5526)                    |
| #5   | Gym Hero (0.3400)                     | Gym Hero (0.5461)                     |

**Notable change:** Iron Curtain (metal, angry, energy=0.95) moved from #2 to #1, displacing Storm Runner (rock, intense, energy=0.91).  
Iron Curtain has a *perfect* energy match (1.0) and an *angry* mood match — exactly what this profile wants. Storm Runner is the correct genre (rock) but has neither a mood match (intense ≠ angry) nor as tight an energy match.

**More accurate?** **Yes — this is the clearest win.** Iron Curtain is unambiguously the better recommendation: it nails both mood and energy, while Storm Runner only matches genre and approximates energy. The baseline ranked genre too heavily and promoted a song that misses on mood.

![Deep Intense Rock results with shifted weights](../pics/weightshift_profiles_3.png)

---

### Profile 4: Conflicted Listener (Adversarial)
`genre=classical | mood=sad | energy=0.9 | likes_acoustic=True`

| Rank | Baseline                              | Shifted                               |
|------|---------------------------------------|---------------------------------------|
| #1   | **Moonlit Sonata** (classical, 0.5270)| **Hollow Rain** (folk/sad, 0.5472)    |
| #2   | Hollow Rain (0.4880)                  | Storm Runner (0.4743)                 |
| #3   | Storm Runner (0.2575)                 | Ddu-Du Ddu-Du (0.4624)               |
| #4   | La Rebelion (0.2525)                  | Gym Hero (0.4604)                     |
| #5   | Rooftop Lights (0.2500)               | La Rebelion (0.4600)                  |

**Notable change:** Moonlit Sonata (classical genre match, but energy=0.22 — *far* from target 0.9) lost the top spot to Hollow Rain (folk, sad mood match, energy=0.30).

**More accurate?** This is genuinely ambiguous — the profile is *intentionally* conflicting (sad + classical + high energy). Under baseline, genre loyalty "solved" the conflict by anchoring to classical. Under shifted weights, no single song can satisfy classical + sad + high energy, so the system falls back to mood + approximate energy. Hollow Rain (sad, acoustic, folk) arguably reflects the emotional intent of the profile better. However, none of the top-5 results are a strong fit — the adversarial nature of the profile is exposed by both weight configurations.

![Conflicted Listener results with shifted weights](../pics/weightshift_profiles_4.png)

---

## Summary

| Profile            | Better with Shifted Weights? | Reason                                              |
|--------------------|------------------------------|-----------------------------------------------------|
| High-Energy Pop    | Neutral                      | Top pick unchanged; minor reorder at #2/#3          |
| Chill Lofi         | **Yes**                      | Energy-close + mood-match song promoted appropriately|
| Deep Intense Rock  | **Yes**                      | Perfect mood+energy match beats genre-only match    |
| Conflicted Listener| Ambiguous                    | Neither config handles contradictory prefs well     |

---

## Conclusion

Shifting weight toward energy (0.25 → 0.47) and away from genre (0.35 → 0.16) produced **more accurate results in 2 of 4 profiles**. The most compelling improvement was in "Deep Intense Rock," where a metal song with a perfect angry mood and exact energy match was rightly elevated over a rock song that missed on mood. This reflects a real-world insight: **genre is a weak proxy for taste when mood and energy signals are available**.

The main trade-off: the system now surfaces cross-genre songs more aggressively. A pop listener might see non-pop songs ranked higher if the energy match is strong — which could feel surprising even if technically correct.

**Recommendation:** The shifted weights are better for energy-driven listeners. A production system would learn these weights per user from listening history rather than applying one fixed configuration for everyone.
