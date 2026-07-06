# Dynamical metrics

Every universe in a sweep is reduced to a fixed-length **feature vector** — the
numbers the ML clusters and learns to predict. These come from
`reality_sim/metrics.py`.

## The one rule: observe, don't probe

We measure a universe by letting it **run freely from a random soup** and
watching the shape of its evolution. We do **not** perturb a cell and trace the
response — that is *probing* (the deferred signal-speed / light-cone experiment),
and it is deliberately out of scope here. Everything below is a property of the
free trajectory, computed over `steps` generations on a `size × size` torus,
averaged over `reps` random initial conditions (`measure_mean`).

## The features

`measure_run` returns these (all `float`); `FEATURE_NAMES` is the canonical list.

| feature | what it measures | range | reads as |
|---|---|---|---|
| `final_density` | live fraction at the last step | 0–1 | did matter vanish, persist, or fill everything |
| `mean_density` | mean live fraction over the 2nd half | 0–1 | steady-state amount of "matter" |
| `std_density` | its std over the 2nd half | ≥0 | is the population steady or swinging |
| `max_density` | peak live fraction ever reached | 0–1 | transient explosion size |
| `mean_activity` | **fraction of cells that change per step**, 2nd-half mean | 0–1 | the order↔chaos dial: ~0 frozen, ~0.5 boiling |
| `activity_last` | change fraction at the final step | 0–1 | still evolving vs. settled |
| `activity_early` | change fraction, 1st-half mean | 0–1 | how lively the transient was |
| `activity_drop` | `activity_early − activity_late` | −1–1 | how much it calmed down (transient decay) |
| `growth` | `ln((pop_final+1)/(pop_init+1))` | real | net expansion/collapse from the soup |
| `spatial_entropy` | Shannon entropy of 2×2 block patterns at the end | 0–4 bits | structured (low) vs. random-looking (high) |
| `period` | exact cycle length found within 16 steps, else 0 | 0–16 | 1 still life, 2 blinker, 0 = long/aperiodic |
| `alive` | is the final population > 0 | 0/1 | did the universe survive at all |

### Activity — the important one

`mean_activity` is the fraction of cells whose state differs from the previous
step (a normalized Hamming distance between consecutive grids). It is the single
best order/chaos coordinate:

- **≈ 0** — the universe froze (still lifes, crystals, or death).
- **≈ 0.5** — the universe is boiling; nearly every cell flips every step (noise).
- **small but sustained** — the fingerprint of the interesting regime: something
  is always happening, but locally, not everywhere. Conway lives here
  (`mean_activity ≈ 0.07`).

### Spatial entropy

`block_entropy` bins every 2×2 neighborhood of the final grid into one of 16
patterns and takes the Shannon entropy of that distribution (0–4 bits). A uniform
or empty field scores ~0; a structured field (Conway ash) scores low-to-moderate;
salt-and-pepper noise scores near 4. It separates *structured* complexity from
*random-looking* chaos, which raw activity alone cannot.

### Period

After the run, `_detect_period` steps the engine up to 16 more times looking for
an exact recurrence of the final state. A short period means the universe settled
into order (a still life is period 1, a blinker period 2); `0` means no short
cycle was found — a long transient, the hallmark of chaotic *or* complex
dynamics. (This mutates the engine, so it runs last, after all other features.)

## Why these, and not more

They are cheap (all computed from the population time series, consecutive-grid
diffs, and one final-state pass), family-agnostic in spirit, and together they
span the four qualitative fates a universe can meet — die, freeze, boil, or do
something interesting in between. That four-way split is exactly what the
clustering in [sweeps.md](sweeps.md) recovers.
