"""Rule-space for the life-like (2-state outer-totalistic) family.

This is the search space the ML sweep explores. A life-like law is fully
described by a birth set B and a survival set S, each a subset of the neighbor
counts {0..8}. That is exactly **18 bits** (B0..B8, S0..S8) — a clean, finite,
enumerable universe of universes (2**18 = 262,144 of them), which is what makes
"search rule-space with a model" well-posed.

Everything here converts between the three representations we need:
  * bit-vector  (np.uint8[18]) — the ML feature/label space,
  * (birth, survival) lists   — human-facing,
  * a :class:`~reality_sim.lawset.LawSet` — runnable by the engine.
"""

from __future__ import annotations

import numpy as np

from .lawset import LawSet

N_COUNTS = 9  # neighbor counts 0..8 for a Moore neighborhood
N_BITS = 2 * N_COUNTS  # 18: B0..B8 then S0..S8
BIT_LABELS: list[str] = [f"B{i}" for i in range(N_COUNTS)] + [f"S{i}" for i in range(N_COUNTS)]

# Neutral 2-state palette + a max-entropy 50% soup as the standard proving ground.
_SWEEP_PALETTE = ["#0b0f1a", "#e8f0ff"]
_SWEEP_SEED = {"kind": "random", "density": 0.5}


def bs_to_bits(birth, survival) -> np.ndarray:
    bits = np.zeros(N_BITS, dtype=np.uint8)
    for b in birth:
        bits[int(b)] = 1
    for s in survival:
        bits[N_COUNTS + int(s)] = 1
    return bits


def bits_to_bs(bits) -> tuple[list[int], list[int]]:
    bits = np.asarray(bits)
    birth = [i for i in range(N_COUNTS) if bits[i]]
    survival = [i for i in range(N_COUNTS) if bits[N_COUNTS + i]]
    return birth, survival


def rule_name(birth, survival) -> str:
    """Canonical life-like name, e.g. B3/S23. Empty sets render as just 'B'/'S'."""
    return "B" + "".join(str(b) for b in sorted(birth)) + "/S" + "".join(str(s) for s in sorted(survival))


def bits_to_lawset(bits, lid: str = "rule", name: str | None = None) -> LawSet:
    birth, survival = bits_to_bs(bits)
    nm = name or rule_name(birth, survival)
    return LawSet(
        id=lid,
        name=nm,
        description=f"life-like {rule_name(birth, survival)}",
        family="life",
        states=2,
        params={"birth": birth, "survival": survival},
        palette=list(_SWEEP_PALETTE),
        seed=dict(_SWEEP_SEED),
    )


def random_bits(rng: np.random.Generator, p: float = 0.5, allow_b0: bool = False) -> np.ndarray:
    """A random life-like rule. B0 (birth on an empty neighborhood) makes the
    vacuum itself flip every step — a degenerate strobe that dominates surveys —
    so it is excluded by default, matching standard rule-space studies."""
    bits = (rng.random(N_BITS) < p).astype(np.uint8)
    if not allow_b0:
        bits[0] = 0
    return bits


# A spread of famous life-like rules across the behavioral classes. These are
# swept alongside the random rules and annotated on the phase map — they are the
# landmarks that tell us whether the learned map of rule-space makes sense
# (Conway should land at the edge of chaos, Seeds should be explosive, etc.).
KNOWN_RULES: dict[str, tuple[list[int], list[int]]] = {
    "Conway": ([3], [2, 3]),
    "HighLife": ([3, 6], [2, 3]),
    "Day&Night": ([3, 6, 7, 8], [3, 4, 6, 7, 8]),
    "Seeds": ([2], []),
    "LifeWithoutDeath": ([3], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    "Maze": ([3], [1, 2, 3, 4, 5]),
    "Mazectric": ([3], [1, 2, 3, 4]),
    "Replicator": ([1, 3, 5, 7], [1, 3, 5, 7]),
    "2x2": ([3, 6], [1, 2, 5]),
    "Coral": ([3], [4, 5, 6, 7, 8]),
    "Anneal": ([4, 6, 7, 8], [3, 5, 6, 7, 8]),
    "Diamoeba": ([3, 5, 6, 7, 8], [5, 6, 7, 8]),
    "Move": ([3, 6, 8], [2, 4, 5]),
    "Gnarl": ([1], [1]),
    "Assimilation": ([3, 4, 5], [4, 5, 6, 7]),
    "WalledCities": ([4, 5, 6, 7, 8], [2, 3, 4, 5]),
}


def known_bits() -> list[tuple[str, np.ndarray]]:
    return [(name, bs_to_bits(b, s)) for name, (b, s) in KNOWN_RULES.items()]
