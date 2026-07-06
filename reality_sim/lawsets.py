"""The library of concrete universes — the actual "laws of physics" you can run.

Each entry is a :class:`~reality_sim.lawset.LawSet`, and now each also declares
its **live-tunable controls**: the knobs the viewer renders as widgets so you can
edit a universe's law while it runs (toggle Conway's birth rules, stretch the
excitable medium's refractory tail, tip the forest between smoldering and
firestorm). Adding a universe is adding an entry here.
"""

from __future__ import annotations

import colorsys
from dataclasses import replace

import numpy as np

from .lawset import LawSet

LIBRARY: dict[str, LawSet] = {}
ORDER: list[str] = []


def register(ls: LawSet) -> LawSet:
    LIBRARY[ls.id] = ls
    ORDER.append(ls.id)
    return ls


# --- control-spec builders (the UI knobs each family exposes) -----------------

def life_controls() -> list[dict]:
    return [
        {"key": "birth", "label": "birth  B", "type": "set9"},
        {"key": "survival", "label": "survival  S", "type": "set9"},
        {"key": "density", "label": "seed density", "type": "float", "min": 0.05, "max": 0.95, "step": 0.01},
    ]


def excitable_controls() -> list[dict]:
    return [
        {"key": "threshold", "label": "ignition threshold", "type": "int", "min": 1, "max": 8, "step": 1},
        {"key": "states", "label": "refractory length", "type": "int", "min": 3, "max": 32, "step": 1},
        {"key": "density", "label": "seed density", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05},
    ]


def forestfire_controls() -> list[dict]:
    return [
        {"key": "p", "label": "tree growth  p", "type": "float", "min": 0.0, "max": 0.3, "step": 0.002},
        {"key": "f", "label": "lightning  f", "type": "float", "min": 0.0, "max": 0.02, "step": 0.0002},
        {"key": "density", "label": "initial forest", "type": "float", "min": 0.0, "max": 1.0, "step": 0.02},
    ]


def lenia_controls() -> list[dict]:
    return [
        {"key": "mu", "label": "growth center  μ", "type": "float", "min": 0.0, "max": 0.5, "step": 0.005},
        {"key": "sigma", "label": "growth width  σ", "type": "float", "min": 0.005, "max": 0.08, "step": 0.001},
        {"key": "density", "label": "seed density", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05},
    ]


# --- palettes -----------------------------------------------------------------

_LIFE_PALETTE = ["#0b0f1a", "#e8f0ff"]


def excitable_palette(n: int) -> list[str]:
    """Resting = near-black indigo; excited = hot cyan-white; refractory fades
    back down to deep indigo. Public because the live `states` knob regenerates it
    when the refractory length changes."""
    pal = []
    for k in range(n):
        if k == 0:
            pal.append("#060814")
            continue
        t = (k - 1) / max(n - 2, 1)  # 0 at the excited front .. 1 at last refractory
        r = int(220 * (1 - t) + 18 * t)
        g = int(245 * (1 - t) + 26 * t)
        b = int(255 * (1 - t) + 90 * t)
        pal.append(f"#{r:02x}{g:02x}{b:02x}")
    return pal


def _lerp_hex(c0: str, c1: str, t: float) -> str:
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def gradient_palette(stops: list[tuple[float, str]], n: int = 256) -> list[str]:
    """Interpolate a list of (position, hex) stops into an n-entry gradient — used
    to render continuous (Lenia) fields as 256 quantized colors."""
    pal = []
    for i in range(n):
        t = i / (n - 1)
        if t <= stops[0][0]:
            pal.append(stops[0][1]); continue
        if t >= stops[-1][0]:
            pal.append(stops[-1][1]); continue
        for j in range(len(stops) - 1):
            p0, c0 = stops[j]
            p1, c1 = stops[j + 1]
            if p0 <= t <= p1:
                pal.append(_lerp_hex(c0, c1, (t - p0) / (p1 - p0) if p1 > p0 else 0.0))
                break
    return pal


_LENIA_STOPS = [(0.0, "#04060f"), (0.25, "#0b2f5e"), (0.5, "#1a9aa0"),
                (0.75, "#f2e85c"), (1.0, "#ffffff")]


# --- life-like universes (2-state) -------------------------------------------

register(LawSet(
    id="conway",
    name="Conway's Life",
    description=(
        "B3/S23 — the reference universe. Gliders, oscillators, and structures "
        "capable of universal computation. Toggle a birth/survival bit to feel the "
        "edge of chaos move."
    ),
    family="life",
    states=2,
    params={"birth": [3], "survival": [2, 3]},
    palette=_LIFE_PALETTE,
    seed={"kind": "random", "density": 0.22},
    controls=life_controls(),
))

register(LawSet(
    id="highlife",
    name="HighLife",
    description=(
        "B36/S23 — almost Conway, but birth-on-6 gives it a genuine self-replicator. "
        "Turn B6 off live and watch it become Conway."
    ),
    family="life",
    states=2,
    params={"birth": [3, 6], "survival": [2, 3]},
    palette=["#0b0f1a", "#ffd76a"],
    seed={"kind": "random", "density": 0.22},
    controls=life_controls(),
))

register(LawSet(
    id="daynight",
    name="Day & Night",
    description=(
        "B3678/S34678 — symmetric under swapping live<->dead, so 'matter' and "
        "'vacuum' obey the same law. Grows blobby domains with their own physics-feel."
    ),
    family="life",
    states=2,
    params={"birth": [3, 6, 7, 8], "survival": [3, 4, 6, 7, 8]},
    palette=["#0b0f1a", "#8ef0c0"],
    seed={"kind": "random", "density": 0.5},
    controls=life_controls(),
))


# --- excitable-medium universe (multi-state) ---------------------------------

_EXCITABLE_STATES = 16
register(LawSet(
    id="excitable",
    name="Excitable Medium",
    description=(
        "Greenberg-Hastings — self-organizing spiral waves with a definite signal "
        "speed. Stretch the refractory length or raise the ignition threshold to "
        "reshape the waves."
    ),
    family="excitable",
    states=_EXCITABLE_STATES,
    params={"threshold": 1},
    palette=excitable_palette(_EXCITABLE_STATES),
    seed={"kind": "random", "density": 1.0},
    controls=excitable_controls(),
))


# --- forest-fire universe (stochastic, self-organized criticality) -----------

register(LawSet(
    id="forestfire",
    name="Forest Fire",
    description=(
        "Drossel-Schwabl — the first stochastic universe. Empty grows trees (rate p); "
        "lightning (rate f) or a burning neighbor ignites them; fire dies to ash. With "
        "f << p it self-tunes to the critical point: a universe that finds the edge of "
        "chaos on its own."
    ),
    family="forestfire",
    states=3,
    params={"p": 0.03, "f": 0.0006},
    palette=["#0d130d", "#2f9e44", "#ff7038"],  # empty, tree, fire
    seed={"kind": "random", "density": 0.4},
    controls=forestfire_controls(),
))


# --- Lenia: a continuous universe (real-valued cells) ------------------------

register(LawSet(
    id="lenia",
    name="Lenia",
    description=(
        "A continuous CA — real-valued cells evolved by a smooth kernel and a "
        "growth function. Glider-like 'creatures' crawl out of the soup. Tune the "
        "growth center μ / width σ to morph them (or reseed for new life)."
    ),
    family="lenia",
    states=256,
    # A *moving* multi-ring set (found by search) rather than the classic single-ring
    # params, which freeze into static spots from a random soup.
    params={"R": 12, "mu": 0.265, "sigma": 0.048, "dt": 0.1, "beta": [1.0, 0.58, 0.9]},
    palette=gradient_palette(_LENIA_STOPS),
    seed={"kind": "random", "density": 0.6},
    controls=lenia_controls(),
))


# --- random universe generation ----------------------------------------------

def _color(rng: np.random.Generator, hue=None, sat=(0.55, 0.95), val=(0.85, 1.0)) -> str:
    h = float(rng.uniform(*hue)) if hue else float(rng.random())
    s = float(rng.uniform(*sat))
    v = float(rng.uniform(*val))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _multistate_palette(rng: np.random.Generator, n: int) -> list[str]:
    """State 0 = near-black background; states 1..n-1 = distinct vivid hues spread
    around the color wheel, so a generated multi-state type reads clearly."""
    pal = ["#07070d"]
    base = float(rng.random())
    for k in range(1, n):
        hue = (base + 0.8 * k / max(n - 1, 1)) % 1.0
        pal.append(_color(rng, hue=(hue, hue)))
    return pal


def random_totalistic_lawset(rng: np.random.Generator) -> LawSet:
    """Invent a whole new automaton **type**: a multi-state totalistic CA with a
    randomly generated transition table. Unlike the fixed families, the *rule
    structure itself* is generated, so each one is a qualitatively different kind
    of universe. Sparsity-biased (many entries map to 0) so patterns tend to live
    on a background rather than saturate, and `T[0,0]=0` keeps empty space empty."""
    n = int(rng.integers(3, 6))               # 3..5 states → visibly a new type
    r = 1 if rng.random() < 0.75 else 2
    cells = (2 * r + 1) ** 2 - 1
    max_sum = cells * (n - 1)
    quiet = float(rng.uniform(0.5, 0.8))      # sparsity: most contexts map to empty
    table = np.where(
        rng.random((n, max_sum + 1)) < quiet,
        0, rng.integers(1, n, size=(n, max_sum + 1)),
    ).astype(np.uint8)
    table[0, 0] = 0
    tag = int(rng.integers(1000, 10000))
    return LawSet(
        id=f"rnd-{tag}",
        name=f"random type · {n}-state r{r}",
        description=f"a procedurally generated universe TYPE — {n}-state totalistic CA (radius {r})",
        family="totalistic",
        states=n,
        params={"radius": r, "table": table.tolist()},
        palette=_multistate_palette(rng, n),
        seed={"kind": "random", "density": round(float(rng.uniform(0.3, 0.7)), 2)},
        controls=[{"key": "density", "label": "seed density", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05}],
    )


def _type_interest(lawset: LawSet) -> float:
    """Score a generated type by its free-evolution dynamics: reward alive,
    sustained-but-moderate activity (structured motion, not frozen and not boiling)
    and some spatial structure; ~0 for dead or chaotic. Reuses the sweep metrics."""
    import math
    from .metrics import measure_run
    f = measure_run(lawset, size=64, steps=120, seed=0)
    if not f["alive"]:
        return 0.0
    band = math.exp(-(((f["mean_activity"] - 0.1) / 0.1) ** 2))  # peak near 0.1
    ent = min(f["spatial_entropy"], 3.5) / 3.5
    sat = 1.0 if f["final_density"] < 0.9 else 0.3
    return band * (0.4 + 0.6 * ent) * sat


def random_type(rng: np.random.Generator, tries: int = 10) -> LawSet:
    """Roll several random totalistic *types* and keep the most interesting one
    (most of automata-space is chaos, so we sample and curate). This is the same
    idea as the ML sweep, applied one roll at a time."""
    best, best_score = None, -1.0
    for _ in range(tries):
        cand = random_totalistic_lawset(rng)
        s = _type_interest(cand)
        if s > best_score:
            best, best_score = cand, s
    return best


def _random_gradient(rng: np.random.Generator, n: int = 256) -> list[str]:
    h = float(rng.random())
    mid = _color(rng, hue=(h, h))
    hi = _color(rng, hue=((h + 0.12) % 1.0, (h + 0.12) % 1.0), sat=(0.15, 0.4), val=(0.95, 1.0))
    return gradient_palette([(0.0, "#04060f"), (0.5, mid), (1.0, hi)], n)


def random_lenia_lawset(rng: np.random.Generator) -> LawSet:
    """A random *continuous* universe (Lenia): random kernel rings + growth params."""
    R = int(rng.integers(10, 17))
    nb = int(rng.choice([1, 2, 2, 3, 3]))           # bias to multi-ring (more dynamic)
    beta = [round(float(rng.uniform(0.3, 1.0)), 2) for _ in range(nb)]
    beta[0] = 1.0
    mu = round(float(rng.uniform(0.15, 0.30)), 3)   # the lively-but-not-dead band
    sigma = round(float(rng.uniform(0.03, 0.05)), 3)
    tag = int(rng.integers(1000, 10000))
    return LawSet(
        id=f"rnd-{tag}",
        name=f"random type · Lenia R{R}",
        description=f"a procedurally generated CONTINUOUS universe (Lenia, R={R}, μ={mu}, σ={sigma})",
        family="lenia",
        states=256,
        params={"R": R, "mu": mu, "sigma": sigma, "dt": 0.1, "beta": beta},
        palette=_random_gradient(rng),
        seed={"kind": "random", "density": round(float(rng.uniform(0.4, 0.8)), 2)},
        controls=lenia_controls(),
    )


def _lenia_score(lawset: LawSet) -> float:
    """Curation score for a continuous type: alive (mass in a sane band), spatially
    **structured** (high std → localized shapes, not uniform gray) AND still
    **moving** (nonzero frame-to-frame change after the transient). The motion term
    is the fix for "Lenia always freezes into circular spots" — those static
    Turing-pattern attractors score high on structure but ~0 on motion, so rewarding
    motion steers rolls toward pulsing / drifting / rotating patterns instead."""
    from .engine import make_engine
    eng = make_engine(lawset, (56, 56), np.random.default_rng(0))
    for _ in range(90):
        eng.step()                       # settle well past the transient so we
    prev = eng.field.copy()              # measure *sustained* motion, not a dying flurry
    motion = 0.0
    for _ in range(25):
        eng.step()
        motion += float(np.abs(eng.field - prev).mean())
        prev = eng.field.copy()
    motion /= 25.0
    m = float(eng.field.mean())
    if m < 0.01 or m > 0.55:             # dead or blown out
        return 0.0
    move = min(max((motion - 0.0006) / 0.004, 0.0), 1.0)   # frozen -> 0, moving -> 1
    return float(eng.field.std()) * move


def random_lenia(rng: np.random.Generator, tries: int = 5) -> LawSet:
    best, best_score = None, -1.0
    for _ in range(tries):
        cand = random_lenia_lawset(rng)
        s = _lenia_score(cand)
        if s > best_score:
            best, best_score = cand, s
    return best if best is not None else random_lenia_lawset(rng)


def random_lawset(rng: np.random.Generator) -> LawSet:
    """Invent a fresh, random universe. Picks a random engine family and random
    parameters within sensible ranges, plus a random palette so every roll looks
    distinct. Sometimes generates a whole new *type*: a curated random totalistic
    CA (discrete) or a curated random Lenia (continuous). Otherwise a random rule
    within a known family. Excitable always uses threshold=1."""
    from . import rulespace  # local import avoids any import-order cycle

    # totalistic ×2 and lenia → ~50% of rolls are a procedurally generated TYPE.
    family = ("life", "excitable", "forestfire",
              "totalistic", "totalistic", "lenia")[int(rng.integers(0, 6))]
    if family == "totalistic":
        return random_type(rng)
    if family == "lenia":
        return random_lenia(rng)

    tag = int(rng.integers(1000, 10000))

    if family == "life":
        base = rulespace.bits_to_lawset(rulespace.random_bits(rng), lid=f"rnd-{tag}")
        return replace(
            base,
            name=f"random · {base.name}",
            description=f"a randomly invented universe — life-like {base.name}",
            palette=["#0b0f1a", _color(rng)],
            seed={"kind": "random", "density": round(float(rng.uniform(0.15, 0.4)), 3)},
        )

    if family == "excitable":
        states = int(rng.integers(6, 25))
        return LawSet(
            id=f"rnd-{tag}",
            name=f"random · Excitable ×{states}",
            description=f"a randomly invented excitable medium ({states} refractory states)",
            family="excitable",
            states=states,
            params={"threshold": 1},
            palette=excitable_palette(states),
            seed={"kind": "random", "density": round(float(rng.uniform(0.5, 1.0)), 2)},
            controls=excitable_controls(),
        )

    p = round(float(rng.uniform(0.01, 0.12)), 4)
    f = round(float(rng.uniform(0.0002, 0.004)), 5)
    return LawSet(
        id=f"rnd-{tag}",
        name=f"random · Forest p={p} f={f}",
        description=f"a randomly invented forest fire (growth p={p}, lightning f={f})",
        family="forestfire",
        states=3,
        params={"p": p, "f": f},
        palette=["#0d130d", _color(rng, hue=(0.25, 0.45)), _color(rng, hue=(0.0, 0.1))],
        seed={"kind": "random", "density": round(float(rng.uniform(0.2, 0.5)), 2)},
        controls=forestfire_controls(),
    )


def get(lawset_id: str) -> LawSet:
    return LIBRARY[lawset_id]


def catalog() -> list[dict]:
    """Ordered, JSON-serializable list of every universe, for the picker UI."""
    return [LIBRARY[i].to_public() for i in ORDER]


DEFAULT_ID = "conway"
