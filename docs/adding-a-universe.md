# Adding a Universe

There are two very different kinds of "add a universe," and it's worth being
clear about which one you're doing:

1. **A new rule in an existing family** (a new life-like rule, a new
   excitable variant) — this is a pure data change: one new `LawSet` entry in
   `reality_sim/lawsets.py`. No engine, server, or frontend code changes.
2. **A whole new kind of physics** (a new engine family — continuum fields,
   N-body, anything that isn't outer-totalistic-on-a-Moore-neighborhood or
   Greenberg-Hastings) — this means writing a new `Engine` subclass and
   registering it.

Most of the time you want #1. This doc covers both, in that order.

## 1. A new universe in an existing family

### The `LawSet` fields you control

Every entry in `reality_sim/lawsets.py` is built with `register(LawSet(...))`.
The fields:

| field | meaning |
|---|---|
| `id` | short stable identifier — used in URLs / the `set_lawset` command |
| `name` | human label shown in the picker |
| `description` | what this universe *is* / what emerges — shown under the picker |
| `family` | which engine evolves it; must be a key in `engine.ENGINES` (`"life"` or `"excitable"` today) |
| `states` | number of distinct cell states, `0..states-1` |
| `params` | family-specific rule parameters (see below) |
| `palette` | one `"#rrggbb"` string per state, index == state value |
| `seed` | initial-condition recipe: `{"kind": "random", "density": ...}` or `{"kind": "clear"}` |

For the **`"life"`** family, `params` is `{"birth": [...], "survival": [...]}`
— lists of Moore-neighbor counts (`0..8`). `LifeEngine` turns each list into a
boolean lookup table indexed by neighbor count: a dead cell is born if its
live-neighbor count is in `birth`; a live cell survives if its count is in
`survival`. Anything not listed in `survival` dies; anything not listed in
`birth` stays dead. Nothing enforces it in code, but every shipped `"life"`
LawSet uses `states=2` (dead/alive) — that's the natural fit for a
birth/survival rule over a two-state grid.

For the **`"excitable"`** family, `params` is `{"threshold": n}` — the number
of currently-excited (state `1`) Moore neighbors a resting (state `0`) cell
needs to fire. `states` sets how many refractory steps a cell spends
recovering (state `k` always advances to `k+1`, wrapping `states-1` back to
`0`). See the gotcha below before picking a threshold.

### Worked example: Seeds (a life-like rule)

Seeds (`B2/S`) is a good second example precisely because it's *not* like
Conway: no cell ever survives — every live cell dies every generation — so
the only thing keeping the universe going is birth on exactly 2 neighbors.
The result is explosive, non-settling growth rather than Conway's mix of
stable/oscillating/traveling structures.

```python
register(LawSet(
    id="seeds",
    name="Seeds",
    description=(
        "B2/S — every live cell dies every generation; the universe only "
        "persists by birth on exactly 2 neighbors. No stable structures, "
        "just relentless sparking growth."
    ),
    family="life",
    states=2,
    params={"birth": [2], "survival": []},
    palette=["#0b0f1a", "#ff6b6b"],
    seed={"kind": "random", "density": 0.1},
))
```

Notes on the choices:

- `"survival": []` is deliberate, not an oversight — it's what makes Seeds
  *Seeds*. `LifeEngine` builds `_survival_lut` from `set(int(x) for x in
  params.get("survival", [2, 3]))`; an empty list gives an all-`False` LUT,
  so no live cell ever survives regardless of neighbor count.
- The seed density is low (`0.1` vs. Conway's `0.22`) because Seeds grows
  explosively from almost any live cell — starting dense just floods the
  grid in a few generations.
- Add the entry anywhere in the "life-like universes" section of
  `lawsets.py`; registration order is display order (`ORDER` is a plain
  list, appended to by `register()`).

### A second one-liner: Maze

Maze (`B3/S12345`) is another life-like rule worth knowing about: it births
like Conway (on exactly 3 neighbors) but survives across a much wider band
(`1` through `5`), which makes it fill space with stable corridor-like
structures instead of gliders:

```python
params={"birth": [3], "survival": [1, 2, 3, 4, 5]}
```

Same family, same engine, same server and frontend code — only the two lists
change.

## 2. A whole new engine family

When a new physics genuinely doesn't fit "outer-totalistic rule over a Moore
neighborhood" or "Greenberg-Hastings excitable medium," write a new `Engine`
subclass in `reality_sim/engine.py` (or your own module, since `ENGINES` is
just a plain module-level dict you can add to):

1. **Subclass `Engine`** and set a class-level `family` string — this is the
   key `make_engine()` will look up.
2. **Implement `seed(self)`** — build `self.grid` from `self.lawset.seed`
   (`recipe.get("kind")`, typically `"random"` or `"clear"`), and reset
   `self.generation = 0`. Follow `LifeEngine`/`ExcitableEngine`'s pattern of
   raising `ValueError` on an unrecognized `kind`.
3. **Implement `step(self)`** — advance `self.grid` by exactly one generation
   using vectorized numpy/scipy (no per-cell Python loops — that's what keeps
   a step under a millisecond even at large grid sizes), and increment
   `self.generation`.
4. **Optionally override `stats(self)`** to report family-specific numbers —
   `ExcitableEngine` adds an `"excited"` count on top of the base
   `generation`/`live`/`density`.
5. **Keep `self.grid` a 2-D `uint8` array with values in `0..lawset.states-1`**
   — this is the one invariant `server.py` and `frontend/app.js` depend on;
   everything downstream of the engine is generic *because* they never see
   anything else.
6. **Register it**: add `ENGINES["myfamily"] = MyEngine` (or add the class
   directly into the `ENGINES` dict literal in `engine.py` if you're editing
   the package itself).
7. **Tag matching `LawSet`s** with `family="myfamily"` in `lawsets.py`
   (or wherever you register universes) — `make_engine()` picks up the new
   family automatically, with no other code path to touch.

That's the entire seam: `LawSet.family` plus the `ENGINES` dict is what makes
the system pluggable, and it's intentionally the *only* place new physics has
to be wired in.

## Gotcha: the excitable engine needs `threshold=1` to sustain waves

The shipped `"excitable"` LawSet uses `params={"threshold": 1}` with `states =
16`, and that's not an arbitrary default — it was found by sweeping the
parameter. `ExcitableEngine`'s rule is: a resting cell fires when its count of
*excited* (state `1`, not refractory) Moore neighbors meets `threshold`. With
16 states, a traveling wavefront is thin — only a one-cell-wide ring of cells
is actually in state `1` at any given tick, with everything behind it already
advanced into refractory states `2..15`.

That thinness is exactly why `threshold >= 2` kills the dynamics once `states
>= 8`: a resting cell sitting just ahead of a thin, mostly-convex wavefront
typically only borders *one* excited cell at a time, not two. Demanding two
simultaneously-excited neighbors means most of the front simply fails to
propagate past its first tick or two — the wave induces briefly, then dies
out, rather than sustaining the self-organizing spiral/target waves the rule
is supposed to produce. `threshold=1` is what actually lets those waves
survive and propagate indefinitely. If you add a new excitable-family
universe with a different `states` count or a different intended threshold,
treat this as a real constraint to re-check, not a knob to tune blind —
verify waves actually sustain past the first few dozen generations before
registering it.
