# reality-sim

Simulate **toy universes with pluggable laws of physics**, and watch them evolve
in the browser.

The organizing idea: *inventing new physics == writing a new `LawSet`.* A
`LawSet` is plain data (dimensionality, number of states, neighborhood, rule
parameters, palette). Everything else — the numpy engines, the live viewer, the
future measurement probes — is generic over it, so adding a universe is adding an
entry, not rewriting the stack.

> **What this is and isn't.** These are *toy* universes: discrete cellular
> automata whose rules we choose. They let us ask big questions in a well-posed
> way — *how fast can a signal travel here? can a structure survive crossing into
> a universe with different rules?* — and get real, measurable answers **about
> those universes.** They are not a proof about our own physics; we don't have
> reality's true rules in closed form. The value is that every "law of physics"
> is something we can run, watch, and measure.

## Quick start

```bash
pip install -e .          # numpy, scipy, aiohttp
python -m reality_sim.server
# open http://127.0.0.1:8770
```

In the viewer: pick a universe, scrub speed, resize the grid, and **drag on the
canvas to paint structures / inject signals** into a running universe.
Keyboard: `space` play/pause, `s` step, `r` reseed.

## Headless / batch

```python
import numpy as np
from reality_sim import make_engine, lawsets

eng = make_engine(lawsets.get("excitable"), shape=(512, 512),
                  rng=np.random.default_rng(0))
for _ in range(1000):
    eng.step()
print(eng.stats())          # {'generation': 1000, 'live': ..., 'excited': ...}
```

The engine is fully vectorized (neighbor counts via `scipy.ndimage.convolve`,
`mode="wrap"` → every universe is a torus), so the same code that prototypes at
240² scales to large grids on a bigger machine.

## The universes so far

| id | family | what it is |
|----|--------|------------|
| `conway` | life | B3/S23 — gliders, oscillators, universal computation |
| `highlife` | life | B36/S23 — has a genuine self-replicator |
| `daynight` | life | B3678/S34678 — matter/vacuum-symmetric domains |
| `excitable` | excitable | Greenberg-Hastings, 16 states — spiral waves with a definite signal speed |
| `forestfire` | forestfire | Drossel-Schwabl — stochastic; self-organizes to criticality |

Every universe exposes **live-tunable knobs** in the viewer — edit the law while
it runs and the pattern reacts in place (the grid isn't reset): toggle Conway's
birth/survival bits, stretch the excitable medium's refractory tail, or tip the
forest between smoldering and firestorm with the growth `p` / lightning `f` dials.

![forest fire under rising lightning rate](docs/img/forestfire.png)

## Rule-space sweeps + ML

The viewer runs one universe; the sweep pipeline runs *thousands* and puts a
model on top, to ask: **can we predict what kind of universe a law makes without
simulating it?** For the life-like family (a law = 18 bits, B0..B8/S0..S8) the
answer is yes ~90% of the time.

```bash
pip install -e ".[ml]"      # adds scikit-learn, pandas, matplotlib
python -m reality_sim.sweep    --n 4000 --size 64 --steps 250 --out data/sweeps/life.parquet
python -m reality_sim.analysis --in data/sweeps/life.parquet --out data/sweeps
```

The sweep evaluates ~560 universes/second across all cores (4,000 laws in <8s),
measuring each one's *free* dynamics (no probing). The analysis then clusters
rule-space into four regimes — **Dead / Frozen / Complex / Chaotic** — and trains
RandomForests to predict a law's regime (**CV accuracy 0.905**) and activity
(**CV R² 0.873**) from its bits alone. The blind search rediscovers Conway and
HighLife in the Complex "edge of chaos" corner. Full write-up and figures in
[docs/sweeps.md](docs/sweeps.md).

![phase diagram](docs/img/phase_diagram.png)

## Documentation

- [docs/concepts.md](docs/concepts.md) — what these toy universes are, and the honest scope
- [docs/architecture.md](docs/architecture.md) — how the package fits together
- [docs/protocol.md](docs/protocol.md) — the websocket wire protocol
- [docs/adding-a-universe.md](docs/adding-a-universe.md) — add a law-set or a whole engine family
- [docs/metrics.md](docs/metrics.md) — the dynamical features a sweep measures
- [docs/sweeps.md](docs/sweeps.md) — the rule-space sweep + ML pipeline

## Architecture

```
reality_sim/
  lawset.py     LawSet — the portable spec of one universe's physics
  engine.py     numpy engines (family "life", "excitable") + make_engine()
  lawsets.py    the library of concrete universes
  server.py     aiohttp: streams binary grid frames over a websocket
frontend/       canvas viewer + controls (vanilla JS, no build step)
```

**Wire protocol.** Client sends JSON commands (`play`/`pause`/`step`/`reset`/
`set_lawset`/`set_size`/`set_fps`/`paint`). Server streams binary frames
(`<uint32 w><uint32 h><uint32 generation>` + row-major `uint8` states) plus a
JSON `status` on structural changes. One sim loop per connection is the sole
writer, so there are no interleaved-send races.

## Roadmap (the north star)

- **✅ ML-driven rule search** — sweep rule-space, learn the law→behavior map.
  Done for the life-like family; see [docs/sweeps.md](docs/sweeps.md).
- **active search** — use the learned surrogate + Bayesian optimization to
  *propose* laws in the rare Complex corner instead of sampling uniformly.
- **complexity / replication frontier** — measure how fast self-replicating or
  computing structure arises (proxy for "how fast could a civilization bootstrap
  physics?").
- **signal speed / light-cone** *(probing)* — perturb one cell, measure how fast
  the disturbance front spreads; hunt for laws where structured signals outrun the
  naive 1-cell/generation bound ("is FTL possible *here*?").
- **universe boundaries** — stitch two law-sets across a wall; transplant a stable
  structure across it and measure whether it survives, dissolves, or detonates.
- **new engine families** — continuum fields / N-body, then a quantum wavefunction
  engine, all behind the same `LawSet` seam.

See [`BRIEF.md`](BRIEF.md) for the kickoff framing.
