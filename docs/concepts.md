# Concepts

## What a cellular automaton is

A cellular automaton (CA) is one of the simplest models of "a physics" you can
write down: a grid of cells, each holding one of a small set of discrete
states, all updating together in discrete time steps according to a *local*
rule — a cell's next state depends only on its own state and the states of a
fixed neighborhood around it. No cell ever looks further than its neighbors,
and every cell obeys the same rule at the same time.

Everything in reality-sim is that idea, made concrete:

- **Grid** — a 2-D array, `(h, w)`, stored as `uint8` (`Engine.grid` in
  `reality_sim/engine.py`).
- **Discrete states** — an integer `0..states-1` per cell; what a state
  *means* (dead/alive, resting/excited/refractory) is defined per universe.
- **Local update rule** — computed via a convolution over the Moore
  neighborhood (the 8 surrounding cells), evaluated for every cell at once.
- **Discrete time** — a `generation` counter, incremented by exactly one per
  `step()`.
- **Toroidal** — `scipy.ndimage.convolve(..., mode="wrap")` means the grid's
  edges wrap around; there is no boundary, no "edge of the universe" where the
  rule has to behave differently. Every cell has a full neighborhood, always.

Two rule families ship today. `LifeEngine` implements *outer-totalistic
life-like* rules — a birth set and a survival set over the neighbor count
(Conway's B3/S23 and its relatives). `ExcitableEngine` implements a
Greenberg-Hastings *excitable medium* — resting/excited/refractory states that
produce self-sustaining spiral and target waves, a genuinely different flavor
of dynamics from Life's gliders and oscillators. They're deliberately
different enough to prove the substrate isn't secretly "Life plus reskins."

## `LawSet`: a physics as a value

The `LawSet` dataclass (`reality_sim/lawset.py`) is the thing this whole
project is organized around: *inventing new physics == writing a new
`LawSet`.* Concretely, a `LawSet` fixes:

- which engine family evolves it (`family`),
- how many distinct states a cell can hold (`states`),
- the family-specific rule parameters (`params` — e.g. birth/survival sets, or
  an excitation threshold),
- how the viewer should draw it (`palette`),
- and how to initialize it (`seed`).

That's the entire vocabulary needed to name a toy universe. It says nothing
about *how* the rule is computed — that's the engine's job — which is exactly
why a `LawSet` can be treated as data: logged, diffed, serialized to the
browser, or (eventually) searched over, without dragging any numpy or asyncio
along with it.

## Honest scope: these are toy universes, not a theory of everything

It's worth being precise about what running one of these simulations does and
doesn't tell you.

**What it does:** a `LawSet` is a complete, closed specification of a
universe. Once you fix it, every question you can ask about *that* universe —
how fast does a signal cross the grid, does a glider survive contact with a
different rule, how quickly does something self-replicating appear — has a
real, computable answer. You can run it and watch the answer happen.

**What it doesn't:** none of this is a claim that our own universe *is* a
cellular automaton, or a proof about the actual laws of physics. We don't have
reality's true update rule in closed form to plug in here — and that's not a
gap this package is trying to close by brute force. It's not a compute
problem ("simulate harder and you'll converge on real physics"); it's an
epistemic one — we don't know the rule, so there's nothing to encode. What
reality-sim gives you instead is a substrate where *chosen* rules become
things you can point at and measure, which is a meaningfully different (and
much more modest) claim than "this models our universe."

## The causal light cone

The single most important physical fact about any grid in this package falls
out of one implementation detail: neighbor counts are computed with a 3×3
convolution (the `MOORE` kernel), so a cell's next state can only depend on
cells at most 1 step away, in a single generation. Run that forward and the
consequence is exact and unavoidable: a disturbance at one cell can influence
a cell `d` steps away only after at least `d` generations have passed.
Information — a signal, a perturbation, a painted structure — cannot outrun
one cell per generation, no matter which `LawSet` you're running.

This is the discrete analogue of a speed-of-light limit, and it's the same
shape of idea as the Lieb-Robinson bound in quantum many-body physics: even in
systems with no literal relativistic speed limit built in, *locality* of the
update rule alone forces a finite, computable "light cone" outside of which
correlations can't yet exist. Here it's not an emergent bound derived from
some deeper structure — it's baked in by construction, because
`mode="wrap"` convolution with a fixed small kernel is the only way
information moves. But that's precisely what makes it useful: it's a known,
exact baseline (exactly 1 cell/generation) that any candidate "faster"
mechanism — a clever structure, an unusual rule — would have to be measured
against and beat.

## The north star: grand questions as measurements

The reason the causal light cone matters isn't philosophical — it's that it
turns big, otherwise-unanswerable questions into well-posed measurements on a
running universe:

- **"Is faster-than-light signaling possible here?"** Given the 1-cell/
  generation bound above, this becomes a concrete experiment: perturb a
  single cell, track how fast the resulting disturbance front actually
  spreads, and ask whether any structured pattern (as opposed to raw
  neighbor-counting) can make that front outrun the naive bound. Either it
  can in a given `LawSet`, or it can't — and now you have a number, not a
  hunch.
- **"Could a civilization bootstrap itself here, and how fast?"** A
  civilization discovering physics is a hard thing to define in the
  abstract, but "how quickly does self-replicating or computation-capable
  structure arise from a given seed" is a measurable proxy — and it's not
  hypothetical: HighLife's B36/S23 rule has a genuine self-replicator
  precisely because of its extra birth-on-6 case, unlike Conway's B3/S23.
  The frontier of *when* replicators reliably emerge from a random or
  engineered seed is a real, countable quantity.
- **"What happens if you cross into a universe with different laws?"**
  Stitch two `LawSet`s together across a boundary and transplant a stable
  structure (a glider, a spiral wave) across it. Whether it survives,
  dissolves, or detonates on contact with foreign rules is not a thought
  experiment once you can actually build the wall and run the transplant.

None of these probes exist in the codebase yet — the substrate above (grid,
engine, wire protocol, viewer) is what's built today, and it's what makes
those questions askable in the first place. But the shape of the answer is
already visible in how the system is built: every one of those "grand
questions" reduces to *watch a running universe and measure something about
its light cone, its structures, or its boundary.* That reduction — from
philosophy to instrumentation — is the actual point of the project.
