# reality-sim — kickoff brief

- **Problem:** Author alternative "laws of physics" as lattice update rules and
  actually *watch* the resulting toy universes evolve — as a step toward the
  bigger question of whether reality can be universally simulated at all.
- **Done looks like:** A pluggable package where a **law-set is a module**
  (`family, states, neighborhood, rule params, palette`); pick one, evolve it,
  and see it live in the browser with play / pause / step, a rule picker, and a
  brush to inject structures.
- **Not now:** The formal "check against real QM" machinery; continuum and
  quantum engines; performance/scaling; hosting or multi-user. Get one substrate
  beautiful and extensible first.
- **First slice (DONE):** numpy CA core + `LawSet` spec + 4 universes (Conway,
  HighLife, Day&Night, Greenberg-Hastings excitable medium) → streamed over a
  websocket to a browser canvas with transport controls, universe picker, live
  resize, and a paint brush. Verified end-to-end.
- **Open question (the north star):** what does *"check against our quantum
  physics"* concretely mean as a pass/fail — recover QM as a limit? Not violate
  locality / no-signaling / unitarity? Deferred on purpose: much easier to answer
  once there are real universes on screen to point at and *measure*.

## Honest framing

These are toy universes we define; they yield real, measurable answers **about
those universes**, not a theorem about our own physics. The grand questions
(FTL, a civilization discovering it, crossing into a foreign-rules universe and
being annihilated) become well-posed as **probes** on a running universe:
signal-front speed vs. the light-cone; the replication/complexity frontier;
structure survival across a law-set boundary. That's the roadmap.
