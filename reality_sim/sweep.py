"""Sweep rule-space in parallel: evaluate thousands of universes, one row each.

This is the data-generation half of the ML pipeline. It samples random life-like
rules (plus all the known landmark rules), evolves each from a random soup, and
records its dynamical feature vector. Output is a tidy table (one rule per row:
its 18 rule bits + its measured features) — exactly the shape scikit-learn wants.

Runs across all cores via a process pool; ~thousands of rules per minute on a
workstation, and the same code scales to a cluster for a full census.

CLI::

    python -m reality_sim.sweep --n 3000 --size 64 --steps 250 --out data/sweeps/life.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import rulespace
from .metrics import measure_mean


def _eval_rule(payload: tuple) -> dict:
    """Worker: evaluate one rule. Must be top-level to be picklable across the
    process pool. ``payload`` is (bits_bytes, name, known, size, steps, reps, seed)."""
    bits_list, name, known, size, steps, reps, seed = payload
    bits = np.array(bits_list, dtype=np.uint8)
    lawset = rulespace.bits_to_lawset(bits, name=name)
    feats = measure_mean(lawset, size=size, steps=steps, reps=reps, seed=seed)

    birth, survival = rulespace.bits_to_bs(bits)
    row: dict = {
        "name": lawset.name,
        "rule": rulespace.rule_name(birth, survival),
        "known": bool(known),
    }
    for i, label in enumerate(rulespace.BIT_LABELS):
        row[label] = int(bits[i])
    row.update(feats)
    return row


def sweep(n: int = 2000, size: int = 64, steps: int = 200, reps: int = 2,
          seed: int = 0, workers: int | None = None, allow_b0: bool = False,
          include_known: bool = True, progress: bool = True) -> pd.DataFrame:
    """Evaluate ``n`` random rules (+ known landmarks) and return a DataFrame."""
    rng = np.random.default_rng(seed)
    tasks: list[tuple] = []
    seen: set[bytes] = set()

    if include_known:
        for name, bits in rulespace.known_bits():
            seen.add(bits.tobytes())
            tasks.append((bits.tolist(), name, True, size, steps, reps, seed))

    while len(seen) < n + (len(rulespace.KNOWN_RULES) if include_known else 0):
        bits = rulespace.random_bits(rng, allow_b0=allow_b0)
        key = bits.tobytes()
        if key in seen:
            continue
        seen.add(key)
        tasks.append((bits.tolist(), None, False, size, steps, reps, seed))

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_eval_rule, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if progress and (i % 200 == 0 or i == len(tasks)):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(tasks)} rules  ({rate:.0f}/s)", file=sys.stderr)

    df = pd.DataFrame(rows)
    # Stable column order: metadata, rule bits, then features.
    meta = ["name", "rule", "known"]
    cols = meta + rulespace.BIT_LABELS + [c for c in df.columns if c not in meta + rulespace.BIT_LABELS]
    return df[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="sweep life-like rule-space")
    ap.add_argument("--n", type=int, default=2000, help="number of random rules")
    ap.add_argument("--size", type=int, default=64, help="grid edge length")
    ap.add_argument("--steps", type=int, default=200, help="generations per run")
    ap.add_argument("--reps", type=int, default=2, help="random seeds averaged per rule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--allow-b0", action="store_true", help="include the degenerate B0 rules")
    ap.add_argument("--no-known", action="store_true", help="skip the landmark rules")
    ap.add_argument("--out", default="data/sweeps/life.parquet")
    args = ap.parse_args()

    print(f"sweeping {args.n} rules  (size={args.size}, steps={args.steps}, reps={args.reps})", file=sys.stderr)
    df = sweep(n=args.n, size=args.size, steps=args.steps, reps=args.reps, seed=args.seed,
               workers=args.workers, allow_b0=args.allow_b0, include_known=not args.no_known)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} rows -> {out}", file=sys.stderr)
    print(f"  alive: {int(df['alive'].sum())}  |  columns: {len(df.columns)}", file=sys.stderr)


if __name__ == "__main__":
    main()
