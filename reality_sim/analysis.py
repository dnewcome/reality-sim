"""ML over a rule-space sweep: map it, name its regimes, and learn the law->behavior map.

Given a sweep table (from :mod:`reality_sim.sweep`) this does three things:

  1. **Unsupervised map.** Standardize the dynamical features, KMeans them into
     regimes, and PCA to 2-D for a picture. This is the "map of possible
     universes" — and the landmark rules (Conway, Seeds, ...) let us sanity-check
     that the map's regions mean what we think.

  2. **Name the regimes.** Each cluster is matched to an archetype
     (Dead / Frozen / Complex / Chaotic) by its mean feature profile, so the map
     is readable instead of "cluster 0/1/2/3".

  3. **Learn the law -> behavior map.** Train a RandomForest to predict a rule's
     regime purely from its 18 birth/survival bits, and another to regress its
     activity. Cross-validated scores say *how predictable* rule-space is, and the
     feature importances say *which* birth/survival counts decide a universe's fate
     — i.e. which knobs of the law matter most.

CLI::

    python -m reality_sim.analysis --in data/sweeps/life.parquet --out data/sweeps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from . import rulespace
from .engine import make_engine
from .metrics import FEATURE_NAMES

# Archetype feature profiles (raw units) used to name the KMeans clusters.
# Columns: [final_density, mean_activity, spatial_entropy, period].
SIGNAL = ["final_density", "mean_activity", "spatial_entropy", "period"]
ARCHETYPES = {
    "Dead":    [0.00, 0.00, 0.2, 1.0],
    "Frozen":  [0.55, 0.00, 2.8, 1.0],
    "Complex": [0.08, 0.06, 1.6, 0.0],
    "Chaotic": [0.30, 0.35, 3.4, 0.0],
}
PHASE_COLORS = {
    "Dead": "#3b4252", "Frozen": "#5cc8ff", "Complex": "#ffd76a", "Chaotic": "#ff6b8a",
}

_INK = "#dfe7f5"
_BG = "#0b0f1a"


def _dark(ax):
    ax.set_facecolor(_BG)
    for s in ax.spines.values():
        s.set_color("#2a3550")
    ax.tick_params(colors=_INK, labelsize=8)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def label_clusters(df: pd.DataFrame, k: int) -> dict[int, str]:
    """Assign each cluster the best-matching archetype name (Hungarian-optimal)."""
    means = df.groupby("cluster")[SIGNAL].mean()
    if k != len(ARCHETYPES):
        order = means["mean_activity"].sort_values().index
        return {c: f"Regime {i + 1}" for i, c in enumerate(order)}
    names = list(ARCHETYPES)
    A = np.array([ARCHETYPES[n] for n in names], dtype=float)
    M = means.to_numpy()
    mu, sd = M.mean(0), M.std(0) + 1e-9
    Az, Mz = (A - mu) / sd, (M - mu) / sd
    cost = np.linalg.norm(Mz[:, None, :] - Az[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    idx = list(means.index)
    return {idx[r]: names[c] for r, c in zip(rows, cols)}


def _render_final(bits: np.ndarray, size: int = 110, steps: int = 250, seed: int = 0) -> np.ndarray:
    ls = rulespace.bits_to_lawset(bits)
    eng = make_engine(ls, (size, size), np.random.default_rng(seed))
    for _ in range(steps):
        eng.step()
    g = (eng.grid != 0).astype(np.uint8)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[g == 0] = (11, 15, 26)
    rgb[g == 1] = (232, 240, 255)
    return rgb


def analyze(df: pd.DataFrame, out_dir: str | Path, k: int = 4, seed: int = 0) -> dict:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # --- 1. cluster + PCA -------------------------------------------------
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
    df["cluster"] = km.labels_
    names = label_clusters(df, k)
    df["phase"] = df["cluster"].map(names)
    # Distance to own cluster centroid → lets us pick the most *typical* member
    # of each regime as its visual exemplar.
    df["centroid_dist"] = km.transform(Xs)[np.arange(len(df)), km.labels_]

    xy = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    df["pc1"], df["pc2"] = xy[:, 0], xy[:, 1]

    # --- 2. learn law -> behavior ----------------------------------------
    bits = df[rulespace.BIT_LABELS].to_numpy()
    y_cls = df["cluster"].to_numpy()
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    cls_acc = float(cross_val_score(clf, bits, y_cls, cv=5).mean())
    clf.fit(bits, y_cls)

    reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    reg_r2 = float(cross_val_score(reg, bits, df["mean_activity"], cv=5, scoring="r2").mean())
    reg.fit(bits, df["mean_activity"])

    # --- 3. "interesting" universes the search surfaced ------------------
    # Heuristic peaked at the Conway-like edge of chaos: alive, aperiodic, SPARSE
    # (low density), sustained-but-low activity, and *moderate* (not maximal)
    # spatial entropy — i.e. localized structure, not boiling noise. A ranking aid,
    # not ground truth; the honest observables are the cluster + the CV scores.
    def _g(x, mu, sig):
        return np.exp(-(((x - mu) / sig) ** 2))
    df["interest"] = (
        df["alive"] * (df["period"] == 0)
        * _g(df["mean_activity"], 0.05, 0.05)
        * _g(df["final_density"], 0.12, 0.12)
        * _g(df["spatial_entropy"], 1.8, 1.2)
    )

    summary = {
        "n_rules": int(len(df)),
        "k": k,
        "phase_counts": {names[c]: int((df["cluster"] == c).sum()) for c in sorted(names)},
        "classifier_cv_accuracy": round(cls_acc, 3),
        "regressor_cv_r2": round(reg_r2, 3),
        "known_rule_phases": {
            r["name"]: r["phase"] for _, r in df[df["known"]].iterrows()
        },
        "top_interesting": [
            {"rule": r["rule"], "phase": r["phase"],
             "activity": round(r["mean_activity"], 3),
             "entropy": round(r["spatial_entropy"], 2),
             "known": bool(r["known"]), "name": r["name"]}
            for _, r in df.sort_values("interest", ascending=False).head(10).iterrows()
        ],
    }

    _figures(df, names, clf, reg, out / "figures")
    _write_report(df, names, summary, out / "report.md")
    df.to_parquet(out / "analyzed.parquet")
    return summary


def _figures(df, names, clf, reg, fig_dir: Path) -> None:
    phase_of = {c: names[c] for c in names}

    # (a) PCA phase map
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=_BG)
    _dark(ax)
    for c, nm in phase_of.items():
        sub = df[df["cluster"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=7, alpha=0.55,
                   color=PHASE_COLORS.get(nm, "#888"), label=f"{nm} ({len(sub)})", linewidths=0)
    known = df[df["known"]]
    ax.scatter(known["pc1"], known["pc2"], s=44, facecolor="none",
               edgecolor="white", linewidths=1.1, zorder=5)
    for _, r in known.iterrows():
        ax.annotate(r["name"], (r["pc1"], r["pc2"]), color="white", fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Map of rule-space  ·  PCA of dynamical features")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leg = ax.legend(loc="best", fontsize=8, framealpha=0.15)
    for t in leg.get_texts():
        t.set_color(_INK)
    fig.savefig(fig_dir / "phase_map.png", dpi=130, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)

    # (b) interpretable phase diagram: activity vs density
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=_BG)
    _dark(ax)
    for c, nm in phase_of.items():
        sub = df[df["cluster"] == c]
        ax.scatter(sub["mean_activity"], sub["final_density"], s=7, alpha=0.5,
                   color=PHASE_COLORS.get(nm, "#888"), label=nm, linewidths=0)
    for _, r in df[df["known"]].iterrows():
        ax.scatter(r["mean_activity"], r["final_density"], s=44, facecolor="none",
                   edgecolor="white", linewidths=1.1, zorder=5)
        ax.annotate(r["name"], (r["mean_activity"], r["final_density"]), color="white",
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_title("Phase diagram  ·  order → chaos")
    ax.set_xlabel("sustained activity  (frozen → boiling)")
    ax.set_ylabel("final density")
    leg = ax.legend(loc="best", fontsize=8, framealpha=0.15)
    for t in leg.get_texts():
        t.set_color(_INK)
    fig.savefig(fig_dir / "phase_diagram.png", dpi=130, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)

    # (c) rule-bit importances
    order = np.arange(rulespace.N_BITS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=_BG)
    for ax, imp, title, col in (
        (axes[0], clf.feature_importances_, "predicting REGIME from the law", "#ffd76a"),
        (axes[1], reg.feature_importances_, "predicting ACTIVITY from the law", "#5cc8ff"),
    ):
        _dark(ax)
        ax.barh([rulespace.BIT_LABELS[i] for i in order], imp[order], color=col)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("importance")
    fig.suptitle("Which birth/survival counts decide a universe's fate", color=_INK)
    fig.savefig(fig_dir / "importances.png", dpi=130, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)

    # (d) visual exemplars: one representative universe per phase
    phases = [nm for _, nm in sorted(phase_of.items())]
    fig, axes = plt.subplots(1, len(phases), figsize=(3.1 * len(phases), 3.4), facecolor=_BG)
    if len(phases) == 1:
        axes = [axes]
    for ax, nm in zip(axes, phases):
        sub = df[df["phase"] == nm]
        known_sub = sub[sub["known"]]
        if nm == "Complex" and len(known_sub):
            # showcase a recognizable classic (e.g. Conway) nearest the centroid
            row = known_sub.sort_values("centroid_dist").iloc[0]
        else:
            # the most typical member of the regime
            row = sub.sort_values("centroid_dist").iloc[0]
        bits = row[rulespace.BIT_LABELS].to_numpy().astype(np.uint8)
        ax.imshow(_render_final(bits), interpolation="nearest")
        ax.set_title(f"{nm}\n{row['rule']}", color=PHASE_COLORS.get(nm, _INK), fontsize=9)
        ax.axis("off")
    fig.suptitle("A universe from each regime (final state)", color=_INK)
    fig.savefig(fig_dir / "exemplars.png", dpi=130, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)


def _df_to_md(frame: pd.DataFrame, index_name: str) -> str:
    """Render a DataFrame as a markdown table without the optional tabulate dep."""
    cols = list(frame.columns)
    head = "| " + " | ".join([index_name] + [str(c) for c in cols]) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    rows = [
        "| " + " | ".join([str(idx)] + [f"{frame.loc[idx, c]:g}" for c in cols]) + " |"
        for idx in frame.index
    ]
    return "\n".join([head, sep] + rows)


def _write_report(df, names, summary, path: Path) -> None:
    means = df.groupby("phase")[SIGNAL + ["alive"]].mean().round(3)
    lines = [
        "# Rule-space sweep — analysis report", "",
        f"- rules evaluated: **{summary['n_rules']}**",
        f"- regimes (k): **{summary['k']}**",
        f"- classifier CV accuracy (law bits → regime): **{summary['classifier_cv_accuracy']}**",
        f"- regressor CV R² (law bits → activity): **{summary['regressor_cv_r2']}**", "",
        "## Regime sizes", "",
        "| regime | count |", "|---|---|",
    ]
    for nm, cnt in summary["phase_counts"].items():
        lines.append(f"| {nm} | {cnt} |")
    lines += ["", "## Regime mean profiles", "", _df_to_md(means, "regime"), ""]
    lines += ["## Where the landmark rules landed", "", "| rule | regime |", "|---|---|"]
    for nm, ph in sorted(summary["known_rule_phases"].items()):
        lines.append(f"| {nm} | {ph} |")
    lines += ["", "## Most interesting universes the search surfaced", "",
              "| rule | regime | activity | entropy | known |", "|---|---|---|---|---|"]
    for t in summary["top_interesting"]:
        lines.append(f"| {t['rule']} | {t['phase']} | {t['activity']} | {t['entropy']} | {'yes' if t['known'] else ''} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="ML analysis of a rule-space sweep")
    ap.add_argument("--in", dest="inp", default="data/sweeps/life.parquet")
    ap.add_argument("--out", default="data/sweeps")
    ap.add_argument("--k", type=int, default=4, help="number of regimes (KMeans clusters)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    summary = analyze(df, args.out, k=args.k, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
