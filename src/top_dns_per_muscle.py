"""
================================================================================
 top_dns_per_muscle.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 04 — Reporting view on top of step 01: for each wing muscle, the
            top N DN groups ranked by direct input fraction.
STATUS    : Not a new metric — reuses step 01's exact groupwise input-fraction
            computation (build_dn_groups/build_mn_groups/compute_mn_total_input/
            compute_groupwise_input_fraction are identical to that script) and
            adds a per-muscle top-N ranking + leaderboard figure on top. Kept
            self-contained (recomputes from raw data rather than reading step
            01's saved CSV) for the same reason every script in this pipeline
            is standalone — see CLAUDE.md §4 on why a shared/config-driven
            approach was avoided.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT ANSWERS
--------------------------------------------------------------------------------
"For each wing muscle, which DN types supply the most direct input, and how
much?" — the human-readable leaderboard version of step 01's DN-group x
muscle matrix.

RANKED BY INPUT FRACTION, NOT RAW SYNAPSE COUNTS
   Per this project's core rule (CLAUDE.md §2): connection strength is input
   fraction, never a raw synapse count. "Top 5 DNs by input" here means top 5
   by `input_fraction` (synapses(DN group -> muscle) / muscle's TOTAL input
   across the whole connectome) — the same target's-eye influence measure
   step 01 established, not by raw synapse count, which conflates "strong
   connection" with "big neuron" (see step 01's docstring for the full
   argument). Raw synapse counts are still included as a supplementary
   column for context, never as the ranking criterion or the reported
   strength.

WHY A FIXED-SIZE, ALL-MUSCLES FIGURE WORKS HERE (UNLIKE STEPS 01-03)
   Steps 01-03's heatmaps had to cap DN-group rows (TOP_N_ROWS) because the
   number of DN groups clearing threshold varies and can run into the
   hundreds. Here there is no such problem: there are always exactly 18 wing
   muscles (however many motor_pools.csv defines), a small fixed number, so
   the figure can show ALL of them as one small-multiples grid (one mini
   bar chart per muscle) without any truncation logic.

--------------------------------------------------------------------------------
GROUPING  — identical to step 01
--------------------------------------------------------------------------------
  * DN groups  = shared 'Primary Cell Type'. Untyped DNs kept as singletons.
  * MN groups  = muscles / motor pools, from data/processed/motor_pools/motor_pools.csv.

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
  data/raw/connections_princeton.csv           columns: pre_root_id, post_root_id,
                                                 neuropil, syn_count, nt_type
  data/raw/neurons.csv                          columns incl. Root ID, Super Class,
                                                 Primary Cell Type
  data/processed/motor_pools/motor_pools.csv    columns: muscle, motor_neuron_ids

OUTPUTS  (written to results/top_dns_per_muscle/)
  top_dns_per_muscle_long.csv    one row per (muscle, rank): dn_group, input_fraction,
                                  numerator_synapses, denominator_input
  top_dns_per_muscle_wide.csv    one row per muscle: rank1_dn_group, rank1_input_fraction,
                                  ... rankN_*, cumulative_topN_fraction
  figure_top_dns_per_muscle.png  small-multiples grid, one mini bar chart per muscle
  run_summary.txt                counts, parameters

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/top_dns_per_muscle.py             # real data
  python src/top_dns_per_muscle.py --selftest   # synthetic check

Author: (you).  Reviewed logic, hand-checked on synthetic data.
================================================================================
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "top_dns_per_muscle"

CONNECTIONS_CSV = DATA_DIR / "raw" / "connections_princeton.csv"
NEURONS_CSV     = DATA_DIR / "raw" / "neurons.csv"
MOTOR_POOLS_CSV = DATA_DIR / "processed" / "motor_pools" / "motor_pools.csv"


# =============================================================================
# COLUMN NAMES  — identical to step 01.
# =============================================================================
COL_ROOT_ID     = "Root ID"
COL_SUPER_CLASS = "Super Class"
COL_CELL_TYPE   = "Primary Cell Type"

CONN_SOURCE = "pre_root_id"
CONN_TARGET = "post_root_id"
CONN_WEIGHT = "syn_count"

DN_SUPER_CLASS = "descending"


# =============================================================================
# PARAMETERS
# =============================================================================
KEEP_UNTYPED_DNS = True

# The actual ask: top N DNs per muscle. A labelled constant so it's a
# one-line change if you want top 3 or top 10 instead.
TOP_N = 5


# =============================================================================
# CORE COMPUTATION  — identical to step 01 (see that script for full
# rationale on grouping and the input-fraction formula).
# =============================================================================
def build_dn_groups(neurons: pd.DataFrame) -> pd.Series:
    dns = neurons[neurons[COL_SUPER_CLASS] == DN_SUPER_CLASS].copy()
    cell_type = dns[COL_CELL_TYPE]
    has_type = cell_type.notna() & (cell_type.astype(str).str.strip() != "")
    labels = cell_type.astype("object").copy()
    if KEEP_UNTYPED_DNS:
        labels.loc[~has_type] = "untyped_" + dns.loc[~has_type, COL_ROOT_ID].astype(str)
    else:
        labels.loc[~has_type] = np.nan
    group_map = pd.Series(labels.values, index=dns[COL_ROOT_ID].values, name="dn_group")
    return group_map.dropna()


def build_mn_groups(motor_pools: pd.DataFrame) -> pd.Series:
    mapping = {}
    for _, row in motor_pools.iterrows():
        muscle = row["muscle"]
        mn_ids = ast.literal_eval(row["motor_neuron_ids"]) \
            if isinstance(row["motor_neuron_ids"], str) else row["motor_neuron_ids"]
        for mn_id in mn_ids:
            mapping[mn_id] = muscle
    return pd.Series(mapping, name="muscle")


def compute_mn_total_input(connections: pd.DataFrame,
                           mn_ids: np.ndarray) -> pd.Series:
    total_input_all = connections.groupby(CONN_TARGET)[CONN_WEIGHT].sum()
    return total_input_all.reindex(mn_ids).fillna(0.0)


def compute_groupwise_input_fraction(connections: pd.DataFrame,
                                     dn_group_map: pd.Series,
                                     mn_group_map: pd.Series,
                                     mn_total_input: pd.Series) -> pd.DataFrame:
    dn_ids = set(dn_group_map.index)
    mn_ids = set(mn_group_map.index)

    mask = connections[CONN_SOURCE].isin(dn_ids) & connections[CONN_TARGET].isin(mn_ids)
    dn_mn = connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()
    dn_mn["dn_group"] = dn_mn[CONN_SOURCE].map(dn_group_map)
    dn_mn["muscle"]   = dn_mn[CONN_TARGET].map(mn_group_map)

    numerator = (dn_mn.groupby(["dn_group", "muscle"])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())

    muscle_of_mn = mn_group_map
    denom_per_muscle = (mn_total_input.groupby(muscle_of_mn).sum()
                        .rename("denominator_input"))

    out = numerator.merge(denom_per_muscle, left_on="muscle", right_index=True, how="left")
    out["input_fraction"] = out["numerator_synapses"] / out["denominator_input"]
    out = out.sort_values("input_fraction", ascending=False).reset_index(drop=True)
    return out


# =============================================================================
# THE NEW PART: top-N ranking per muscle
# =============================================================================
def top_n_per_muscle(long_df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """
    Return long_df restricted to, for each muscle, its top `n` rows by
    input_fraction, with a 1-indexed `rank` column added. Muscles with fewer
    than `n` contributing DN groups simply get however many they have — no
    padding, nothing invented.
    """
    ranked = long_df.sort_values(["muscle", "input_fraction"], ascending=[True, False])
    top = ranked.groupby("muscle", sort=False).head(n).copy()
    top["rank"] = top.groupby("muscle").cumcount() + 1
    cols = ["muscle", "rank", "dn_group", "input_fraction",
            "numerator_synapses", "denominator_input"]
    return top[cols].reset_index(drop=True)


def make_wide_table(top_df: pd.DataFrame, all_muscles: pd.Index, n: int = TOP_N) -> pd.DataFrame:
    """
    One row per muscle (including muscles with zero direct DN input at all,
    so nothing is silently missing), columns rank1_dn_group, rank1_input_fraction,
    ... rankN_*, plus cumulative_topN_fraction (sum of whatever ranks exist —
    <= n if a muscle has fewer than n contributing DN groups) and denominator_input.
    """
    rows = []
    for muscle in all_muscles:
        sub = top_df[top_df["muscle"] == muscle].sort_values("rank")
        row = {"muscle": muscle}
        denom = sub["denominator_input"].iloc[0] if len(sub) else np.nan
        row["denominator_input"] = denom
        cum = 0.0
        for r in range(1, n + 1):
            match = sub[sub["rank"] == r]
            if len(match):
                row[f"rank{r}_dn_group"] = match["dn_group"].iloc[0]
                row[f"rank{r}_input_fraction"] = match["input_fraction"].iloc[0]
                cum += match["input_fraction"].iloc[0]
            else:
                row[f"rank{r}_dn_group"] = None
                row[f"rank{r}_input_fraction"] = 0.0
        row[f"cumulative_top{n}_fraction"] = cum
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"cumulative_top{n}_fraction", ascending=False)


# =============================================================================
# FIGURE  — small multiples, one mini bar chart per muscle. Fixed at 16:9
#           (13.33 x 7.5in); no row-count capping needed since the muscle
#           count is small and fixed (unlike steps 01-03's DN-group heatmaps).
# =============================================================================
def make_figure(top_df: pd.DataFrame, all_muscles: pd.Index, out_path: Path,
                n: int = TOP_N, demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    n_muscles = len(all_muscles)
    ncols = 6
    nrows = math.ceil(n_muscles / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13.33, 7.5), dpi=150)
    axes = np.atleast_2d(axes)

    for i, muscle in enumerate(all_muscles):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = top_df[top_df["muscle"] == muscle].sort_values("rank")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "no direct\nDN input", ha="center", va="center", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            sub = sub.iloc[::-1]  # strongest at top of the horizontal bar chart
            ax.barh(range(len(sub)), sub["input_fraction"].values,
                    color="#3B6EA5", edgecolor="black", linewidth=0.3)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub["dn_group"].values, fontsize=5.5)
            ax.tick_params(axis="x", labelsize=5.5)
            ax.set_xlim(0, max(sub["input_fraction"].max() * 1.15, 0.001))
        ax.set_title(str(muscle), fontsize=8, fontweight="bold")

    # Blank out any unused grid cells (n_muscles may not fill nrows*ncols exactly).
    for i in range(n_muscles, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if demo:
        fig.text(0.99, 0.99, "DEMO / SELF-TEST DATA — not real results",
                 fontsize=9, color="#B00020", fontweight="bold", va="top", ha="right")

    fig.suptitle(f"Top {n} DN groups by direct input fraction, per wing muscle",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, 0.95,
             "bars = groupwise input fraction $F_{\\mathrm{in}}$ (step 01) — "
             "NOT raw synapse count", fontsize=8, color="#444444", ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


# =============================================================================
# I/O + ORCHESTRATION
# =============================================================================
def load_inputs():
    for p in (CONNECTIONS_CSV, NEURONS_CSV, MOTOR_POOLS_CSV):
        if not p.exists():
            sys.exit(f"[ERROR] Required input not found: {p}\n"
                     f"        Edit the PATHS block at the top of this script.")
    print(f"Loading connections : {CONNECTIONS_CSV}")
    connections = pd.read_csv(CONNECTIONS_CSV)
    print(f"Loading neurons     : {NEURONS_CSV}")
    neurons = pd.read_csv(NEURONS_CSV)
    print(f"Loading motor pools : {MOTOR_POOLS_CSV}")
    motor_pools = pd.read_csv(MOTOR_POOLS_CSV)
    return connections, neurons, motor_pools


def run(connections, neurons, motor_pools, out_dir: Path, demo: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    dn_group_map = build_dn_groups(neurons)
    mn_group_map = build_mn_groups(motor_pools)
    all_muscles = pd.Index(sorted(mn_group_map.unique()))

    print(f"\nDN neurons grouped : {len(dn_group_map):,} DNs -> {dn_group_map.nunique():,} groups")
    print(f"Wing MNs           : {len(mn_group_map):,} MNs -> {len(all_muscles):,} muscles")

    mn_total_input = compute_mn_total_input(connections, mn_group_map.index.values)
    long_df = compute_groupwise_input_fraction(connections, dn_group_map, mn_group_map, mn_total_input)

    top_df = top_n_per_muscle(long_df, TOP_N)
    wide_df = make_wide_table(top_df, all_muscles, TOP_N)

    # reindex over ALL muscles (not just ones in top_df) so a muscle with
    # zero contributing DN groups counts as "0 < TOP_N", not silently omitted.
    n_contributing = top_df.groupby("muscle")["rank"].max().reindex(all_muscles).fillna(0)
    n_muscles_short = int((n_contributing < TOP_N).sum())
    print(f"\nMuscles with fewer than {TOP_N} contributing DN groups: {n_muscles_short} "
          f"of {len(all_muscles)}")

    print(f"\n--- Top {TOP_N} DNs per muscle ---")
    for muscle in all_muscles:
        sub = top_df[top_df["muscle"] == muscle]
        if len(sub) == 0:
            print(f"  {muscle}: (no direct DN input at all)")
            continue
        parts = [f"{r.dn_group} ({r.input_fraction:.1%})" for r in sub.itertuples()]
        print(f"  {muscle}: " + ", ".join(parts))

    top_df.to_csv(out_dir / "top_dns_per_muscle_long.csv", index=False)
    wide_df.to_csv(out_dir / "top_dns_per_muscle_wide.csv", index=False)

    fig_path = out_dir / "figure_top_dns_per_muscle.png"
    make_figure(top_df, all_muscles, fig_path, TOP_N, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Top-N DNs per wing muscle, by direct input fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"top N                        : {TOP_N}\n")
        fh.write(f"keep untyped DNs             : {KEEP_UNTYPED_DNS}\n")
        fh.write(f"DN neurons / groups          : {len(dn_group_map)} / {dn_group_map.nunique()}\n")
        fh.write(f"wing MNs / muscles           : {len(mn_group_map)} / {len(all_muscles)}\n")
        fh.write(f"muscles with < {TOP_N} contributing DN groups : {n_muscles_short}\n\n")
        fh.write(f"top {TOP_N} DNs per muscle:\n")
        for muscle in all_muscles:
            sub = top_df[top_df["muscle"] == muscle]
            if len(sub) == 0:
                fh.write(f"  {muscle}: (no direct DN input at all)\n")
                continue
            parts = [f"{r.dn_group} ({r.input_fraction:.1%})" for r in sub.itertuples()]
            fh.write(f"  {muscle}: " + ", ".join(parts) + "\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return top_df, wide_df


# =============================================================================
# SELF-TEST
# =============================================================================
def selftest():
    """
    Step 01's exact synthetic connectome. Each muscle only has ONE
    contributing DN group here (muscleA<-DNx, muscleB<-DNy), so with
    TOP_N=5 this exercises the "fewer than N available" path — the
    important thing to check is that it shows exactly what exists (1 row
    per muscle) rather than erroring or padding with fake rows.
    """
    connections = pd.DataFrame({
        CONN_SOURCE: [101, 101, 102, 103, 999, 999, 999],
        CONN_TARGET: [201, 202, 201, 203, 201, 202, 203],
        CONN_WEIGHT: [10,  5,   5,   20,  30,  15,  30],
    })
    neurons = pd.DataFrame({
        COL_ROOT_ID:     [101, 102, 103, 201, 202, 203, 999],
        COL_SUPER_CLASS: ["descending", "descending", "descending",
                          "motor", "motor", "motor", "intrinsic"],
        COL_CELL_TYPE:   ["DNx", "DNx", "DNy", None, None, None, "someIN"],
    })
    motor_pools = pd.DataFrame({
        "muscle": ["muscleA", "muscleB"],
        "motor_neuron_ids": ["[201, 202]", "[203]"],
    })

    dn_map = build_dn_groups(neurons)
    mn_map = build_mn_groups(motor_pools)
    mn_tot = compute_mn_total_input(connections, mn_map.index.values)
    long_df = compute_groupwise_input_fraction(connections, dn_map, mn_map, mn_tot)
    top_df = top_n_per_muscle(long_df, TOP_N)

    print("Self-test top-N table:")
    print(top_df.to_string(index=False))

    assert len(top_df) == 2, f"expected 2 rows (1 per muscle, both under TOP_N), got {len(top_df)}"
    assert set(top_df["muscle"]) == {"muscleA", "muscleB"}
    row_a = top_df[top_df["muscle"] == "muscleA"].iloc[0]
    row_b = top_df[top_df["muscle"] == "muscleB"].iloc[0]
    assert row_a["dn_group"] == "DNx" and round(row_a["input_fraction"], 6) == 0.307692
    assert row_b["dn_group"] == "DNy" and round(row_b["input_fraction"], 6) == 0.4
    assert row_a["rank"] == 1 and row_b["rank"] == 1
    print("\n[OK] top-N ranking matches hand calc, and the 'fewer than N available' "
          "case shows exactly what exists rather than padding or erroring.")

    demo_dir = Path("/tmp/top_dns_per_muscle_selftest")
    run(connections, neurons, motor_pools, demo_dir, demo=True)
    print("\n[OK] full pipeline + figure ran on synthetic data.")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic-data correctness check and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    connections, neurons, motor_pools = load_inputs()
    run(connections, neurons, motor_pools, RESULTS_DIR, demo=False)


if __name__ == "__main__":
    main()
