"""
================================================================================
 power_vs_steering_direct_vs_indirect.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 08 — Quick analysis: do POWER muscles and STEERING muscles rely
            differently on direct vs. indirect (via premotor IN) DN drive?
STATUS    : Deliberately lean. Reuses steps 01 and 05's exact math
            (direct input fraction, indirect fraction) — no new metric,
            just a different aggregation (by muscle_type instead of by
            individual muscle). Self-contained per this pipeline's
            standalone-script convention.

--------------------------------------------------------------------------------
THE QUESTION
--------------------------------------------------------------------------------
Prior literature (Cheong et al. and others) splits wing muscles into POWER
(the big flight-power muscles: DLM, DVM) and STEERING (basalar, haltere,
axillary, pleurosternal — the fine-control muscles). That split is already
in this project's curated data: `motor_pools.csv`'s `muscle_type` column
(5 power muscles, 13 steering muscles) — no new classification needed.

The question: for a muscle's total DN-attributable drive, how much comes
DIRECTLY (DN->MN, step 01) vs INDIRECTLY (DN->premotor IN->MN, step 05)? And
does that split differ systematically between power and steering muscles —
i.e. does one type lean more on a few strong direct pathways ("sparse")
while the other leans more on the indirect, IN-mediated route?

--------------------------------------------------------------------------------
METRICS (per muscle, then aggregated by muscle_type)
--------------------------------------------------------------------------------
  direct_total_fraction   = sum over all DN groups of F_in(DN->muscle)     (step 01)
  indirect_total_fraction = sum over all DN groups of F_indirect(DN->muscle) (step 05)
  direct_share            = direct_total_fraction /
                             (direct_total_fraction + indirect_total_fraction)

`direct_share` is the actual answer to "which route does this muscle lean
on" in one number per muscle, bounded in [0, 1] by construction (it's a
share of the SUM, not of either component's own scale) — 1.0 = entirely
direct, 0.0 = entirely indirect. It sidesteps the fact that
indirect_total_fraction isn't bounded by 1 the way direct_total_fraction is
(see step 05's docstring) by only ever comparing the two AGAINST EACH OTHER,
never treating either as an absolute probability.

Power and steering muscles are compared on `direct_share`'s distribution —
mean/median plus every individual muscle's value (only 5 + 13 data points,
small enough to just look at directly rather than reach for a formal test).

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
  data/raw/connections_princeton.csv           columns: pre_root_id, post_root_id,
                                                 neuropil, syn_count, nt_type
  data/raw/neurons.csv                          columns incl. Root ID, Super Class,
                                                 Primary Cell Type
  data/processed/motor_pools/motor_pools.csv    columns: muscle, muscle_type,
                                                 muscle_group, motor_neuron_ids

OUTPUTS  (written to results/power_vs_steering_direct_vs_indirect/)
  muscle_direct_vs_indirect.csv     per muscle: muscle_type, direct_total_fraction,
                                     indirect_total_fraction, direct_share
  muscle_type_summary.csv           per muscle_type: n_muscles, mean/median of each metric
  figure_power_vs_steering.png      grouped bar (mean direct vs indirect per type)
                                     + per-muscle strip plot of direct_share
  run_summary.txt

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/power_vs_steering_direct_vs_indirect.py             # real data
  python src/power_vs_steering_direct_vs_indirect.py --selftest   # synthetic check

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "power_vs_steering_direct_vs_indirect"

CONNECTIONS_CSV = DATA_DIR / "raw" / "connections_princeton.csv"
NEURONS_CSV     = DATA_DIR / "raw" / "neurons.csv"
MOTOR_POOLS_CSV = DATA_DIR / "processed" / "motor_pools" / "motor_pools.csv"


# =============================================================================
# COLUMN NAMES
# =============================================================================
COL_ROOT_ID     = "Root ID"
COL_SUPER_CLASS = "Super Class"
COL_CELL_TYPE   = "Primary Cell Type"

CONN_SOURCE = "pre_root_id"
CONN_TARGET = "post_root_id"
CONN_WEIGHT = "syn_count"

DN_SUPER_CLASS = "descending"
IN_SUPER_CLASS = "ventral_nerve_cord_intrinsic"


# =============================================================================
# PARAMETERS
# =============================================================================
KEEP_UNTYPED_DNS = True
KEEP_UNTYPED_INS = True

# --- Figure style: validated palette (dataviz skill's references/palette.md) ---
# muscle_type is a 2-category nominal (power/steering, no inherent order) —
# fixed hue-order assignment: slot 1 (blue) power, slot 2 (orange) steering.
CAT_BLUE, CAT_ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_HAIRLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
MUSCLE_TYPE_COLORS = {"power": CAT_BLUE, "steering": CAT_ORANGE}


# =============================================================================
# CORE COMPUTATION  — identical to steps 01/05 (see those scripts for the
# full rationale on each formula); reused here rather than imported, same
# standalone-script convention as the rest of this pipeline.
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


def build_muscle_type_map(motor_pools: pd.DataFrame) -> pd.Series:
    """muscle -> muscle_type ('power' / 'steering'), straight from the
    curated motor_pools.csv — no new classification, already in the data."""
    return pd.Series(motor_pools["muscle_type"].values, index=motor_pools["muscle"].values,
                     name="muscle_type")


def build_in_groups(neurons: pd.DataFrame, connections: pd.DataFrame, mn_ids: set) -> pd.Series:
    candidates = neurons[neurons[COL_SUPER_CLASS] == IN_SUPER_CLASS].copy()
    candidate_ids = set(candidates[COL_ROOT_ID])
    premotor_mask = (connections[CONN_SOURCE].isin(candidate_ids) &
                     connections[CONN_TARGET].isin(mn_ids))
    premotor_ids = set(connections.loc[premotor_mask, CONN_SOURCE].unique())
    ins = candidates[candidates[COL_ROOT_ID].isin(premotor_ids)].copy()

    cell_type = ins[COL_CELL_TYPE]
    has_type = cell_type.notna() & (cell_type.astype(str).str.strip() != "")
    labels = cell_type.astype("object").copy()
    if KEEP_UNTYPED_INS:
        labels.loc[~has_type] = "untyped_" + ins.loc[~has_type, COL_ROOT_ID].astype(str)
    else:
        labels.loc[~has_type] = np.nan
    group_map = pd.Series(labels.values, index=ins[COL_ROOT_ID].values, name="in_group")
    return group_map.dropna()


def compute_total_input(connections: pd.DataFrame, target_ids) -> pd.Series:
    total_input_all = connections.groupby(CONN_TARGET)[CONN_WEIGHT].sum()
    return total_input_all.reindex(list(target_ids)).fillna(0.0)


def compute_groupwise_fraction(connections: pd.DataFrame,
                               source_group_map: pd.Series,
                               target_group_map: pd.Series,
                               target_total_input: pd.Series,
                               source_col: str, target_col: str) -> pd.DataFrame:
    """Identical generic function to steps 05/06."""
    source_ids = set(source_group_map.index)
    target_ids = set(target_group_map.index)
    mask = connections[CONN_SOURCE].isin(source_ids) & connections[CONN_TARGET].isin(target_ids)
    edges = connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()
    edges[source_col] = edges[CONN_SOURCE].map(source_group_map)
    edges[target_col] = edges[CONN_TARGET].map(target_group_map)
    numerator = (edges.groupby([source_col, target_col])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())
    denom_per_target_group = (target_total_input.groupby(target_group_map).sum()
                              .rename("denominator_input"))
    out = numerator.merge(denom_per_target_group, left_on=target_col, right_index=True, how="left")
    out["input_fraction"] = out["numerator_synapses"] / out["denominator_input"]
    return out.sort_values("input_fraction", ascending=False).reset_index(drop=True)


def compute_indirect(dn_in_long: pd.DataFrame, in_mn_long: pd.DataFrame) -> pd.DataFrame:
    """F_indirect via matrix product — identical to step 05."""
    dn_in_matrix = dn_in_long.pivot_table(index="dn_group", columns="in_group",
                                          values="input_fraction", aggfunc="sum", fill_value=0.0)
    in_mn_matrix = in_mn_long.pivot_table(index="in_group", columns="muscle",
                                          values="input_fraction", aggfunc="sum", fill_value=0.0)
    all_in_groups = sorted(set(dn_in_matrix.columns) | set(in_mn_matrix.index))
    dn_in_aligned = dn_in_matrix.reindex(columns=all_in_groups, fill_value=0.0)
    in_mn_aligned = in_mn_matrix.reindex(index=all_in_groups, fill_value=0.0)
    indirect_matrix = dn_in_aligned.dot(in_mn_aligned)
    indirect_matrix.index.name = "dn_group"
    indirect_matrix.columns.name = "muscle"
    long_df = indirect_matrix.stack().reset_index()
    long_df.columns = ["dn_group", "muscle", "indirect_fraction"]
    return long_df[long_df["indirect_fraction"] > 0].reset_index(drop=True)


# =============================================================================
# THE NEW PART: aggregate direct vs indirect by muscle_type
# =============================================================================
def summarise_per_muscle(direct_long: pd.DataFrame, indirect_long: pd.DataFrame,
                         muscle_type_map: pd.Series, all_muscles: pd.Index) -> pd.DataFrame:
    direct_total = (direct_long.groupby("muscle")["input_fraction"].sum()
                    .rename("direct_total_fraction"))
    indirect_total = (indirect_long.groupby("muscle")["indirect_fraction"].sum()
                      .rename("indirect_total_fraction"))

    summary = pd.DataFrame(index=all_muscles)
    summary = summary.join(direct_total).join(indirect_total).fillna(0.0)
    summary["muscle_type"] = muscle_type_map.reindex(all_muscles)
    combined = summary["direct_total_fraction"] + summary["indirect_total_fraction"]
    summary["direct_share"] = np.where(combined > 0, summary["direct_total_fraction"] / combined, np.nan)
    return summary[["muscle_type", "direct_total_fraction", "indirect_total_fraction", "direct_share"]] \
        .sort_values(["muscle_type", "direct_share"], ascending=[True, False])


def summarise_per_type(per_muscle: pd.DataFrame) -> pd.DataFrame:
    g = per_muscle.groupby("muscle_type")
    return pd.DataFrame({
        "n_muscles": g.size(),
        "mean_direct_total_fraction": g["direct_total_fraction"].mean(),
        "mean_indirect_total_fraction": g["indirect_total_fraction"].mean(),
        "mean_direct_share": g["direct_share"].mean(),
        "median_direct_share": g["direct_share"].median(),
    }).sort_values("mean_direct_share", ascending=False)


# =============================================================================
# FIGURE  — deliberately simple: 2 panels, fixed at 16:9. This is a "quick
#           analysis" — no need for the heavier multi-panel treatment
#           earlier steps used.
# =============================================================================
def make_figure(per_muscle: pd.DataFrame, per_type: pd.DataFrame,
                out_path: Path, demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 6.0), dpi=150)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.07, right=0.98, wspace=0.28)

    types = list(per_type.index)

    # --- Left: grouped bar, mean direct vs indirect fraction per type ------
    # Colors here encode the MEASURE (direct/indirect), consistent with
    # every other figure in this pipeline that shows both side by side
    # (step 06's paired bars use the same blue=direct, orange=indirect).
    # No bar border — the gap between the two bars in a group does the
    # separating.
    x = np.arange(len(types))
    width = 0.35
    ax1.bar(x - width / 2, per_type["mean_direct_total_fraction"], width,
           label="mean direct", color=CAT_BLUE)
    ax1.bar(x + width / 2, per_type["mean_indirect_total_fraction"], width,
           label="mean indirect", color=CAT_ORANGE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}\n(n={int(per_type.loc[t, 'n_muscles'])})" for t in types],
                        fontsize=10, color=INK_SOFT)
    ax1.tick_params(axis="y", colors=AXIS_LINE)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_color(AXIS_LINE)
    ax1.set_ylabel("total DN-attributable fraction", fontsize=9.5, fontweight="bold", color=INK)
    ax1.set_title("mean direct vs. indirect strength", fontsize=10.5, color=INK)
    ax1.legend(fontsize=8.5, frameon=False, labelcolor=INK_SOFT)

    # --- Right: per-muscle direct_share, split by type ----------------------
    # Colors here encode the CATEGORICAL identity (power/steering), fixed
    # order per MUSCLE_TYPE_COLORS. Every point is labelled deliberately —
    # with only 5 + 13 = 18 muscles total this is a small, fully-inspectable
    # set, not the dense-scatter case the "never label every point" rule
    # targets; the whole point of this panel is to show each one directly.
    for i, t in enumerate(types):
        sub = per_muscle[per_muscle["muscle_type"] == t]
        jitter = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.15
        ax2.scatter(np.full(len(sub), i) + jitter, sub["direct_share"],
                   color=MUSCLE_TYPE_COLORS.get(t, INK_MUTED), edgecolor="white", linewidth=0.6,
                   s=50, zorder=3, label=t)
        ax2.hlines(sub["direct_share"].mean(), i - 0.2, i + 0.2,
                  color=INK, linewidth=2, zorder=4)
        for muscle, val in sub["direct_share"].items():
            ax2.annotate(muscle, (i + 0.16, val), fontsize=6, va="center", color=INK_SOFT)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, fontsize=10, color=INK_SOFT)
    ax2.set_xlim(-0.5, len(types) - 0.5)
    ax2.set_ylim(-0.05, 1.05)
    # Dashed deliberately (not a plain gridline) — it marks the genuine
    # direct==indirect threshold, and dashing is the right signal for "this
    # is a meaningful reference value," not decoration.
    ax2.axhline(0.5, color=AXIS_LINE, linestyle="--", linewidth=0.8, zorder=0)
    ax2.tick_params(axis="y", colors=AXIS_LINE)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax2.spines[spine].set_color(AXIS_LINE)
    ax2.set_ylabel("direct_share  (1.0 = fully direct, 0.0 = fully indirect)",
                  fontsize=9, fontweight="bold", color=INK)
    ax2.set_title("per-muscle direct_share (dark bar = type mean)", fontsize=10.5, color=INK)

    if demo:
        fig.text(0.99, 0.99, "DEMO / SELF-TEST DATA — not real results",
                 fontsize=9, color="#B00020", fontweight="bold", va="top", ha="right")

    fig.suptitle("Power vs. steering muscles: direct vs. indirect DN drive",
                 fontsize=13, fontweight="bold", color=INK)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
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
    muscle_type_map = build_muscle_type_map(motor_pools)
    all_muscles = pd.Index(sorted(mn_group_map.unique()))
    in_group_map = build_in_groups(neurons, connections, set(mn_group_map.index))

    mn_total_input = compute_total_input(connections, mn_group_map.index.values)
    in_total_input = compute_total_input(connections, in_group_map.index.values)

    direct_long = compute_groupwise_fraction(
        connections, dn_group_map, mn_group_map, mn_total_input, "dn_group", "muscle")
    dn_in_long = compute_groupwise_fraction(
        connections, dn_group_map, in_group_map, in_total_input, "dn_group", "in_group")
    in_mn_long = compute_groupwise_fraction(
        connections, in_group_map, mn_group_map, mn_total_input, "in_group", "muscle")
    indirect_long = compute_indirect(dn_in_long, in_mn_long)

    per_muscle = summarise_per_muscle(direct_long, indirect_long, muscle_type_map, all_muscles)
    per_type = summarise_per_type(per_muscle)

    print(f"\nMuscles: {len(all_muscles)} ({dict(muscle_type_map.value_counts())})")
    print("\n--- Per-muscle direct vs indirect ---")
    print(per_muscle.round(4).to_string())
    print("\n--- Per-type summary ---")
    print(per_type.round(4).to_string())

    leader = per_type["mean_direct_share"].idxmax()
    other = [t for t in per_type.index if t != leader]
    if other:
        gap = per_type.loc[leader, "mean_direct_share"] - per_type.loc[other[0], "mean_direct_share"]
        print(f"\n'{leader}' muscles lean more direct (mean direct_share "
              f"{per_type.loc[leader, 'mean_direct_share']:.3f} vs "
              f"{per_type.loc[other[0], 'mean_direct_share']:.3f} for '{other[0]}', "
              f"gap = {gap:.3f}).")

    per_muscle.to_csv(out_dir / "muscle_direct_vs_indirect.csv")
    per_type.to_csv(out_dir / "muscle_type_summary.csv")

    fig_path = out_dir / "figure_power_vs_steering.png"
    make_figure(per_muscle, per_type, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Power vs. steering: direct vs. indirect DN drive — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"muscles: {len(all_muscles)} {dict(muscle_type_map.value_counts())}\n\n")
        fh.write("per-muscle:\n")
        fh.write(per_muscle.round(4).to_string())
        fh.write("\n\nper-type summary:\n")
        fh.write(per_type.round(4).to_string())
        fh.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return per_muscle, per_type


# =============================================================================
# SELF-TEST
# =============================================================================
def selftest():
    """
    muscleA (power) is driven ENTIRELY directly; muscleB (steering) is
    driven ENTIRELY indirectly — a clean, hand-verifiable contrast:

      101 (DNx) -> 201 (muscleA, power), DIRECT: 20 synapses
      102 (DNy) -> 301 (IN, premotor)          : 20 synapses
      301 (IN)  -> 202 (muscleB, steering)     : 15 synapses
      997 (other) -> 201                       : 10 synapses (padding)
      998 (other) -> 202                       : 10 synapses (padding)
      999 (other) -> 301                       : 10 synapses (padding)

    TotalInput(201) = 20+10 = 30      TotalInput(202) = 15+10 = 25
    TotalInput(301) = 20+10 = 30

    direct_total_fraction(muscleA)   = 20/30 = 0.666667   (only direct edge)
    indirect_total_fraction(muscleA) = 0                  (no path via IN 301)
    direct_total_fraction(muscleB)   = 0                  (no direct edge)
    indirect_total_fraction(muscleB) = (20/30)*(15/25) = 0.4

    direct_share(muscleA) = 0.666667 / 0.666667 = 1.0   (fully direct)
    direct_share(muscleB) = 0 / 0.4              = 0.0   (fully indirect)

    So this toy example should show power=fully direct, steering=fully
    indirect — an extreme, unambiguous case to verify the aggregation
    logic, not a claim about which way the real data will go.
    """
    connections = pd.DataFrame({
        CONN_SOURCE: [101, 102, 301, 997, 998, 999],
        CONN_TARGET: [201, 301, 202, 201, 202, 301],
        CONN_WEIGHT: [20,  20,  15,  10,  10,  10],
    })
    neurons = pd.DataFrame({
        COL_ROOT_ID:     [101, 102, 201, 202, 301, 997, 998, 999],
        COL_SUPER_CLASS: ["descending", "descending", "motor", "motor",
                          "ventral_nerve_cord_intrinsic", "sensory", "sensory", "sensory"],
        COL_CELL_TYPE:   ["DNx", "DNy", None, None, "INa", None, None, None],
    })
    motor_pools = pd.DataFrame({
        "muscle": ["muscleA", "muscleB"],
        "muscle_type": ["power", "steering"],
        "motor_neuron_ids": ["[201]", "[202]"],
    })

    dn_map = build_dn_groups(neurons)
    mn_map = build_mn_groups(motor_pools)
    muscle_type_map = build_muscle_type_map(motor_pools)
    all_muscles = pd.Index(["muscleA", "muscleB"])
    in_map = build_in_groups(neurons, connections, set(mn_map.index))
    mn_tot = compute_total_input(connections, mn_map.index.values)
    in_tot = compute_total_input(connections, in_map.index.values)

    direct_long = compute_groupwise_fraction(connections, dn_map, mn_map, mn_tot, "dn_group", "muscle")
    dn_in_long = compute_groupwise_fraction(connections, dn_map, in_map, in_tot, "dn_group", "in_group")
    in_mn_long = compute_groupwise_fraction(connections, in_map, mn_map, mn_tot, "in_group", "muscle")
    indirect_long = compute_indirect(dn_in_long, in_mn_long)

    per_muscle = summarise_per_muscle(direct_long, indirect_long, muscle_type_map, all_muscles)
    print("Per-muscle summary:")
    print(per_muscle.to_string())

    assert round(per_muscle.loc["muscleA", "direct_total_fraction"], 6) == round(2 / 3, 6)
    assert per_muscle.loc["muscleA", "indirect_total_fraction"] == 0.0
    assert round(per_muscle.loc["muscleA", "direct_share"], 6) == 1.0
    assert per_muscle.loc["muscleB", "direct_total_fraction"] == 0.0
    assert round(per_muscle.loc["muscleB", "indirect_total_fraction"], 6) == 0.4
    assert round(per_muscle.loc["muscleB", "direct_share"], 6) == 0.0

    per_type = summarise_per_type(per_muscle)
    print("\nPer-type summary:")
    print(per_type.to_string())
    assert round(per_type.loc["power", "mean_direct_share"], 6) == 1.0
    assert round(per_type.loc["steering", "mean_direct_share"], 6) == 0.0

    print("\n[OK] direct/indirect totals, direct_share, and the power-vs-steering "
          "aggregation all match hand calc.")

    demo_dir = Path("/tmp/power_vs_steering_selftest")
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
