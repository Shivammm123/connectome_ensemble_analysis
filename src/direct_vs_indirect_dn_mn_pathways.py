"""
================================================================================
 direct_vs_indirect_dn_mn_pathways.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 06 — Anchor indirect connectivity to the sparse, already-trusted
            direct structure, rather than reporting the full dense DN x
            muscle indirect matrix on its own.
STATUS    : Answers a narrower, more interpretable question than step 05
            alone: "for a DN group that already has substantial DIRECT
            input to a muscle, does it ALSO have substantial INDIRECT
            (via premotor IN) input to that SAME muscle?" Self-contained —
            recomputes step 01's direct fraction and step 05's indirect
            fraction from raw data, same standalone-script convention as
            the rest of this pipeline.

--------------------------------------------------------------------------------
WHY THIS STEP EXISTS
--------------------------------------------------------------------------------
Step 05 computes F_indirect for every DN-group x muscle pair reachable
through ANY premotor IN. With ~2,153 premotor INs bridging a sparse 429-edge
direct layer, that indirect layer is combinatorially much denser — many DN
groups end up with a small nonzero F_indirect to many muscles, most of it
not meaningfully interpretable on its own (see step 05's docstring: it's
also not bounded by 1.0 the way F_in is). Reporting that whole dense matrix
risks looking "convincing" without actually being interpretable.

This script anchors the comparison to what's already trusted: step 01's
direct edges. For each (DN group, muscle) pair, it reports F_in and
F_indirect SIDE BY SIDE and classifies the pair into exactly one of:

  * direct_only    — F_in >= THRESHOLD, F_indirect <  THRESHOLD
  * indirect_only  — F_in <  THRESHOLD, F_indirect >= THRESHOLD
  * both           — F_in >= THRESHOLD, F_indirect >= THRESHOLD
  * (pairs below THRESHOLD on both sides are dropped entirely — not
     "neither", just not substantial enough to report either way)

This is a properly-normalised redo of the pilot's old `dn_pathway_strategies.csv`
concept (direct_only / indirect_only / both), which used a raw min_synapses
threshold rather than input fraction — same idea, correct math this time.

--------------------------------------------------------------------------------
TWO RESOLUTIONS REPORTED — DON'T CONFLATE THEM
--------------------------------------------------------------------------------
  1. EDGE level (the primary output): classification per (DN group, muscle)
     pair — "does DNx have both direct and indirect input to THIS muscle."
     This is what answers the question as asked.
  2. DN-GROUP level (a coarser rollup, `dn_group_pathway_strategy.csv`):
     across ALL the muscles a DN group touches (directly or indirectly, not
     necessarily the SAME muscle for both), does it ever use direct-only,
     indirect-only, or both pathway types? This is closer to the pilot's
     original DN-level framing, but is a coarser summary — a DN classified
     as "uses both" here might use direct on one muscle and indirect on a
     completely different one. Don't read it as "this DN's connections are
     each double-confirmed" — check the edge-level table for that.

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
  data/raw/connections_princeton.csv           columns: pre_root_id, post_root_id,
                                                 neuropil, syn_count, nt_type
  data/raw/neurons.csv                          columns incl. Root ID, Super Class,
                                                 Primary Cell Type
  data/processed/motor_pools/motor_pools.csv    columns: muscle, motor_neuron_ids

OUTPUTS  (written to results/direct_vs_indirect_dn_mn_pathways/)
  direct_vs_indirect_long.csv          every (dn_group, muscle) pair with F_in
                                        and/or F_indirect >= THRESHOLD, both
                                        values, and pathway_type
  dn_group_pathway_strategy.csv        DN-group-level rollup (coarser, see above)
  figure_direct_vs_indirect_pathways.png  scatter (F_in vs F_indirect per edge)
                                        + pathway-type count bar
  run_summary.txt                      counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/direct_vs_indirect_dn_mn_pathways.py             # real data
  python src/direct_vs_indirect_dn_mn_pathways.py --selftest   # synthetic check

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "direct_vs_indirect_dn_mn_pathways"

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
# Same THRESHOLD used throughout steps 01-05, applied here to BOTH F_in and
# F_indirect for the direct_only/indirect_only/both classification.
THRESHOLD = 0.01

KEEP_UNTYPED_DNS = True
KEEP_UNTYPED_INS = True

# --- Figure style: validated palette (dataviz skill's references/palette.md) ---
# pathway_type is a genuine categorical (3 identities, no inherent order) —
# fixed hue-order assignment, not hand-picked: slot 1 (blue) direct_only,
# slot 2 (orange) indirect_only, slot 3 (aqua) both.
CAT_BLUE, CAT_ORANGE, CAT_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_HAIRLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
PATHWAY_TYPE_COLORS = {"direct_only": CAT_BLUE, "indirect_only": CAT_ORANGE, "both": CAT_AQUA}


# =============================================================================
# CORE COMPUTATION  — steps 01 and 05's exact math, self-contained here.
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
    """Identical to step 05's generic function — see that script for detail."""
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
# THE NEW PART: anchor indirect to direct, classify, roll up
# =============================================================================
def compare_direct_vs_indirect(direct_long: pd.DataFrame, indirect_long: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join direct and indirect fractions on (dn_group, muscle), keep only
    pairs substantial on at least one side (>= THRESHOLD), and classify.
    """
    direct = direct_long[["dn_group", "muscle", "input_fraction"]].rename(
        columns={"input_fraction": "F_in"})
    indirect = indirect_long[["dn_group", "muscle", "indirect_fraction"]].rename(
        columns={"indirect_fraction": "F_indirect"})

    combined = pd.merge(direct, indirect, on=["dn_group", "muscle"], how="outer")
    combined["F_in"] = combined["F_in"].fillna(0.0)
    combined["F_indirect"] = combined["F_indirect"].fillna(0.0)

    # Compute the flags as columns first, filter on those columns, THEN
    # classify — keeps the masks and the dataframe rows trivially aligned
    # (they're columns of the same object) rather than relying on filtering
    # two separately-indexed boolean Series in lockstep.
    combined["direct_ok"] = combined["F_in"] >= THRESHOLD
    combined["indirect_ok"] = combined["F_indirect"] >= THRESHOLD
    combined = combined[combined["direct_ok"] | combined["indirect_ok"]].copy()

    # default="" (not the implicit 0): NumPy 2.x no longer silently promotes
    # an int default against a string choicelist (TypeError otherwise). Every
    # row here is guaranteed to match one of the three conditions anyway
    # (that's exactly the direct_ok | indirect_ok filter above), so the
    # default never actually gets used — it just has to be dtype-compatible.
    combined["pathway_type"] = np.select(
        [combined["direct_ok"] & combined["indirect_ok"],
         combined["direct_ok"] & ~combined["indirect_ok"],
         ~combined["direct_ok"] & combined["indirect_ok"]],
        ["both", "direct_only", "indirect_only"],
        default="",
    )
    combined = combined.drop(columns=["direct_ok", "indirect_ok"])
    combined = combined.sort_values(["pathway_type", "F_in"], ascending=[True, False]).reset_index(drop=True)
    return combined


def rollup_dn_group_strategy(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Coarser DN-group-level view: across all muscles a DN group appears with
    (in `combined`, i.e. substantial by either measure), what pathway types
    does it ever use? NOT the same claim as the edge-level table — see
    module docstring's "two resolutions" note.
    """
    rows = []
    for dn_group, sub in combined.groupby("dn_group"):
        types_used = set(sub["pathway_type"])
        n_direct_only = int((sub["pathway_type"] == "direct_only").sum())
        n_indirect_only = int((sub["pathway_type"] == "indirect_only").sum())
        n_both = int((sub["pathway_type"] == "both").sum())
        if types_used == {"both"} or (n_both > 0 and len(types_used) == 1):
            overall = "both_on_every_muscle"
        elif "both" in types_used or len(types_used) > 1:
            overall = "mixed_across_muscles"
        elif types_used == {"direct_only"}:
            overall = "direct_only"
        elif types_used == {"indirect_only"}:
            overall = "indirect_only"
        else:
            overall = "mixed_across_muscles"
        rows.append({
            "dn_group": dn_group,
            "n_muscles_direct_only": n_direct_only,
            "n_muscles_indirect_only": n_indirect_only,
            "n_muscles_both": n_both,
            "n_muscles_total": len(sub),
            "overall_strategy": overall,
        })
    return pd.DataFrame(rows).sort_values("n_muscles_total", ascending=False).reset_index(drop=True)


# =============================================================================
# FIGURE  — fixed at 16:9 (13.33 x 7.5in), same convention as steps 01-05.
# =============================================================================
def make_figure(combined: pd.DataFrame, out_path: Path, demo: bool = False) -> None:
    """
    Three side-by-side named panels, one per pathway_type — no scatter, no
    unlabelled dots. A scatter-plus-labels design was tried first and
    dropped: cramming 40 point labels onto one plot next to a second panel
    made everything cramped and hard to read. Three separate horizontal
    bar-of-names panels, each with real room, is more legible and directly
    answers "which DNs, by name" for every category, not just "both".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    MAX_PER_PANEL = 12
    colors = PATHWAY_TYPE_COLORS

    # Added directly to `combined` (not a separate copy) since draw_panel()
    # below is a closure over this exact variable — a separate copy would
    # never actually be seen by it.
    combined = combined.copy()
    combined["_min_side"] = combined[["F_in", "F_indirect"]].min(axis=1)

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    gs = GridSpec(2, 3, height_ratios=[0.8, 5.4], hspace=0.45, wspace=0.55,
                  top=0.88, bottom=0.08, left=0.05, right=0.98)

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    # Explicit short lines rather than one long string — at the full 13.33in
    # figure width with LaTeX math markup, one unbroken line risks running
    # past the edges instead of wrapping (matplotlib text doesn't auto-wrap).
    ax_text.text(0.5, 0.5,
                 f"Each bar = one (DN group $\\rightarrow$ muscle) edge substantial ($\\geq$ {THRESHOLD:g}) by\n"
                 f"direct input fraction $F_{{\\mathrm{{in}}}}$ (step 01) and/or indirect fraction "
                 f"$F_{{\\mathrm{{indirect}}}}$ (step 05).\n"
                 "Anchored to the sparse, already-trusted direct layer — see script docstring for why.",
                 fontsize=9, va="center", ha="center", color=INK_SOFT, linespacing=1.5)
    if demo:
        ax_text.text(0.99, 0.9, "DEMO / SELF-TEST DATA — not real results",
                     fontsize=9, color="#B00020", fontweight="bold", va="top", ha="right")

    def draw_panel(ax, ptype, sort_col, xlabel, paired=False):
        df = combined[combined["pathway_type"] == ptype].copy()
        n_total = len(df)
        df = df.sort_values(sort_col, ascending=False).head(MAX_PER_PANEL)
        title = ptype.replace("_", " ")
        if n_total > MAX_PER_PANEL:
            title += f"  ({MAX_PER_PANEL} of {n_total})"
        else:
            title += f"  (n={n_total})"
        ax.set_title(title, fontsize=10, fontweight="bold", color=colors[ptype])

        if n_total == 0:
            ax.text(0.5, 0.5, "none found", ha="center", va="center",
                    fontsize=9, color=INK_MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return

        df = df.iloc[::-1]  # strongest at top
        labels = [f"{r.dn_group}→{r.muscle}" for r in df.itertuples()]
        y = np.arange(len(df))
        # No border on any bar — spacing (the bar_h gap for paired bars,
        # matplotlib's default width for single bars) separates them.
        if paired:
            bar_h = 0.38
            ax.barh(y + bar_h / 2, df["F_in"].values, height=bar_h,
                   color=colors["direct_only"], label="$F_{in}$")
            ax.barh(y - bar_h / 2, df["F_indirect"].values, height=bar_h,
                   color=colors["indirect_only"], label="$F_{indirect}$")
            ax.legend(fontsize=6.5, loc="lower right", frameon=False, labelcolor=INK_SOFT)
        else:
            ax.barh(y, df[sort_col].values, color=colors[ptype])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5, color=INK_SOFT)
        ax.tick_params(axis="x", colors=AXIS_LINE)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(AXIS_LINE)
        ax.set_xlabel(xlabel, fontsize=8, fontweight="bold", color=INK)

    ax_direct = fig.add_subplot(gs[1, 0])
    draw_panel(ax_direct, "direct_only", "F_in", "$F_{in}$")

    ax_indirect = fig.add_subplot(gs[1, 1])
    draw_panel(ax_indirect, "indirect_only", "F_indirect", "$F_{indirect}$")

    ax_both = fig.add_subplot(gs[1, 2])
    draw_panel(ax_both, "both", "_min_side", "input fraction", paired=True)

    fig.suptitle("Direct vs. indirect DN$\\rightarrow$MN pathways, anchored to the direct layer",
                 fontsize=13, fontweight="bold", y=0.98, color=INK)
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

    combined = compare_direct_vs_indirect(direct_long, indirect_long)
    strategy = rollup_dn_group_strategy(combined)

    print(f"\nDN groups            : {dn_group_map.nunique():,}")
    print(f"Muscles              : {mn_group_map.nunique():,}")
    print(f"IN groups (premotor) : {in_group_map.nunique():,}")
    print(f"\nSubstantial edges (>= {THRESHOLD:g} on F_in and/or F_indirect): {len(combined)}")
    print(combined["pathway_type"].value_counts().to_string())

    n_direct_substantial = int((direct_long["input_fraction"] >= THRESHOLD).sum())
    n_direct_with_indirect = int((combined["pathway_type"] == "both").sum())
    print(f"\nOf {n_direct_substantial} substantial DIRECT edges (step 01), "
          f"{n_direct_with_indirect} ({n_direct_with_indirect / max(n_direct_substantial, 1):.1%}) "
          f"also have substantial indirect reinforcement to the SAME muscle.")

    print(f"\n--- DN-group strategy rollup (top 10 by muscle count) ---")
    print(strategy.head(10).to_string(index=False))

    combined.to_csv(out_dir / "direct_vs_indirect_long.csv", index=False)
    strategy.to_csv(out_dir / "dn_group_pathway_strategy.csv", index=False)

    fig_path = out_dir / "figure_direct_vs_indirect_pathways.png"
    make_figure(combined, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Direct vs. indirect DN->MN pathways — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold                    : {THRESHOLD}\n")
        fh.write(f"DN groups / muscles / IN groups : {dn_group_map.nunique()} / "
                 f"{mn_group_map.nunique()} / {in_group_map.nunique()}\n")
        fh.write(f"substantial edges (either side) : {len(combined)}\n")
        fh.write(combined["pathway_type"].value_counts().to_string())
        fh.write(f"\n\nOf {n_direct_substantial} substantial direct edges, "
                 f"{n_direct_with_indirect} also have substantial indirect reinforcement "
                 f"to the same muscle.\n\n")
        fh.write("DN-group strategy rollup (top 10 by muscle count):\n")
        fh.write(strategy.head(10).to_string(index=False))
        fh.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return combined, strategy


# =============================================================================
# SELF-TEST
# =============================================================================
def selftest():
    """
    Synthetic connectome covering all three pathway types against ONE
    muscle, so the classification logic is unambiguous:

      DNx (101) -> MN 201 direct (5 synapses) AND -> IN 301 (10) -> MN 201 (8)
                   => BOTH direct and indirect to muscleA
      DNy (102) -> MN 201 direct (6 synapses), no IN edges at all
                   => DIRECT ONLY
      DNz (103) -> IN 302 (15) -> MN 201 (4), no direct edge at all
                   => INDIRECT ONLY

    TotalInput(201) = 5+6+8+4+21(other, neuron 997) = 44
    TotalInput(301) = 10+30(other, neuron 999)      = 40
    TotalInput(302) = 15+20(other, neuron 998)      = 35

    F_in(DNx->muscleA)  = 5/44  = 0.113636
    F_in(DNy->muscleA)  = 6/44  = 0.136364
    F(DNx->INa)         = 10/40 = 0.25
    F(INa->muscleA)     = 8/44  = 0.181818
    F_indirect(DNx->muscleA) = 0.25 * 0.181818... = 1/22 = 0.045455
    F(DNz->INb)         = 15/35 = 0.428571
    F(INb->muscleA)     = 4/44  = 0.090909
    F_indirect(DNz->muscleA) = (15/35)*(4/44) = 3/77 = 0.038961

    All three well above THRESHOLD=0.01, so classification is unambiguous:
      DNx -> "both", DNy -> "direct_only", DNz -> "indirect_only"
    """
    connections = pd.DataFrame({
        CONN_SOURCE: [101, 102, 101, 999, 301, 103, 998, 302, 997],
        CONN_TARGET: [201, 201, 301, 301, 201, 302, 302, 201, 201],
        CONN_WEIGHT: [5,   6,   10,  30,  8,   15,  20,  4,   21],
    })
    neurons = pd.DataFrame({
        COL_ROOT_ID:     [101, 102, 103, 201, 301, 302, 999, 998, 997],
        COL_SUPER_CLASS: ["descending", "descending", "descending", "motor",
                          "ventral_nerve_cord_intrinsic", "ventral_nerve_cord_intrinsic",
                          "sensory", "sensory", "sensory"],
        COL_CELL_TYPE:   ["DNx", "DNy", "DNz", None, "INa", "INb", None, None, None],
    })
    motor_pools = pd.DataFrame({
        "muscle": ["muscleA"],
        "motor_neuron_ids": ["[201]"],
    })

    dn_map = build_dn_groups(neurons)
    mn_map = build_mn_groups(motor_pools)
    in_map = build_in_groups(neurons, connections, set(mn_map.index))
    mn_tot = compute_total_input(connections, mn_map.index.values)
    in_tot = compute_total_input(connections, in_map.index.values)

    direct_long = compute_groupwise_fraction(connections, dn_map, mn_map, mn_tot, "dn_group", "muscle")
    dn_in_long = compute_groupwise_fraction(connections, dn_map, in_map, in_tot, "dn_group", "in_group")
    in_mn_long = compute_groupwise_fraction(connections, in_map, mn_map, mn_tot, "in_group", "muscle")
    indirect_long = compute_indirect(dn_in_long, in_mn_long)

    combined = compare_direct_vs_indirect(direct_long, indirect_long)
    print("Combined direct-vs-indirect table:")
    print(combined.to_string(index=False))

    got = {r.dn_group: (round(r.F_in, 6), round(r.F_indirect, 6), r.pathway_type)
           for r in combined.itertuples()}
    exp = {
        "DNx": (round(5 / 44, 6), round(1 / 22, 6), "both"),
        "DNy": (round(6 / 44, 6), 0.0, "direct_only"),
        "DNz": (0.0, round(3 / 77, 6), "indirect_only"),
    }
    assert got == exp, f"\nMISMATCH\n got={got}\n exp={exp}"
    print("\n[OK] F_in, F_indirect, and pathway_type classification all match hand calc "
          "for all three categories (both / direct_only / indirect_only).")

    strategy = rollup_dn_group_strategy(combined)
    print("\nDN-group strategy rollup:")
    print(strategy.to_string(index=False))
    strat_map = dict(zip(strategy["dn_group"], strategy["overall_strategy"]))
    assert strat_map["DNx"] == "both_on_every_muscle"
    assert strat_map["DNy"] == "direct_only"
    assert strat_map["DNz"] == "indirect_only"
    print("[OK] DN-group rollup matches expectations.")

    demo_dir = Path("/tmp/direct_vs_indirect_selftest")
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
