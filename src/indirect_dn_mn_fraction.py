"""
================================================================================
 indirect_dn_mn_fraction.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 05 — Indirect DN -> IN -> MN connectivity (2-hop), GROUPWISE,
            same "relative strength" input-fraction logic as steps 01-04,
            extended across one premotor-interneuron hop.
STATUS    : First venture past direct connectivity. Still unsigned, still
            shallow (2-hop only, no 3+-hop cascades) — see the "why 2-hop
            only" note below. Raw synapse counts are still never reported as
            a strength; every quantity here is built from input fractions.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES
--------------------------------------------------------------------------------
For every DN group and every wing muscle, the indirect strength of the
2-hop DN -> premotor IN -> MN pathway, combined over every qualifying
premotor interneuron (IN) group that bridges them:

   F_indirect(DN->MN) = sum over IN in P of  F(DN->IN) * F(IN->MN)

where P is the set of premotor IN groups (defined below), and both F(DN->IN)
and F(IN->MN) are computed with EXACTLY step 01's input-fraction formula —
just with the source/target relabelled for each hop:

   F(DN->IN) = synapses(DN group -> IN group) / IN group's TOTAL input
               (whole connectome)      <- "what share of this IN's input
                                           comes from this DN type?"
   F(IN->MN) = synapses(IN group -> MN group) / MN group's TOTAL input
               (whole connectome)      <- literally step 01's formula,
                                           source relabelled DN->IN.

This is computed as a straightforward matrix product: build the DN-group x
IN-group fraction matrix A and the IN-group x muscle fraction matrix B (both
via the identical function, called twice), then F_indirect = A @ B. Two-hop
"effective connectivity" via weighted-adjacency matrix multiplication is a
standard, well-established formulation in network analysis; this script
just restricts the intermediate layer to premotor interneurons specifically
(see below) rather than the full connectome, keeping it tractable and
directly comparable to steps 01-04's DN-group x muscle shape.

--------------------------------------------------------------------------------
WHY THIS RESTS ON AN ASSUMPTION — READ BEFORE TRUSTING THE NUMBERS
--------------------------------------------------------------------------------
Multiplying two input fractions together implicitly assumes a PROPORTIONAL
FLOW-THROUGH model: that the same *share* of an IN's total input that comes
from a given DN also determines the *share* of that IN's output to a given
muscle attributable to the DN. That is a standard simplifying approximation
in descriptive connectomics (not this project's invention), NOT a claim
about actual neural computation — real neurons don't linearly apportion
inputs to outputs. Treat F_indirect as a *relative influence score* for
ranking and comparison, not a literal probability or flow quantity.

CRITICALLY: unlike F_in (step 01), F_indirect is NOT bounded by 1.0 when
summed. F_in's bound came from each target's fixed input budget being
divided among all its direct sources. F_indirect sums PRODUCTS across many
different INs, each with its OWN independent denominator — there's no
shared budget forcing the sum down, so nothing here mathematically prevents
F_indirect(DN->MN) from exceeding what F_in ever could. This script's
per-muscle diagnostic reports the summed value across all DN groups purely
descriptively — there's no Cheong-style external reference number for an
indirect-only quantity to check it against (their ~9-10% figure was direct
only), so don't expect it to land near that range, and don't read "small"
or "large" as validation either way without more context.

--------------------------------------------------------------------------------
DEFINING "PREMOTOR INTERNEURON" (the intermediate hop)
--------------------------------------------------------------------------------
Restricted to VNC-intrinsic neurons only (`Super Class ==
"ventral_nerve_cord_intrinsic"`) that have at least one DIRECT synapse onto
a wing MN. This matches Cheong et al.'s own premotor-circuit scope and the
pilot's prior "vnc_interneurons" categorization, and — checked against the
real data (2026-08-21) — needs no separate VNC-restriction step the way
step 02b needed one for DNs: VNC-intrinsic neurons' synapses (as source OR
target) are **100% VNC-tagged already** (7,248,662 / 7,248,662), because
"VNC-intrinsic" is a classification that already means the neuron's whole
arbor is confined to the VNC. Population sizes verified against the real
data: 12,759 VNC-intrinsic neurons total (12,692 with a `Primary Cell
Type`) — matches the pilot's own count of 12,759 VNC interneurons exactly,
a reassuring cross-check — of which 2,153 qualify as premotor by this
definition (close to, though not identical to, the pilot's differently-
defined count of 2,080; different thresholding, same ballpark).

--------------------------------------------------------------------------------
WHY 2-HOP ONLY (not 3+, not full effective connectivity)
--------------------------------------------------------------------------------
CLAUDE.md's original scope note says multi-hop is deprioritised "for now,
focus on direct and shallow structure first" — 2-hop through a well-defined
premotor-IN layer IS that shallow structure, and matches Cheong's own scope.
3+-hop cascades (DN->IN1->IN2->MN, etc.) multiply the combinatorics and the
flow-through assumption's inaccuracy at every added hop, and don't have a
clean intermediate population to restrict to the way "premotor IN" does
here. Left for a later step if the 2-hop picture turns out to need it.

--------------------------------------------------------------------------------
GROUPING  — same convention as steps 01-04, extended to the IN layer
--------------------------------------------------------------------------------
  * DN groups  = shared 'Primary Cell Type' (KEEP_UNTYPED_DNS).
  * IN groups  = shared 'Primary Cell Type' among premotor INs (KEEP_UNTYPED_INS).
  * MN groups  = muscles / motor pools, from data/processed/motor_pools/motor_pools.csv.

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
  data/raw/connections_princeton.csv           columns: pre_root_id, post_root_id,
                                                 neuropil, syn_count, nt_type
  data/raw/neurons.csv                          columns incl. Root ID, Super Class,
                                                 Primary Cell Type
  data/processed/motor_pools/motor_pools.csv    columns: muscle, motor_neuron_ids

OUTPUTS  (written to results/indirect_dn_mn_fraction/)
  dn_to_in_fraction_long.csv          DN group -> IN group, F(DN->IN), nonzero only
  in_to_mn_fraction_long.csv          IN group -> muscle, F(IN->MN), nonzero only
  indirect_dn_mn_fraction_matrix.csv  DN-group x muscle matrix of F_indirect (the main result)
  indirect_dn_mn_fraction_long.csv    long form, nonzero DN->muscle pairs, sorted desc
  indirect_dn_mn_thresholded.csv      edges with F_indirect >= THRESHOLD
  muscle_direct_vs_indirect_totals.csv  per-muscle: summed direct (step 01-style, recomputed
                                       inline for comparison) vs summed indirect vs combined
  figure_indirect_dn_mn_fraction.png  heatmap + formula panel + sanity bar
  run_summary.txt                     counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/indirect_dn_mn_fraction.py             # real data
  python src/indirect_dn_mn_fraction.py --selftest   # synthetic check

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "indirect_dn_mn_fraction"

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
IN_SUPER_CLASS = "ventral_nerve_cord_intrinsic"   # premotor-IN candidate pool


# =============================================================================
# PARAMETERS
# =============================================================================
# Display/reporting cutoff on F_indirect. Same value as steps 01-04 for
# consistency, but NOTE: F_indirect isn't on the same bounded 0-1 probability
# scale F_in was (see docstring) — this is a magnitude cutoff for
# readability, not literally "1% of the muscle's input" in the strict sense.
THRESHOLD = 0.01

KEEP_UNTYPED_DNS = True
KEEP_UNTYPED_INS = True   # same "don't silently drop" rationale as DNs

# Figure sizing — same convention as steps 01-04: fixed 16:9 (13.33 x 7.5in),
# heatmap capped to the top TOP_N_ROWS DN groups by strongest edge so the
# figure stays slide-sized. Full matrix always in the saved CSVs regardless.
TOP_N_ROWS = 25

# --- Figure style: validated palette (dataviz skill's references/palette.md) ---
CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CAT_MAGENTA, CAT_GREEN, CAT_VIOLET, CAT_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_HAIRLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                         "#256abf", "#184f95", "#0d366b"]

# Optional: DN groups to force-show + call out in the figure regardless of
# whether they make the TOP_N_ROWS cut — same mechanism as step 01.
HIGHLIGHT_DN_GROUPS = []
HIGHLIGHT_COLOR = CAT_GREEN


# =============================================================================
# CORE COMPUTATION
# =============================================================================
def build_dn_groups(neurons: pd.DataFrame) -> pd.Series:
    """Identical to step 01."""
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
    """Identical to step 01."""
    mapping = {}
    for _, row in motor_pools.iterrows():
        muscle = row["muscle"]
        mn_ids = ast.literal_eval(row["motor_neuron_ids"]) \
            if isinstance(row["motor_neuron_ids"], str) else row["motor_neuron_ids"]
        for mn_id in mn_ids:
            mapping[mn_id] = muscle
    return pd.Series(mapping, name="muscle")


def build_in_groups(neurons: pd.DataFrame, connections: pd.DataFrame,
                    mn_ids: set) -> pd.Series:
    """
    Premotor VNC-intrinsic interneurons: Super Class == IN_SUPER_CLASS AND at
    least one direct synapse onto a wing MN (mn_ids). See module docstring
    for the real-data population counts and why no VNC-restriction step is
    needed here. Grouped by 'Primary Cell Type', same convention as DNs.
    """
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
    """
    Generic denominator builder — total input synapses onto each target
    neuron, summed over the WHOLE connectome. Identical logic to step 01's
    compute_mn_total_input, generalised so it can build either an MN's or
    an IN's total-input denominator (same math either way).
    """
    total_input_all = connections.groupby(CONN_TARGET)[CONN_WEIGHT].sum()
    return total_input_all.reindex(list(target_ids)).fillna(0.0)


def compute_groupwise_fraction(connections: pd.DataFrame,
                               source_group_map: pd.Series,
                               target_group_map: pd.Series,
                               target_total_input: pd.Series,
                               source_col: str, target_col: str) -> pd.DataFrame:
    """
    Generic groupwise input-fraction computation — identical formula to step
    01's compute_groupwise_input_fraction, generalised over column names so
    the SAME function computes both the DN->IN leg and the IN->MN leg
    (called twice below, once per hop) rather than duplicating the logic.
    """
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
    out = out.sort_values("input_fraction", ascending=False).reset_index(drop=True)
    return out


def combine_indirect(dn_in_long: pd.DataFrame, in_mn_long: pd.DataFrame):
    """
    F_indirect(DN->MN) = sum over IN of F(DN->IN) * F(IN->MN), computed as a
    matrix product: (DN group x IN group) @ (IN group x muscle). The two
    matrices' IN-group axes are reindexed onto their union (fill 0) first so
    `.dot()` aligns correctly regardless of which INs appear in only one leg
    (e.g. an IN with DN input but zero measured wing-muscle output, or vice
    versa — both legitimately possible and both correctly contribute 0).
    """
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
    long_df = long_df[long_df["indirect_fraction"] > 0]
    long_df = long_df.sort_values("indirect_fraction", ascending=False).reset_index(drop=True)
    return long_df, indirect_matrix


def summarise_direct_vs_indirect(connections: pd.DataFrame,
                                 dn_group_map: pd.Series, mn_group_map: pd.Series,
                                 mn_total_input: pd.Series,
                                 indirect_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per-muscle: summed DIRECT DN fraction (step 01's math, recomputed inline
    here so this script stays self-contained rather than reading step 01's
    saved CSV — same standalone-script convention as the rest of the
    pipeline) vs summed INDIRECT fraction vs their sum. Answers "how much of
    a muscle's DN-attributable drive is direct vs mediated by a premotor IN?"
    """
    direct_long = compute_groupwise_fraction(
        connections, dn_group_map, mn_group_map, mn_total_input, "dn_group", "muscle")
    direct_total = (direct_long.groupby("muscle")["input_fraction"].sum()
                    .rename("direct_total_fraction"))
    indirect_total = (indirect_long.groupby("muscle")["indirect_fraction"].sum()
                      .rename("indirect_total_fraction"))

    all_muscles = pd.Index(sorted(mn_group_map.unique()))
    summary = pd.DataFrame(index=all_muscles)
    summary = summary.join(direct_total).join(indirect_total).fillna(0.0)
    summary["combined_total"] = summary["direct_total_fraction"] + summary["indirect_total_fraction"]
    return summary.sort_values("combined_total", ascending=False)


# =============================================================================
# FIGURE  — fixed at 16:9 (13.33 x 7.5in), same conventions as steps 01-04.
# =============================================================================
def make_figure(indirect_long: pd.DataFrame, out_path: Path, demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap

    seq_blue = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE_STEPS)

    mat = indirect_long.pivot_table(index="dn_group", columns="muscle",
                                    values="indirect_fraction", aggfunc="sum", fill_value=0.0)
    keep_rows = mat.max(axis=1) >= THRESHOLD
    mat_shown = mat.loc[keep_rows]
    mat_shown = mat_shown.loc[mat_shown.max(axis=1).sort_values(ascending=False).index]
    n_above_threshold = mat_shown.shape[0]
    truncated = n_above_threshold > TOP_N_ROWS
    if truncated:
        mat_shown = mat_shown.iloc[:TOP_N_ROWS]

    highlight_set = set()
    n_forced = 0
    for name in HIGHLIGHT_DN_GROUPS:
        if name not in mat.index:
            print(f"    [warn] HIGHLIGHT_DN_GROUPS: '{name}' has no indirect wing-muscle "
                  f"pathway in this data — skipped.")
            continue
        highlight_set.add(name)
        if name not in mat_shown.index:
            mat_shown = pd.concat([mat_shown, mat.loc[[name]]])
            n_forced += 1
    if highlight_set:
        mat_shown = mat_shown.loc[mat_shown.max(axis=1).sort_values(ascending=False).index]

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    gs = GridSpec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1.0, 5.2],
                  hspace=0.5, wspace=0.30, top=0.90, bottom=0.09,
                  left=0.06, right=0.99)

    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis("off")
    formula = (
        r"$F_{\mathrm{indirect}}(DN\!\rightarrow\!MN) \;=\; "
        r"\sum_{IN\,\in\,P} F(DN\!\rightarrow\!IN)\cdot F(IN\!\rightarrow\!MN)$"
    )
    ax_formula.text(0.02, 0.55, formula, fontsize=13.5, va="center", color=INK)
    ax_formula.text(0.50, 0.55,
                    "P = premotor VNC-intrinsic interneuron groups\n"
                    "each F(·) is step 01's input-fraction formula, one hop\n"
                    "NOT bounded by 1 when summed — relative score, not a probability",
                    fontsize=8.5, va="center", color=INK_SOFT)
    if demo:
        ax_formula.text(0.99, 0.95, "DEMO / SELF-TEST DATA — not real results",
                        fontsize=9, color="#B00020", fontweight="bold",
                        va="top", ha="right")

    ax = fig.add_subplot(gs[1, 0])
    if mat_shown.size == 0:
        ax.text(0.5, 0.5, "No DN group reaches the threshold.",
                ha="center", va="center")
        ax.axis("off")
    else:
        im = ax.imshow(mat_shown.values, aspect="auto", cmap=seq_blue,
                       vmin=0.0, vmax=max(mat_shown.values.max(), THRESHOLD))
        ax.set_xticks(range(mat_shown.shape[1]))
        ax.set_xticklabels(mat_shown.columns, rotation=90, fontsize=8, color=INK_SOFT)
        ax.set_yticks(range(mat_shown.shape[0]))
        ax.set_yticklabels(mat_shown.index, fontsize=7, color=INK_SOFT)
        ax.tick_params(colors=AXIS_LINE)
        for spine in ax.spines.values():
            spine.set_color(AXIS_LINE)
        ax.set_xlabel("Wing motor-neuron group (muscle)", fontsize=9, fontweight="bold", color=INK)
        if truncated:
            ylabel = (f"DN group  (top {len(mat_shown) - n_forced} of {n_above_threshold} "
                      f"with max $F_{{\\mathrm{{indirect}}}}$ $\\geq$ {THRESHOLD:g}"
                      + (f" + {n_forced} highlighted" if n_forced else "") + ")")
        else:
            ylabel = f"DN group  (max $F_{{\\mathrm{{indirect}}}}$ across muscles $\\geq$ {THRESHOLD:g})"
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", color=INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("indirect fraction $F_{\\mathrm{indirect}}$", fontsize=8, color=INK)
        cbar.ax.tick_params(colors=AXIS_LINE, labelcolor=INK_SOFT)
        cbar.outline.set_edgecolor(AXIS_LINE)

        if highlight_set:
            for i, dn_name in enumerate(mat_shown.index):
                if dn_name in highlight_set:
                    ax.get_yticklabels()[i].set_color(HIGHLIGHT_COLOR)
                    ax.get_yticklabels()[i].set_fontweight("bold")
                    ax.add_patch(Rectangle(
                        (-0.5, i - 0.5), mat_shown.shape[1], 1.0,
                        fill=False, edgecolor=HIGHLIGHT_COLOR, linewidth=2.0,
                        zorder=5, clip_on=False))

    # --- Sanity panel (bottom-right): summed indirect fraction per muscle --
    axb = fig.add_subplot(gs[1, 1])
    per_muscle_indirect = (indirect_long.groupby("muscle")["indirect_fraction"].sum()
                           .sort_values())
    axb.barh(range(len(per_muscle_indirect)), per_muscle_indirect.values, color=CAT_BLUE)
    axb.set_yticks(range(len(per_muscle_indirect)))
    axb.set_yticklabels(per_muscle_indirect.index, fontsize=6, color=INK_SOFT)
    axb.tick_params(colors=AXIS_LINE)
    for spine in ["top", "right"]:
        axb.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axb.spines[spine].set_color(AXIS_LINE)
    axb.set_xlabel("summed indirect\nfraction (all DNs)", fontsize=8, fontweight="bold", color=INK)
    axb.set_title("sanity check\n(not bounded by 1 — see docstring)", fontsize=7.5, color=INK)

    fig.suptitle("Indirect descending $\\rightarrow$ IN $\\rightarrow$ wing motor connectivity  ·  "
                 "groupwise 2-hop fraction", fontsize=12.5, fontweight="bold", y=0.985, color=INK)
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

    n_vnc_intrinsic = int((neurons[COL_SUPER_CLASS] == IN_SUPER_CLASS).sum())
    n_untyped_in = int(in_group_map.astype(str).str.startswith("untyped_").sum())
    print(f"\nDN neurons grouped : {len(dn_group_map):,} DNs -> {dn_group_map.nunique():,} groups")
    print(f"Wing MNs           : {len(mn_group_map):,} MNs -> {mn_group_map.nunique():,} muscles")
    print(f"VNC-intrinsic pool : {n_vnc_intrinsic:,} candidates -> "
          f"{len(in_group_map):,} qualify as premotor -> {in_group_map.nunique():,} IN groups "
          f"({n_untyped_in:,} untyped singletons)")

    mn_total_input = compute_total_input(connections, mn_group_map.index.values)
    in_total_input = compute_total_input(connections, in_group_map.index.values)

    dn_in_long = compute_groupwise_fraction(
        connections, dn_group_map, in_group_map, in_total_input, "dn_group", "in_group")
    in_mn_long = compute_groupwise_fraction(
        connections, in_group_map, mn_group_map, mn_total_input, "in_group", "muscle")

    print(f"\nDN->IN nonzero edges : {len(dn_in_long)}")
    print(f"IN->MN nonzero edges : {len(in_mn_long)}")

    indirect_long, indirect_matrix = combine_indirect(dn_in_long, in_mn_long)
    thresholded = indirect_long[indirect_long["indirect_fraction"] >= THRESHOLD].copy()

    dvi_summary = summarise_direct_vs_indirect(
        connections, dn_group_map, mn_group_map, mn_total_input, indirect_long)

    print(f"\nIndirect DN->muscle nonzero edges : {len(indirect_long)}")
    print(f"edges >= threshold                : {len(thresholded)}")
    print("\n--- Per-muscle: direct vs indirect vs combined (summed over all DN groups) ---")
    print(dvi_summary.round(4).to_string())

    print(f"\n--- Top {min(15, len(indirect_long))} indirect pathways ---")
    print(indirect_long.head(15).to_string(index=False))

    dn_in_long.to_csv(out_dir / "dn_to_in_fraction_long.csv", index=False)
    in_mn_long.to_csv(out_dir / "in_to_mn_fraction_long.csv", index=False)
    indirect_matrix.to_csv(out_dir / "indirect_dn_mn_fraction_matrix.csv")
    indirect_long.to_csv(out_dir / "indirect_dn_mn_fraction_long.csv", index=False)
    thresholded.to_csv(out_dir / "indirect_dn_mn_thresholded.csv", index=False)
    dvi_summary.to_csv(out_dir / "muscle_direct_vs_indirect_totals.csv")

    fig_path = out_dir / "figure_indirect_dn_mn_fraction.png"
    make_figure(indirect_long, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Indirect DN->IN->MN groupwise 2-hop fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold (display)             : {THRESHOLD}\n")
        fh.write(f"keep untyped DNs / INs           : {KEEP_UNTYPED_DNS} / {KEEP_UNTYPED_INS}\n")
        fh.write(f"DN neurons / groups              : {len(dn_group_map)} / {dn_group_map.nunique()}\n")
        fh.write(f"wing MNs / muscles               : {len(mn_group_map)} / {mn_group_map.nunique()}\n")
        fh.write(f"VNC-intrinsic candidates          : {n_vnc_intrinsic}\n")
        fh.write(f"premotor INs / IN groups          : {len(in_group_map)} / {in_group_map.nunique()}\n")
        fh.write(f"  (untyped IN singletons)         : {n_untyped_in}\n")
        fh.write(f"DN->IN nonzero edges              : {len(dn_in_long)}\n")
        fh.write(f"IN->MN nonzero edges              : {len(in_mn_long)}\n")
        fh.write(f"indirect DN->muscle nonzero edges : {len(indirect_long)}\n")
        fh.write(f"edges >= threshold                : {len(thresholded)}\n\n")
        fh.write("per-muscle direct vs indirect vs combined:\n")
        fh.write(dvi_summary.round(4).to_string())
        fh.write("\n\ntop 15 indirect pathways:\n")
        fh.write(indirect_long.head(15).to_string(index=False))
        fh.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return indirect_long, dvi_summary


# =============================================================================
# SELF-TEST
# =============================================================================
def selftest():
    """
    Hand-computed 2-hop synthetic connectome.

    DN 101 (group DNx) -> IN 301 (group INy): 10 synapses
    DN 101             -> IN 302 (group INz): 5 synapses
    DN 101             -> IN 303 (non-premotor VNC-intrinsic, NO edge to any
                                   MN — must be excluded entirely): 7 synapses
    DN 101             -> MN 201 (muscleA), DIRECT edge: 2 synapses
    999 (other, sensory) -> IN 301: 30 synapses
    998 (other, sensory) -> IN 302: 15 synapses
    997 (other, sensory) -> MN 201: 28 synapses
    IN 301 -> MN 201: 8 synapses
    IN 302 -> MN 201: 4 synapses

    Hand calc:
      TotalInput(301) = 10+30 = 40      TotalInput(302) = 5+15 = 20
      TotalInput(201) = 2(direct DN)+8(IN301)+4(IN302)+28(other) = 42

      F(DNx->INy)  = 10/40 = 0.25
      F(DNx->INz)  = 5/20  = 0.25
      F(INy->A)    = 8/42  = 4/21  = 0.190476
      F(INz->A)    = 4/42  = 2/21  = 0.095238

      F_indirect(DNx->A) = 0.25*(4/21) + 0.25*(2/21) = 0.25*(6/21) = 1/14 = 0.071429

      F_direct(DNx->A)   = 2/42 = 1/21 = 0.047619
      combined           = 1/21 + 1/14 = 5/42 = 0.119048

    IN 303 has no edge to any MN, so it must NOT appear in in_group_map at
    all — this is the "premotor filter" check.
    """
    connections = pd.DataFrame({
        CONN_SOURCE: [101, 101, 101, 101, 999, 998, 997, 301, 302],
        CONN_TARGET: [301, 302, 303, 201, 301, 302, 201, 201, 201],
        CONN_WEIGHT: [10,  5,   7,   2,   30,  15,  28,  8,   4],
    })
    neurons = pd.DataFrame({
        COL_ROOT_ID:     [101,          201,     301,                       302,
                          303,                       999,        998,        997],
        COL_SUPER_CLASS: ["descending", "motor", "ventral_nerve_cord_intrinsic",
                          "ventral_nerve_cord_intrinsic",
                          "ventral_nerve_cord_intrinsic", "sensory", "sensory", "sensory"],
        COL_CELL_TYPE:   ["DNx", None, "INy", "INz", "INnonpremotor", None, None, None],
    })
    motor_pools = pd.DataFrame({
        "muscle": ["muscleA"],
        "motor_neuron_ids": ["[201]"],
    })

    dn_map = build_dn_groups(neurons)
    mn_map = build_mn_groups(motor_pools)
    in_map = build_in_groups(neurons, connections, set(mn_map.index))

    print("IN groups found:", dict(in_map))
    assert set(in_map.index) == {301, 302}, \
        f"IN 303 (non-premotor) should be excluded; got index {list(in_map.index)}"
    assert in_map[301] == "INy" and in_map[302] == "INz"

    mn_tot = compute_total_input(connections, mn_map.index.values)
    in_tot = compute_total_input(connections, in_map.index.values)

    dn_in_long = compute_groupwise_fraction(connections, dn_map, in_map, in_tot, "dn_group", "in_group")
    in_mn_long = compute_groupwise_fraction(connections, in_map, mn_map, mn_tot, "in_group", "muscle")

    print("\nDN->IN:")
    print(dn_in_long.to_string(index=False))
    print("\nIN->MN:")
    print(in_mn_long.to_string(index=False))

    got_dn_in = {(r.dn_group, r.in_group): round(r.input_fraction, 6) for r in dn_in_long.itertuples()}
    assert got_dn_in == {("DNx", "INy"): 0.25, ("DNx", "INz"): 0.25}, got_dn_in

    got_in_mn = {(r.in_group, r.muscle): round(r.input_fraction, 6) for r in in_mn_long.itertuples()}
    assert got_in_mn == {("INy", "muscleA"): round(4 / 21, 6), ("INz", "muscleA"): round(2 / 21, 6)}, got_in_mn

    indirect_long, indirect_matrix = combine_indirect(dn_in_long, in_mn_long)
    print("\nIndirect:")
    print(indirect_long.to_string(index=False))
    got_indirect = round(indirect_long.iloc[0]["indirect_fraction"], 6)
    assert got_indirect == round(1 / 14, 6), f"expected {1/14:.6f}, got {got_indirect}"

    dvi = summarise_direct_vs_indirect(connections, dn_map, mn_map, mn_tot, indirect_long)
    print("\nDirect vs indirect vs combined:")
    print(dvi.to_string())
    assert round(dvi.loc["muscleA", "direct_total_fraction"], 6) == round(1 / 21, 6)
    assert round(dvi.loc["muscleA", "indirect_total_fraction"], 6) == round(1 / 14, 6)
    assert round(dvi.loc["muscleA", "combined_total"], 6) == round(5 / 42, 6)

    print("\n[OK] DN->IN, IN->MN, indirect combination, premotor filtering, and the "
          "direct-vs-indirect summary all match hand calc.")

    demo_dir = Path("/tmp/indirect_dn_mn_fraction_selftest")
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
