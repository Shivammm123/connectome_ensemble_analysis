"""
================================================================================
 direct_dn_mn_output_fraction.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 02 — Direct DN -> MN connectivity, GROUPWISE, OUTPUT-FRACTION metric
STATUS    : Same direct, unsigned, single-hop scope as step 01. No neuro-
            transmitter signs, no multi-hop paths. Raw synapse counts are
            NOT used as the connection strength anywhere in the reported
            results; they are only ever a numerator that is immediately
            normalised — same discipline as step 01, different denominator.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES
--------------------------------------------------------------------------------
For every descending-neuron group (G_DN) and every wing motor-neuron group
(G_MN, i.e. a muscle / motor pool), the *groupwise output fraction*:

                     sum over d in G_DN, m in G_MN of  w(d -> m)
   F_out(G_DN->G_MN) = -------------------------------------------------
                          sum over d in G_DN of  TotalOutput(d)

  * NUMERATOR   = all synapses from the DN group onto the MN group. The same
                  quantity as step 01's numerator — same direct DN->MN edges.
  * DENOMINATOR = the DN group's *total* output budget — every synapse that
                  group sends to ANY target in the WHOLE connectome (not just
                  to MNs, not just to our wing-muscle subset).

INTERPRETATION
   F_out answers "what fraction of this DN type's total outgoing synapses are
   spent driving this particular muscle?" It is the source's-eye view of
   *dedication* — how much of the DN's whole budget goes here — as opposed to
   step 01's *influence* question (what fraction of the muscle's input comes
   from here). A DN can have high F_out and low F_in (a small neuron whose
   entire output goes to one minor muscle input) or the reverse (a large
   premotor-adjacent DN supplying a big share of a muscle's input while that
   is a tiny slice of the DN's own enormous output budget). Neither number
   alone tells the whole story — a later step (planned, not this one) takes
   the geometric mean of the two to find edges strong from both sides.

WHY THIS METRIC IS *NOT* USED FOR CROSS-DATASET COMPARISON
   Unlike step 01's input fraction (whose denominator, MN input, is VNC-local
   in every BANC/MANC/maleCNS dataset), THIS metric's denominator is the DN's
   TOTAL output, which in BANC includes brain-side synapses that a VNC-only
   dataset such as MANC simply does not have. The same DN type would show a
   smaller F_out in BANC than in MANC purely because BANC's denominator is
   bigger, with no real change in wing-muscle drive. Use F_out only WITHIN
   BANC — e.g. "how dedicated is this DN to wing control, relative to its own
   total output" — never to compare a DN's dedication across datasets. This
   is exactly the asymmetry step 01's docstring warned about.

--------------------------------------------------------------------------------
THE SUBSET TRAP, MIRRORED
--------------------------------------------------------------------------------
Step 01 broke if the connections file was pre-filtered to only DN->MN edges
(MN input then looked smaller than it is, inflating F_in). This script has
the mirror-image failure mode: if `connections_princeton.csv` were filtered
to only DN->MN edges, a DN's TotalOutput would be missing every non-MN target
it has (other DNs, interneurons, brain-side partners...), so the denominator
would be too small and F_out would be inflated toward 1.0 broadly. There is
no published external reference number for this the way Cheong gave one for
step 01 (~9-10% was specifically about MN input share, not DN output share),
so this script's diagnostic is a heuristic, not a target to match: a handful
of DN groups legitimately dedicating most of their output to one muscle is
plausible (small dedicated command-like neurons exist); MANY DN groups doing
that is the subset-trap signature, not real biology.

--------------------------------------------------------------------------------
GROUPING  — identical to step 01 (see that script for the full rationale)
--------------------------------------------------------------------------------
  * DN groups  = shared 'Primary Cell Type'. Untyped DNs kept as singletons
                 (KEEP_UNTYPED_DNS = True), same as step 01.
  * MN groups  = muscles / motor pools, from data/processed/motor_pools/motor_pools.csv.

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
  data/raw/connections_princeton.csv           columns: pre_root_id, post_root_id,
                                                 neuropil, syn_count, nt_type
  data/raw/neurons.csv                          columns incl. Root ID, Super Class,
                                                 Primary Cell Type
  data/processed/motor_pools/motor_pools.csv    columns: muscle, motor_neuron_ids

OUTPUTS  (written to results/direct_dn_mn_output_fraction/)
  direct_dn_mn_output_fraction_matrix.csv   DN-group x muscle matrix of F_out
  direct_dn_mn_output_fraction_long.csv     long form, non-zero edges only
  direct_dn_mn_output_thresholded.csv       edges with F_out >= THRESHOLD
  dn_group_output_totals.csv                per-DN-group denominator + total wing frac
  figure_direct_dn_mn_output_fraction.png   heatmap + formula panel + sanity bar
  run_summary.txt                           counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  # normal run against your real data (from anywhere):
  python src/direct_dn_mn_output_fraction.py

  # verify the maths on a tiny synthetic connectome with known answers:
  python src/direct_dn_mn_output_fraction.py --selftest

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
# PATHS  — the script lives in src/ and assumes the project root is its parent.
#          Everything is a plain constant so it is obvious what to change.
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent          # .../<project>/src
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../<project>
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "direct_dn_mn_output_fraction"

CONNECTIONS_CSV = DATA_DIR / "raw" / "connections_princeton.csv"
NEURONS_CSV     = DATA_DIR / "raw" / "neurons.csv"
MOTOR_POOLS_CSV = DATA_DIR / "processed" / "motor_pools" / "motor_pools.csv"


# =============================================================================
# COLUMN NAMES  — BANC / Princeton format uses spaces and capitals. Do not
#                 "fix" these to snake_case; that was a recurring pilot bug.
# =============================================================================
COL_ROOT_ID     = "Root ID"
COL_SUPER_CLASS = "Super Class"
COL_CELL_TYPE   = "Primary Cell Type"   # neurons.csv has no bare 'Cell Type' col.

# connections_princeton.csv is the raw Princeton/FlyWire-codex export: one row
# per (pre, post, neuropil), so a given pre->post pair can span several rows.
# Every place these are used does a .sum(), so that's aggregated correctly.
CONN_SOURCE = "pre_root_id"
CONN_TARGET = "post_root_id"
CONN_WEIGHT = "syn_count"

DN_SUPER_CLASS = "descending"   # value in 'Super Class' that marks a DN


# =============================================================================
# PARAMETERS
# =============================================================================
# "Substantial" direct connection threshold on the output fraction. Same value
# as step 01's input-fraction threshold, for consistency — display/reporting
# only, the full continuous matrix is always saved unthresholded.
THRESHOLD = 0.01

# Keep DNs that have no Cell Type as their own singleton groups (True), or
# drop them entirely (False). Default True so nothing is silently lost.
KEEP_UNTYPED_DNS = True

# Heuristic subset-trap flag (NOT an external reference — see docstring: there
# is no Cheong-style published number for output-side dedication). A DN group
# summing >50% of its total output to these 18 wing muscles alone is worth a
# second look; many DN groups doing that is the actual warning sign.
SUBSET_TRAP_WARN_LEVEL = 0.50
SUBSET_TRAP_WARN_GROUP_FRACTION = 0.10   # warn if >10% of DN groups exceed it

# Figure sizing: with 500+ possible DN groups, showing every one that clears
# THRESHOLD somewhere (routinely 80-100+ for this metric) makes an unreadably
# tall figure. Cap the heatmap to the top TOP_N_ROWS by strongest connection —
# the full, untruncated matrix is still in the saved CSVs regardless. The
# figure itself is fixed at 16:9 (13.33 x 7.5in) to drop onto a slide as-is.
TOP_N_ROWS = 25

# --- Figure style: validated palette (dataviz skill's references/palette.md) ---
# Same tokens as step 01 — kept consistent across every figure in this
# pipeline rather than reinvented per script (still duplicated per-file, per
# this project's standalone-script convention, not shared via import).
CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CAT_MAGENTA, CAT_GREEN, CAT_VIOLET, CAT_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_HAIRLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                         "#256abf", "#184f95", "#0d366b"]


# =============================================================================
# CORE COMPUTATION
# =============================================================================
def build_dn_groups(neurons: pd.DataFrame) -> pd.Series:
    """
    Return a Series mapping DN Root ID -> DN group label.

    Group label = 'Primary Cell Type' when present; otherwise (if
    KEEP_UNTYPED_DNS) a singleton label 'untyped_<RootID>'. DNs are those
    whose 'Super Class' equals DN_SUPER_CLASS. Identical to step 01.
    """
    dns = neurons[neurons[COL_SUPER_CLASS] == DN_SUPER_CLASS].copy()

    cell_type = dns[COL_CELL_TYPE]
    has_type = cell_type.notna() & (cell_type.astype(str).str.strip() != "")

    labels = cell_type.astype("object").copy()
    if KEEP_UNTYPED_DNS:
        labels.loc[~has_type] = "untyped_" + dns.loc[~has_type, COL_ROOT_ID].astype(str)
    else:
        labels.loc[~has_type] = np.nan

    group_map = pd.Series(labels.values, index=dns[COL_ROOT_ID].values, name="dn_group")
    group_map = group_map.dropna()
    return group_map


def build_mn_groups(motor_pools: pd.DataFrame) -> pd.Series:
    """
    Return a Series mapping wing MN Root ID -> muscle (motor-pool label).
    Identical to step 01.
    """
    mapping = {}
    for _, row in motor_pools.iterrows():
        muscle = row["muscle"]
        mn_ids = ast.literal_eval(row["motor_neuron_ids"]) \
            if isinstance(row["motor_neuron_ids"], str) else row["motor_neuron_ids"]
        for mn_id in mn_ids:
            mapping[mn_id] = muscle
    return pd.Series(mapping, name="muscle")


def compute_dn_total_output(connections: pd.DataFrame,
                            dn_ids: np.ndarray) -> pd.Series:
    """
    DENOMINATOR builder. Total output synapses from each DN, summed over the
    ENTIRE connectome (every target). This is the step whose correctness
    depends on `connections` being the full edge list — the mirror image of
    step 01's compute_mn_total_input.
    """
    total_output_all = connections.groupby(CONN_SOURCE)[CONN_WEIGHT].sum()
    return total_output_all.reindex(dn_ids).fillna(0.0)


def compute_groupwise_output_fraction(connections: pd.DataFrame,
                                      dn_group_map: pd.Series,
                                      mn_group_map: pd.Series,
                                      dn_total_output: pd.Series) -> pd.DataFrame:
    """
    Return a long-form DataFrame with one row per (dn_group, muscle) that has
    at least one direct synapse, containing:
        dn_group, muscle, numerator_synapses, denominator_output, output_fraction

    NUMERATOR   : groupwise synapses DN group -> MN group (same as step 01).
    DENOMINATOR : groupwise total output of the DN group (sum of member totals).
    """
    dn_ids = set(dn_group_map.index)
    mn_ids = set(mn_group_map.index)

    mask = connections[CONN_SOURCE].isin(dn_ids) & connections[CONN_TARGET].isin(mn_ids)
    dn_mn = connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()

    dn_mn["dn_group"] = dn_mn[CONN_SOURCE].map(dn_group_map)
    dn_mn["muscle"]   = dn_mn[CONN_TARGET].map(mn_group_map)

    numerator = (dn_mn.groupby(["dn_group", "muscle"])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())

    # DENOMINATOR: total output per DN group = sum of member-DN total outputs.
    denom_per_dn_group = (dn_total_output.groupby(dn_group_map).sum()
                          .rename("denominator_output"))

    out = numerator.merge(denom_per_dn_group, left_on="dn_group", right_index=True, how="left")
    out["output_fraction"] = out["numerator_synapses"] / out["denominator_output"]
    out = out.sort_values("output_fraction", ascending=False).reset_index(drop=True)
    return out, denom_per_dn_group


def summarise_per_dn_group(long_df: pd.DataFrame,
                           denom_per_dn_group: pd.Series) -> pd.DataFrame:
    """
    Per-DN-group diagnostic: total output (denominator), summed direct
    output fraction to wing muscles across ALL 18 muscles, and number of
    muscles above THRESHOLD. The summed fraction is this script's subset-trap
    check (see docstring) — the mirror of step 01's per-muscle summed check.
    """
    total_wing_frac = (long_df.groupby("dn_group")["output_fraction"].sum()
                       .rename("total_wing_output_fraction"))
    n_above = (long_df[long_df["output_fraction"] >= THRESHOLD]
               .groupby("dn_group")["muscle"].nunique()
               .rename(f"n_muscles_ge_{THRESHOLD:g}"))

    summary = pd.DataFrame({"denominator_output": denom_per_dn_group})
    summary = summary.join(total_wing_frac).join(n_above)
    summary["total_wing_output_fraction"] = summary["total_wing_output_fraction"].fillna(0.0)
    summary[f"n_muscles_ge_{THRESHOLD:g}"] = \
        summary[f"n_muscles_ge_{THRESHOLD:g}"].fillna(0).astype(int)
    return summary.sort_values("total_wing_output_fraction", ascending=False)


# =============================================================================
# FIGURE  — fixed at 16:9 (13.33 x 7.5in) so it drops straight onto a
#           widescreen slide no matter how many DN groups clear the
#           threshold — see TOP_N_ROWS above for how the row count is capped.
# =============================================================================
def make_figure(long_df: pd.DataFrame,
                per_dn_group: pd.DataFrame,
                out_path: Path,
                demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap

    seq_blue = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE_STEPS)

    # Build DN-group x muscle matrix, keeping only DN groups that reach
    # THRESHOLD somewhere, then cap to the top TOP_N_ROWS by strongest
    # connection for a readable, slide-sized figure. The full (untruncated)
    # matrix is still in the saved CSVs regardless.
    mat = long_df.pivot_table(index="dn_group", columns="muscle",
                              values="output_fraction", aggfunc="sum", fill_value=0.0)
    keep_rows = mat.max(axis=1) >= THRESHOLD
    mat_shown = mat.loc[keep_rows]
    mat_shown = mat_shown.loc[mat_shown.max(axis=1).sort_values(ascending=False).index]
    n_above_threshold = mat_shown.shape[0]
    truncated = n_above_threshold > TOP_N_ROWS
    if truncated:
        mat_shown = mat_shown.iloc[:TOP_N_ROWS]

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    gs = GridSpec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1.0, 5.2],
                  hspace=0.5, wspace=0.30, top=0.90, bottom=0.09,
                  left=0.06, right=0.99)

    # --- Formula / methods panel (top) : shows the logic explicitly --------
    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis("off")
    formula = (
        r"$F_{\mathrm{out}}(G_{DN}\!\rightarrow\!G_{MN}) \;=\; "
        r"\dfrac{\sum_{d\in G_{DN}}\sum_{m\in G_{MN}} w(d\rightarrow m)}"
        r"{\sum_{d\in G_{DN}} \mathrm{TotalOutput}(d)}$"
    )
    ax_formula.text(0.02, 0.55, formula, fontsize=14, va="center", color=INK)
    ax_formula.text(0.50, 0.55,
                    "numerator: all synapses DN type $\\rightarrow$ muscle\n"
                    "denominator: DN type's TOTAL output (whole connectome)",
                    fontsize=9, va="center", color=INK_SOFT)
    if demo:
        ax_formula.text(0.99, 0.95, "DEMO / SELF-TEST DATA — not real results",
                        fontsize=9, color="#B00020", fontweight="bold",
                        va="top", ha="right")

    # --- Heatmap (bottom-left) : the main result ----------------------------
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
            ylabel = (f"DN group  (top {len(mat_shown)} of {n_above_threshold} "
                      f"with max $F_{{\\mathrm{{out}}}}$ $\\geq$ {THRESHOLD:g})")
        else:
            ylabel = f"DN group  (max $F_{{\\mathrm{{out}}}}$ across muscles $\\geq$ {THRESHOLD:g})"
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", color=INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("output fraction $F_{\\mathrm{out}}$", fontsize=8, color=INK)
        cbar.ax.tick_params(colors=AXIS_LINE, labelcolor=INK_SOFT)
        cbar.outline.set_edgecolor(AXIS_LINE)

    # --- Sanity-check bar (bottom-right) : dedication of the SAME DN groups --
    # Restricted to the DN groups shown in the heatmap. No border on the
    # bars — matplotlib's default bar spacing already separates them.
    axb = fig.add_subplot(gs[1, 1])
    pm = per_dn_group.loc[mat_shown.index].sort_values("total_wing_output_fraction") \
        if mat_shown.size else per_dn_group.iloc[0:0]
    axb.barh(range(len(pm)), pm["total_wing_output_fraction"].values, color=CAT_BLUE)
    axb.set_yticks(range(len(pm)))
    axb.set_yticklabels(pm.index, fontsize=6, color=INK_SOFT)
    axb.tick_params(colors=AXIS_LINE)
    for spine in ["top", "right"]:
        axb.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axb.spines[spine].set_color(AXIS_LINE)
    axb.set_xlabel("total output fraction\nto wing muscles (all 18)", fontsize=8,
                  fontweight="bold", color=INK)
    axb.set_title("sanity check", fontsize=9, color=INK)

    fig.suptitle("Direct descending $\\rightarrow$ wing motor connectivity  ·  "
                 "groupwise output fraction", fontsize=13, fontweight="bold", y=0.985, color=INK)
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

    n_dn_groups = dn_group_map.nunique()
    n_untyped = int(dn_group_map.astype(str).str.startswith("untyped_").sum())
    print(f"\nDN neurons grouped : {len(dn_group_map):,} DNs -> {n_dn_groups:,} groups "
          f"({n_untyped:,} kept as untyped singletons)")
    print(f"Wing MNs           : {len(mn_group_map):,} MNs -> "
          f"{mn_group_map.nunique():,} muscles")

    dn_total_output = compute_dn_total_output(connections, dn_group_map.index.values)

    long_df, denom_per_dn_group = compute_groupwise_output_fraction(
        connections, dn_group_map, mn_group_map, dn_total_output)

    per_dn_group = summarise_per_dn_group(long_df, denom_per_dn_group)

    matrix = long_df.pivot_table(index="dn_group", columns="muscle",
                                 values="output_fraction", aggfunc="sum", fill_value=0.0)

    thresholded = long_df[long_df["output_fraction"] >= THRESHOLD].copy()

    # --- Diagnostics / subset-trap check ------------------------------------
    nonzero = per_dn_group[per_dn_group["denominator_output"] > 0]
    mean_total = nonzero["total_wing_output_fraction"].mean() if len(nonzero) else 0.0
    median_total = nonzero["total_wing_output_fraction"].median() if len(nonzero) else 0.0
    max_total = per_dn_group["total_wing_output_fraction"].max() if len(per_dn_group) else 0.0
    frac_high = (nonzero["total_wing_output_fraction"] > SUBSET_TRAP_WARN_LEVEL).mean() \
        if len(nonzero) else 0.0

    print("\n--- DIAGNOSTIC: summed direct-output fraction to wing muscles, per DN group ---")
    print(f"    mean (DN groups w/ any output) : {mean_total:.3f}")
    print(f"    median                         : {median_total:.3f}")
    print(f"    max across all DN groups       : {max_total:.3f}")
    print(f"    fraction of DN groups > {SUBSET_TRAP_WARN_LEVEL:.2f}     : {frac_high:.1%}")
    print(f"    (no external reference for this number — see docstring; this is a")
    print(f"     heuristic check, not a target to match)")
    if frac_high > SUBSET_TRAP_WARN_GROUP_FRACTION:
        print(f"    [WARNING] {frac_high:.1%} of DN groups dedicate >{SUBSET_TRAP_WARN_LEVEL:.0%} of "
              f"their\n"
              f"              ENTIRE output to these {mn_group_map.nunique()} wing muscles alone. A "
              f"few genuinely\n"
              f"              dedicated command-like DNs are plausible; this many is not.\n"
              f"              connections_princeton.csv may be a SUBSET (e.g. only DN->MN\n"
              f"              edges), which would make TotalOutput(d) too small and inflate\n"
              f"              F_out broadly. Verify it is the FULL BANC edge list.")
    else:
        print("    OK: dedication levels are in a plausible range for a full connectome")
        print("        (most DNs spend most of their output on non-wing-MN targets).")

    # --- Save ---------------------------------------------------------------
    matrix.to_csv(out_dir / "direct_dn_mn_output_fraction_matrix.csv")
    long_df.to_csv(out_dir / "direct_dn_mn_output_fraction_long.csv", index=False)
    thresholded.to_csv(out_dir / "direct_dn_mn_output_thresholded.csv", index=False)
    per_dn_group.to_csv(out_dir / "dn_group_output_totals.csv")

    fig_path = out_dir / "figure_direct_dn_mn_output_fraction.png"
    make_figure(long_df, per_dn_group, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Direct DN->MN groupwise output-fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold (display)          : {THRESHOLD}\n")
        fh.write(f"keep untyped DNs             : {KEEP_UNTYPED_DNS}\n")
        fh.write(f"DN neurons / groups          : {len(dn_group_map)} / {n_dn_groups}\n")
        fh.write(f"  (untyped singletons)       : {n_untyped}\n")
        fh.write(f"wing MNs / muscles           : {len(mn_group_map)} / "
                 f"{mn_group_map.nunique()}\n")
        fh.write(f"nonzero DN->muscle edges     : {len(long_df)}\n")
        fh.write(f"edges >= threshold           : {len(thresholded)}\n")
        fh.write(f"mean summed frac/DN group    : {mean_total:.4f}\n")
        fh.write(f"median summed frac/DN group  : {median_total:.4f}\n")
        fh.write(f"max summed frac/DN group     : {max_total:.4f}\n")
        fh.write(f"frac DN groups > {SUBSET_TRAP_WARN_LEVEL:.2f}        : {frac_high:.4f}\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return long_df, per_dn_group


# =============================================================================
# SELF-TEST  — synthetic connectome with hand-computed answers.
# =============================================================================
def selftest():
    """
    Same tiny connectome as step 01's self-test (so the two scripts can be
    cross-checked against each other), with output-fraction answers hand
    computed here.

    DN group DNx = {101,102}, DN group DNy = {103}
    muscleA = {201,202}, muscleB = {203}, plus an interneuron 999 as 'other'.

    Total OUTPUT per DN (sum over every target this DN has, in this toy
    connectome none of 101/102/103 have any target besides these MNs):
      101 : ->201(10) + ->202(5)  = 15
      102 : ->201(5)              = 5
      103 : ->203(20)             = 20
    Denominators: DNx = 15+5 = 20 , DNy = 20
    Numerators  : DNx->A = 10+5+5 = 20 , DNy->B = 20 ; others = 0
    Expected F_out: DNx->A = 20/20 = 1.0 , DNy->B = 20/20 = 1.0
      (DNx and DNy have NO targets outside these MNs in this toy data, so
      their entire output is "dedicated" to it — a clean check that the
      TotalOutput denominator is grouped/summed correctly by source.)
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
    dn_tot = compute_dn_total_output(connections, dn_map.index.values)
    long_df, denom = compute_groupwise_output_fraction(connections, dn_map, mn_map, dn_tot)

    got = {(r.dn_group, r.muscle): round(r.output_fraction, 6)
           for r in long_df.itertuples()}
    exp = {("DNx", "muscleA"): 1.0, ("DNy", "muscleB"): 1.0}

    print("Self-test results:")
    print(long_df.to_string(index=False))
    assert got == exp, f"\nMISMATCH\n got={got}\n exp={exp}"
    assert round(denom["DNx"], 6) == 20.0
    assert round(denom["DNy"], 6) == 20.0
    print("\n[OK] numerators, denominators and output fractions match hand calc.")

    demo_dir = Path("/tmp/direct_dn_mn_output_selftest")
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
