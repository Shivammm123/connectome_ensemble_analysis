"""
================================================================================
 direct_dn_mn_vnc_output_fraction.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 02b — Direct DN -> MN connectivity, GROUPWISE, VNC-RESTRICTED
            OUTPUT-FRACTION metric.
STATUS    : Same direct, unsigned, single-hop scope as steps 01/02. This is a
            variant of step 02 (direct_dn_mn_output_fraction.py), NOT a
            replacement — both answer different questions, see below.

--------------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS (as distinct from step 02)
--------------------------------------------------------------------------------
Step 02's output fraction divides by a DN's TOTAL output across the WHOLE
connectome (brain + VNC). That answers "of everything this DN does anywhere,
what share goes to wing muscles?" — a fair question, but the brain-side part
of the denominator is doing a lot of the work for BANC's DNs specifically,
and (per CLAUDE.md's cross-dataset-replication notes) makes the number
incomparable to a VNC-only dataset like MANC, where no such brain-side output
exists to inflate the denominator.

THIS script asks a narrower, VNC-local question instead: "of everything this
DN does **within the VNC** — its local premotor/interneuron processing
budget, not counting whatever it also does upstream in the brain — what share
goes specifically to wing muscles?" That's the DN-side complement to how MN
input already works: MN dendrites are physically in the VNC, so step 01's
input-fraction denominator was always VNC-local. Restricting the DN's output
denominator the same way makes this metric apples-to-apples with step 01, and
— per the open cross-dataset item in CLAUDE.md §6 — is what would actually be
comparable if this analysis were repeated on MANC or maleCNS.

                sum over d in G_DN, m in G_MN of  w_VNC(d -> m)
   F_out,VNC(G_DN->G_MN) = -----------------------------------------------
                sum over d in G_DN of  TotalOutput_VNC(d)

  * NUMERATOR   = synapses from the DN group onto the MN group, restricted to
                  synapses physically located in a VNC neuropil. In practice
                  this is nearly identical to step 02's numerator, since
                  motor-neuron dendrites are in the VNC by anatomy — but it's
                  filtered the same way as the denominator for internal
                  consistency (numerator is a strict subset of denominator,
                  so F_out,VNC is cleanly bounded in [0, 1]).
  * DENOMINATOR = the DN group's total output, counting ONLY synapses whose
                  `neuropil` column is VNC-tagged (see below) — i.e. only the
                  DN's local VNC arbor, not its brain-side dendritic/axonal
                  output.

HOW "VNC" IS DEFINED HERE
   `connections_princeton.csv` tags every synapse's physical location with a
   `neuropil` column (114 distinct values in this dataset). Every VNC region
   is consistently prefixed `VNC_` (e.g. `VNC_T1_ProNm_R`, `VNC_IntTct_L`,
   `VNC_unassigned_R` — the unassigned bucket still means "somewhere in the
   VNC, sub-region not resolved", so it's kept in). Verified against the real
   file (2026-08-21): VNC-tagged synapses are 36.6% of the whole connectome
   (8.14M / 22.2M). Excluded: `neck_unassigned_L/R` (the brain-VNC connective;
   ambiguous, and negligible — 872 synapses, 0.004% of the total) and every
   `brain_unassigned_*` / named brain / optic-lobe neuropil.
   `is_vnc = connections["neuropil"].str.startswith("VNC_")` is the entire
   filter — no neuron-level classification needed, because the location is
   already recorded per-synapse.

WHAT CHANGED FROM STEP 02, MECHANICALLY
   Nothing about the grouping or the numerator/denominator MATH changed —
   only the input dataframe: everything downstream operates on
   `connections[connections["neuropil"].str.startswith("VNC_")]` instead of
   the full `connections`. A DN with substantial brain-side output will have
   a SMALLER denominator here than in step 02, so F_out,VNC >= F_out for the
   same edge, sometimes by a lot.

SUBSET TRAP, ONE MORE MIRROR
   Same failure mode as step 02 (if connections_princeton.csv were filtered
   to only DN->MN edges, VNC output would look artificially concentrated) —
   plus a NEW one specific to this script: if `neuropil` values were ever
   missing/malformed for a big chunk of real VNC synapses (e.g. logged as
   NaN or an unexpected label), this script would silently undercount VNC
   output and inflate F_out,VNC across the board. The run prints how many
   connectome rows/synapses are VNC-tagged as a first check — compare that
   number if you re-run against an updated data pull.

--------------------------------------------------------------------------------
GROUPING  — identical to steps 01/02
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

OUTPUTS  (written to results/direct_dn_mn_vnc_output_fraction/)
  direct_dn_mn_vnc_output_fraction_matrix.csv   DN-group x muscle matrix of F_out,VNC
  direct_dn_mn_vnc_output_fraction_long.csv     long form, non-zero edges only
  direct_dn_mn_vnc_output_thresholded.csv       edges with F_out,VNC >= THRESHOLD
  dn_group_vnc_output_totals.csv                per-DN-group denominator + total wing frac
  figure_direct_dn_mn_vnc_output_fraction.png   heatmap + formula panel + sanity bar
  run_summary.txt                               counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/direct_dn_mn_vnc_output_fraction.py             # real data
  python src/direct_dn_mn_vnc_output_fraction.py --selftest   # synthetic check

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "direct_dn_mn_vnc_output_fraction"

CONNECTIONS_CSV = DATA_DIR / "raw" / "connections_princeton.csv"
NEURONS_CSV     = DATA_DIR / "raw" / "neurons.csv"
MOTOR_POOLS_CSV = DATA_DIR / "processed" / "motor_pools" / "motor_pools.csv"


# =============================================================================
# COLUMN NAMES
# =============================================================================
COL_ROOT_ID     = "Root ID"
COL_SUPER_CLASS = "Super Class"
COL_CELL_TYPE   = "Primary Cell Type"

CONN_SOURCE   = "pre_root_id"
CONN_TARGET   = "post_root_id"
CONN_WEIGHT   = "syn_count"
CONN_NEUROPIL = "neuropil"

DN_SUPER_CLASS = "descending"

# Every VNC neuropil in this dataset is prefixed this way (verified against
# the real file: 114 distinct neuropil values, VNC ones all match this
# prefix). `VNC_unassigned_*` still counts — sub-region unresolved, but still
# physically in the VNC. `neck_unassigned_*` (the connective) and
# `brain_unassigned_*` do NOT match and are correctly excluded.
VNC_NEUROPIL_PREFIX = "VNC_"


# =============================================================================
# PARAMETERS
# =============================================================================
THRESHOLD = 0.01
KEEP_UNTYPED_DNS = True
SUBSET_TRAP_WARN_LEVEL = 0.50
SUBSET_TRAP_WARN_GROUP_FRACTION = 0.10

# Figure sizing: this metric routinely has 100+ DN groups clearing THRESHOLD
# somewhere. Cap the heatmap to the top TOP_N_ROWS by strongest connection —
# the full, untruncated matrix is still in the saved CSVs regardless. The
# figure itself is fixed at 16:9 (13.33 x 7.5in) to drop onto a slide as-is.
TOP_N_ROWS = 25

# --- Figure style: validated palette (dataviz skill's references/palette.md) ---
CAT_BLUE, CAT_ORANGE, CAT_AQUA, CAT_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CAT_MAGENTA, CAT_GREEN, CAT_VIOLET, CAT_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_HAIRLINE, AXIS_LINE = "#e1e0d9", "#c3c2b7"
SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                         "#256abf", "#184f95", "#0d366b"]


# =============================================================================
# CORE COMPUTATION  — identical in shape to step 02; only the connections
# dataframe passed in differs (pre-filtered to VNC neuropils by the caller).
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


def filter_to_vnc(connections: pd.DataFrame) -> pd.DataFrame:
    """
    The one function that makes this script different from step 02: restrict
    every downstream computation to synapses physically located in a VNC
    neuropil. See module docstring for what counts as VNC and why.
    """
    is_vnc = connections[CONN_NEUROPIL].astype(str).str.startswith(VNC_NEUROPIL_PREFIX)
    return connections.loc[is_vnc].copy()


def compute_dn_total_output(connections: pd.DataFrame,
                            dn_ids: np.ndarray) -> pd.Series:
    """DENOMINATOR builder. `connections` here is already VNC-filtered."""
    total_output_all = connections.groupby(CONN_SOURCE)[CONN_WEIGHT].sum()
    return total_output_all.reindex(dn_ids).fillna(0.0)


def compute_groupwise_output_fraction(connections: pd.DataFrame,
                                      dn_group_map: pd.Series,
                                      mn_group_map: pd.Series,
                                      dn_total_output: pd.Series) -> pd.DataFrame:
    """Same logic as step 02. `connections` here is already VNC-filtered, so
    both numerator and denominator are computed on the same restricted set —
    numerator is a strict subset of denominator, keeping F_out,VNC in [0,1]."""
    dn_ids = set(dn_group_map.index)
    mn_ids = set(mn_group_map.index)

    mask = connections[CONN_SOURCE].isin(dn_ids) & connections[CONN_TARGET].isin(mn_ids)
    dn_mn = connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()

    dn_mn["dn_group"] = dn_mn[CONN_SOURCE].map(dn_group_map)
    dn_mn["muscle"]   = dn_mn[CONN_TARGET].map(mn_group_map)

    numerator = (dn_mn.groupby(["dn_group", "muscle"])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())

    denom_per_dn_group = (dn_total_output.groupby(dn_group_map).sum()
                          .rename("denominator_output"))

    out = numerator.merge(denom_per_dn_group, left_on="dn_group", right_index=True, how="left")
    out["output_fraction"] = out["numerator_synapses"] / out["denominator_output"]
    out = out.sort_values("output_fraction", ascending=False).reset_index(drop=True)
    return out, denom_per_dn_group


def summarise_per_dn_group(long_df: pd.DataFrame,
                           denom_per_dn_group: pd.Series) -> pd.DataFrame:
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

    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis("off")
    formula = (
        r"$F_{\mathrm{out,VNC}}(G_{DN}\!\rightarrow\!G_{MN}) \;=\; "
        r"\dfrac{\sum_{d\in G_{DN}}\sum_{m\in G_{MN}} w_{\mathrm{VNC}}(d\rightarrow m)}"
        r"{\sum_{d\in G_{DN}} \mathrm{TotalOutput}_{\mathrm{VNC}}(d)}$"
    )
    ax_formula.text(0.02, 0.55, formula, fontsize=13, va="center", color=INK)
    ax_formula.text(0.50, 0.55,
                    "numerator: synapses DN type $\\rightarrow$ muscle, VNC neuropils only\n"
                    "denominator: DN type's TOTAL output, VNC neuropils only\n"
                    "(excludes brain-side output — see script docstring)",
                    fontsize=9, va="center", color=INK_SOFT)
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
            ylabel = (f"DN group  (top {len(mat_shown)} of {n_above_threshold} with max "
                      f"$F_{{\\mathrm{{out,VNC}}}}$ $\\geq$ {THRESHOLD:g})")
        else:
            ylabel = (f"DN group  (max $F_{{\\mathrm{{out,VNC}}}}$ across muscles "
                      f"$\\geq$ {THRESHOLD:g})")
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", color=INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("VNC output fraction $F_{\\mathrm{out,VNC}}$", fontsize=8, color=INK)
        cbar.ax.tick_params(colors=AXIS_LINE, labelcolor=INK_SOFT)
        cbar.outline.set_edgecolor(AXIS_LINE)

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
    axb.set_xlabel("total VNC output fraction\nto wing muscles (all 18)", fontsize=8,
                  fontweight="bold", color=INK)
    axb.set_title("sanity check", fontsize=9, color=INK)

    fig.suptitle("Direct descending $\\rightarrow$ wing motor connectivity  ·  "
                 "groupwise VNC-restricted output fraction", fontsize=12.5,
                 fontweight="bold", y=0.985, color=INK)
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

    # --- restrict to VNC before anything else touches `connections` --------
    n_rows_total, syn_total = len(connections), connections[CONN_WEIGHT].sum()
    vnc_connections = filter_to_vnc(connections)
    n_rows_vnc, syn_vnc = len(vnc_connections), vnc_connections[CONN_WEIGHT].sum()
    print(f"\nVNC-restriction    : {n_rows_vnc:,}/{n_rows_total:,} rows "
          f"({n_rows_vnc / n_rows_total:.1%}), "
          f"{syn_vnc:,.0f}/{syn_total:,.0f} synapses ({syn_vnc / syn_total:.1%}) kept")

    dn_total_output = compute_dn_total_output(vnc_connections, dn_group_map.index.values)
    n_dn_zero_vnc = int((dn_total_output == 0.0).sum())
    print(f"DN groups' member neurons with ZERO VNC output: {n_dn_zero_vnc:,} / "
          f"{len(dn_total_output):,} individual DNs "
          f"(a 'descending' neuron with no VNC output at all is unusual — "
          f"check reconstruction if this is large)")

    long_df, denom_per_dn_group = compute_groupwise_output_fraction(
        vnc_connections, dn_group_map, mn_group_map, dn_total_output)

    per_dn_group = summarise_per_dn_group(long_df, denom_per_dn_group)

    matrix = long_df.pivot_table(index="dn_group", columns="muscle",
                                 values="output_fraction", aggfunc="sum", fill_value=0.0)

    thresholded = long_df[long_df["output_fraction"] >= THRESHOLD].copy()

    nonzero = per_dn_group[per_dn_group["denominator_output"] > 0]
    mean_total = nonzero["total_wing_output_fraction"].mean() if len(nonzero) else 0.0
    median_total = nonzero["total_wing_output_fraction"].median() if len(nonzero) else 0.0
    max_total = per_dn_group["total_wing_output_fraction"].max() if len(per_dn_group) else 0.0
    frac_high = (nonzero["total_wing_output_fraction"] > SUBSET_TRAP_WARN_LEVEL).mean() \
        if len(nonzero) else 0.0

    print("\n--- DIAGNOSTIC: summed direct VNC-output fraction to wing muscles, per DN group ---")
    print(f"    mean (DN groups w/ any VNC output) : {mean_total:.3f}")
    print(f"    median                              : {median_total:.3f}")
    print(f"    max across all DN groups            : {max_total:.3f}")
    print(f"    fraction of DN groups > {SUBSET_TRAP_WARN_LEVEL:.2f}          : {frac_high:.1%}")
    print(f"    (no external reference for this number — heuristic check, not a target)")
    if frac_high > SUBSET_TRAP_WARN_GROUP_FRACTION:
        print(f"    [WARNING] {frac_high:.1%} of DN groups dedicate >{SUBSET_TRAP_WARN_LEVEL:.0%} of "
              f"their\n"
              f"              ENTIRE VNC output to these {mn_group_map.nunique()} wing muscles alone.\n"
              f"              Verify the neuropil tagging / connections file completeness.")
    else:
        print("    OK: dedication levels are in a plausible range.")

    matrix.to_csv(out_dir / "direct_dn_mn_vnc_output_fraction_matrix.csv")
    long_df.to_csv(out_dir / "direct_dn_mn_vnc_output_fraction_long.csv", index=False)
    thresholded.to_csv(out_dir / "direct_dn_mn_vnc_output_thresholded.csv", index=False)
    per_dn_group.to_csv(out_dir / "dn_group_vnc_output_totals.csv")

    fig_path = out_dir / "figure_direct_dn_mn_vnc_output_fraction.png"
    make_figure(long_df, per_dn_group, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Direct DN->MN groupwise VNC-restricted output-fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold (display)          : {THRESHOLD}\n")
        fh.write(f"keep untyped DNs             : {KEEP_UNTYPED_DNS}\n")
        fh.write(f"DN neurons / groups          : {len(dn_group_map)} / {n_dn_groups}\n")
        fh.write(f"  (untyped singletons)       : {n_untyped}\n")
        fh.write(f"wing MNs / muscles           : {len(mn_group_map)} / "
                 f"{mn_group_map.nunique()}\n")
        fh.write(f"connections rows kept as VNC : {n_rows_vnc} / {n_rows_total} "
                 f"({n_rows_vnc / n_rows_total:.4f})\n")
        fh.write(f"synapses kept as VNC         : {syn_vnc:.0f} / {syn_total:.0f} "
                 f"({syn_vnc / syn_total:.4f})\n")
        fh.write(f"DN neurons w/ zero VNC output: {n_dn_zero_vnc}\n")
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
# SELF-TEST
# =============================================================================
def selftest():
    """
    Step 02's synthetic connectome, PLUS one extra brain-side edge from DN
    101 (101 -> 888, a generic non-MN, non-DN neuron, tagged with a brain
    neuropil "AL_R") to actually exercise the VNC filter.

    Total output per DN, VNC-restricted (the AL_R edge is excluded):
      101 : ->201(10,VNC) + ->202(5,VNC)             = 15   (NOT +25 AL_R)
      102 : ->201(5,VNC)                              = 5
      103 : ->203(20,VNC)                             = 20
    Denominators: DNx = 15+5 = 20 , DNy = 20   (same as step 02's answer,
      because DNy has no brain-side edge and DNx's only extra edge is
      correctly excluded here)
    Numerators (unchanged, all MN-facing edges are VNC by construction):
      DNx->A = 10+5+5 = 20 , DNy->B = 20
    Expected F_out,VNC: DNx->A = 20/20 = 1.0 , DNy->B = 20/20 = 1.0

    Contrast: if you ran step 02 (whole-connectome output fraction) on THIS
    modified dataset, DNx's denominator would be 15+5+25(AL_R) + 5 = 45, i.e.
    F_out = 20/45 = 0.444 — clearly different from 1.0. This self-test's
    point is exactly that: VNC-restriction recovers 1.0 here by correctly
    excluding the AL_R edge from the denominator, which a whole-connectome
    calculation would not do. If this assert passes, the filter works.
    """
    connections = pd.DataFrame({
        CONN_SOURCE:   [101,   101,   102,   103,   999,   999,   999,   101],
        CONN_TARGET:   [201,   202,   201,   203,   201,   202,   203,   888],
        CONN_WEIGHT:   [10,    5,     5,     20,    30,    15,    30,    25],
        CONN_NEUROPIL: ["VNC_test"] * 7 + ["AL_R"],
    })
    neurons = pd.DataFrame({
        COL_ROOT_ID:     [101, 102, 103, 201, 202, 203, 999, 888],
        COL_SUPER_CLASS: ["descending", "descending", "descending",
                          "motor", "motor", "motor", "intrinsic", "central_brain_intrinsic"],
        COL_CELL_TYPE:   ["DNx", "DNx", "DNy", None, None, None, "someIN", "someBrainIN"],
    })
    motor_pools = pd.DataFrame({
        "muscle": ["muscleA", "muscleB"],
        "motor_neuron_ids": ["[201, 202]", "[203]"],
    })

    dn_map = build_dn_groups(neurons)
    mn_map = build_mn_groups(motor_pools)
    vnc_connections = filter_to_vnc(connections)
    print(f"Rows before VNC filter: {len(connections)}, after: {len(vnc_connections)} "
          f"(the AL_R row should be dropped)")
    assert len(vnc_connections) == 7, "VNC filter should drop exactly the one AL_R row"

    dn_tot = compute_dn_total_output(vnc_connections, dn_map.index.values)
    long_df, denom = compute_groupwise_output_fraction(vnc_connections, dn_map, mn_map, dn_tot)

    got = {(r.dn_group, r.muscle): round(r.output_fraction, 6)
           for r in long_df.itertuples()}
    exp = {("DNx", "muscleA"): 1.0, ("DNy", "muscleB"): 1.0}

    print("Self-test results:")
    print(long_df.to_string(index=False))
    assert got == exp, f"\nMISMATCH\n got={got}\n exp={exp}"
    assert round(denom["DNx"], 6) == 20.0
    assert round(denom["DNy"], 6) == 20.0
    print("\n[OK] VNC filter drops the brain-side edge, numerators/denominators/"
          "fractions match hand calc.")

    demo_dir = Path("/tmp/direct_dn_mn_vnc_output_selftest")
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
