"""
================================================================================
 direct_dn_mn_geometric_mean.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 03 — Direct DN -> MN connectivity, GROUPWISE, GEOMETRIC-MEAN of
            input fraction and (VNC-restricted) output fraction.
STATUS    : Combines steps 01 and 02b. Same direct, unsigned, single-hop
            scope. This script is self-contained (recomputes both component
            fractions from raw data rather than reading steps 01/02b's saved
            CSVs) — same "each script runs standalone" convention as the rest
            of this pipeline.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES, AND WHY
--------------------------------------------------------------------------------
Step 01 asks "what fraction of this muscle's input comes from this DN?"
(influence — target's-eye view). Step 02b asks "what fraction of this DN's
VNC output goes to this muscle?" (dedication — source's-eye view, VNC-
restricted so it's on the same VNC-local footing as step 01's denominator).
Each alone can be misleading in opposite ways:
  * High F_in, low F_out: a DN with an enormous total VNC output budget can
    supply a big *share* of one muscle's input while that muscle is a tiny
    slice of what the DN actually does — its apparent "influence" may not
    reflect any real functional specialisation for that muscle.
  * High F_out, low F_in: a small DN can dedicate all its output to one
    muscle, yet if that muscle gets most of its input from elsewhere, this
    DN barely moves the muscle's total drive.

Cheong et al. 2024 used the geometric mean of the two fractions to surface
edges that are strong from BOTH perspectives at once — their pathway-
exploration metric. This script reproduces that:

   F_geom(G_DN->G_MN) = sqrt( F_in(G_DN->G_MN) * F_out,VNC(G_DN->G_MN) )

F_geom is a pathway/edge-level score, not a per-group total the way steps
01/02b's summed diagnostics were — it doesn't have a clean "sums to <= 1"
interpretation, so this script's main output is a RANKED EDGE LIST (top
wing-control pathways), which is what the geometric mean is actually for.

NOTE ON WHAT "PASSES THRESHOLD" MEANS HERE
   sqrt(a*b) >= T does NOT require both a >= T and b >= T — one very strong
   side can still pull the geometric mean over the line even if the other is
   weak (e.g. F_in=0.50, F_out=0.0002 -> sqrt(0.0001)=0.01, which clears a
   0.01 threshold despite F_out being nowhere near it). The geometric mean
   *dampens* single-sided strength relative to either raw fraction, and
   rewards edges where both sides are at least moderately large — it doesn't
   enforce a strict AND. This script prints how many edges clear THRESHOLD
   under F_in alone, F_out,VNC alone, both individually, and under F_geom, so
   you can see the actual narrowing effect rather than assume one.

WHY THE OUTPUT-FRACTION SIDE IS VNC-RESTRICTED (step 02b), NOT WHOLE-CONNECTOME
(step 02)
   Step 01's F_in denominator (MN total input) is inherently VNC-local — wing
   motor neuron dendrites are physically in the VNC (verified: 100% of wing-MN
   input synapses in this dataset are VNC-tagged, checked 2026-08-21). Pairing
   it with step 02's whole-connectome (brain+VNC) output fraction would mix a
   VNC-local quantity with a brain-inclusive one — a real asymmetry, not just
   a stylistic mismatch. Step 02b's VNC-restricted output fraction keeps both
   sides on the same VNC-local footing, which is also the version that would
   be comparable if this analysis were later repeated on a VNC-only dataset
   (MANC). This script therefore restricts to VNC-tagged connections ONCE, up
   front, and computes BOTH fractions from that same restricted edge set —
   confirmed to leave F_in numerically unchanged from step 01's original
   whole-connectome computation, since wing-MN input is already 100% VNC.

--------------------------------------------------------------------------------
GROUPING  — identical to steps 01/02/02b
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

OUTPUTS  (written to results/direct_dn_mn_geometric_mean/)
  direct_dn_mn_geometric_mean_matrix.csv   DN-group x muscle matrix of F_geom
  direct_dn_mn_geometric_mean_long.csv     long form (all 3 fractions), non-zero
                                            edges only, sorted by F_geom desc —
                                            this IS the ranked pathway list
  direct_dn_mn_geometric_mean_thresholded.csv   edges with F_geom >= THRESHOLD
  metric_agreement_summary.csv             how many edges clear THRESHOLD under
                                            F_in alone / F_out,VNC alone / both /
                                            F_geom — the narrowing-effect check
  figure_direct_dn_mn_geometric_mean.png   heatmap + formula panel + top-pathways bar
  run_summary.txt                          counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  python src/direct_dn_mn_geometric_mean.py             # real data
  python src/direct_dn_mn_geometric_mean.py --selftest   # synthetic check

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "direct_dn_mn_geometric_mean"

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

# See step 02b for the full rationale — every VNC neuropil is prefixed this
# way; `VNC_unassigned_*` counts (still physically VNC), `neck_unassigned_*`
# and `brain_unassigned_*` do not match and are correctly excluded.
VNC_NEUROPIL_PREFIX = "VNC_"


# =============================================================================
# PARAMETERS
# =============================================================================
# Same 0.01 used throughout steps 01/02/02b. Geometric mean has a convenient
# property here: sqrt(T * T) = T, so an edge sitting exactly at THRESHOLD on
# BOTH component fractions also sits at THRESHOLD on F_geom — the cutoff
# stays on the same scale as the components, it isn't an arbitrary new number.
THRESHOLD = 0.01

KEEP_UNTYPED_DNS = True

# Figure sizing — same convention as steps 01/02/02b: cap the heatmap to the
# top TOP_N_ROWS DN groups (by strongest F_geom edge) so the figure stays
# slide-sized regardless of how many groups clear THRESHOLD. Full matrix is
# still in the saved CSVs. Figure fixed at 16:9 (13.33 x 7.5in).
TOP_N_ROWS = 25

# How many individual (DN_group, muscle) pathways to show in the figure's
# ranked-pathway panel.
TOP_N_PATHWAYS_SHOWN = 20


# =============================================================================
# CORE COMPUTATION
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
    """Restrict to synapses physically located in a VNC neuropil. See
    step 02b / this script's docstring for what counts as VNC and why both
    component fractions are computed from this same restricted set."""
    is_vnc = connections[CONN_NEUROPIL].astype(str).str.startswith(VNC_NEUROPIL_PREFIX)
    return connections.loc[is_vnc].copy()


def compute_mn_total_input(vnc_connections: pd.DataFrame,
                           mn_ids: np.ndarray) -> pd.Series:
    """F_in's denominator builder, VNC-restricted (see docstring: this is
    numerically identical to step 01's whole-connectome version for wing
    MNs, since their input is already 100% VNC)."""
    total_input_all = vnc_connections.groupby(CONN_TARGET)[CONN_WEIGHT].sum()
    return total_input_all.reindex(mn_ids).fillna(0.0)


def compute_dn_total_output(vnc_connections: pd.DataFrame,
                            dn_ids: np.ndarray) -> pd.Series:
    """F_out,VNC's denominator builder — identical to step 02b."""
    total_output_all = vnc_connections.groupby(CONN_SOURCE)[CONN_WEIGHT].sum()
    return total_output_all.reindex(dn_ids).fillna(0.0)


def compute_groupwise_geometric_mean(vnc_connections: pd.DataFrame,
                                     dn_group_map: pd.Series,
                                     mn_group_map: pd.Series,
                                     mn_total_input: pd.Series,
                                     dn_total_output: pd.Series) -> pd.DataFrame:
    """
    Return a long-form DataFrame with one row per (dn_group, muscle) that has
    at least one direct synapse, containing both component fractions and
    their geometric mean:
        dn_group, muscle, numerator_synapses,
        denominator_input, input_fraction,
        denominator_output, output_fraction,
        geometric_mean_fraction
    """
    dn_ids = set(dn_group_map.index)
    mn_ids = set(mn_group_map.index)

    mask = vnc_connections[CONN_SOURCE].isin(dn_ids) & vnc_connections[CONN_TARGET].isin(mn_ids)
    dn_mn = vnc_connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()
    dn_mn["dn_group"] = dn_mn[CONN_SOURCE].map(dn_group_map)
    dn_mn["muscle"]   = dn_mn[CONN_TARGET].map(mn_group_map)

    numerator = (dn_mn.groupby(["dn_group", "muscle"])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())

    denom_in_per_muscle = (mn_total_input.groupby(mn_group_map).sum()
                           .rename("denominator_input"))
    denom_out_per_dn = (dn_total_output.groupby(dn_group_map).sum()
                        .rename("denominator_output"))

    out = numerator.merge(denom_in_per_muscle, left_on="muscle", right_index=True, how="left")
    out = out.merge(denom_out_per_dn, left_on="dn_group", right_index=True, how="left")
    out["input_fraction"]  = out["numerator_synapses"] / out["denominator_input"]
    out["output_fraction"] = out["numerator_synapses"] / out["denominator_output"]
    out["geometric_mean_fraction"] = np.sqrt(out["input_fraction"] * out["output_fraction"])
    out = out.sort_values("geometric_mean_fraction", ascending=False).reset_index(drop=True)
    return out


def metric_agreement_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    The narrowing-effect check described in the docstring: how many edges
    clear THRESHOLD under each metric alone vs both individually vs F_geom.
    Confirms (or not) that combining actually does what it's meant to,
    rather than assuming sqrt(a*b) >= T behaves like "a >= T AND b >= T".
    """
    in_pass   = long_df["input_fraction"] >= THRESHOLD
    out_pass  = long_df["output_fraction"] >= THRESHOLD
    both_pass = in_pass & out_pass
    geom_pass = long_df["geometric_mean_fraction"] >= THRESHOLD
    rows = [
        ("input_fraction (F_in) >= threshold",           int(in_pass.sum())),
        ("output_fraction (F_out,VNC) >= threshold",     int(out_pass.sum())),
        ("both individually >= threshold",                int(both_pass.sum())),
        ("geometric_mean_fraction (F_geom) >= threshold", int(geom_pass.sum())),
        ("F_geom pass but NOT both individually",         int((geom_pass & ~both_pass).sum())),
        ("both individually pass but F_geom does not",    int((both_pass & ~geom_pass).sum())),
    ]
    return pd.DataFrame(rows, columns=["criterion", "n_edges"])


# =============================================================================
# FIGURE  — fixed at 16:9 (13.33 x 7.5in) so it drops straight onto a
#           widescreen slide no matter how many DN groups clear the
#           threshold — see TOP_N_ROWS above for how the row count is capped.
# =============================================================================
def make_figure(long_df: pd.DataFrame, out_path: Path, demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mat = long_df.pivot_table(index="dn_group", columns="muscle",
                              values="geometric_mean_fraction", aggfunc="sum", fill_value=0.0)
    keep_rows = mat.max(axis=1) >= THRESHOLD
    mat_shown = mat.loc[keep_rows]
    mat_shown = mat_shown.loc[mat_shown.max(axis=1).sort_values(ascending=False).index]
    n_above_threshold = mat_shown.shape[0]
    truncated = n_above_threshold > TOP_N_ROWS
    if truncated:
        mat_shown = mat_shown.iloc[:TOP_N_ROWS]

    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    gs = GridSpec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1.0, 5.2],
                  hspace=0.5, wspace=0.32, top=0.90, bottom=0.09,
                  left=0.06, right=0.99)

    # --- Formula / methods panel (top) --------------------------------------
    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis("off")
    formula = (
        r"$F_{\mathrm{geom}}(G_{DN}\!\rightarrow\!G_{MN}) \;=\; "
        r"\sqrt{\,F_{\mathrm{in}}(G_{DN}\!\rightarrow\!G_{MN}) "
        r"\cdot F_{\mathrm{out,VNC}}(G_{DN}\!\rightarrow\!G_{MN})\,}$"
    )
    ax_formula.text(0.02, 0.55, formula, fontsize=14, va="center")
    ax_formula.text(0.50, 0.55,
                    "$F_{\\mathrm{in}}$: share of muscle's input from this DN type (step 01)\n"
                    "$F_{\\mathrm{out,VNC}}$: share of DN type's VNC output to this muscle (step 02b)\n"
                    "high only when the edge is strong from BOTH sides",
                    fontsize=9, va="center", color="#444444")
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
        im = ax.imshow(mat_shown.values, aspect="auto", cmap="magma_r",
                       vmin=0.0, vmax=max(mat_shown.values.max(), THRESHOLD))
        ax.set_xticks(range(mat_shown.shape[1]))
        ax.set_xticklabels(mat_shown.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(mat_shown.shape[0]))
        ax.set_yticklabels(mat_shown.index, fontsize=7)
        ax.set_xlabel("Wing motor-neuron group (muscle)", fontsize=9, fontweight="bold")
        if truncated:
            ylabel = (f"DN group  (top {len(mat_shown)} of {n_above_threshold} with max "
                      f"$F_{{\\mathrm{{geom}}}}$ $\\geq$ {THRESHOLD:g})")
        else:
            ylabel = f"DN group  (max $F_{{\\mathrm{{geom}}}}$ across muscles $\\geq$ {THRESHOLD:g})"
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("geometric mean $F_{\\mathrm{geom}}$", fontsize=8)

    # --- Top pathways (bottom-right) : ranked edge list, the actual point --
    # of this metric (Cheong used it for pathway exploration, not group
    # totals) — so the companion panel here is a ranked list, not a summed
    # per-group bar like steps 01/02/02b's "sanity check".
    axb = fig.add_subplot(gs[1, 1])
    top_edges = long_df.sort_values("geometric_mean_fraction", ascending=False) \
        .head(TOP_N_PATHWAYS_SHOWN).iloc[::-1]  # reverse so strongest is at top of barh
    labels = [f"{r.dn_group}$\\rightarrow${r.muscle}" for r in top_edges.itertuples()]
    axb.barh(range(len(top_edges)), top_edges["geometric_mean_fraction"].values,
             color="#3B6EA5", edgecolor="black", linewidth=0.4)
    axb.set_yticks(range(len(top_edges)))
    axb.set_yticklabels(labels, fontsize=6)
    axb.set_xlabel("$F_{\\mathrm{geom}}$", fontsize=8, fontweight="bold")
    axb.set_title(f"top {len(top_edges)} pathways", fontsize=9)

    fig.suptitle("Direct descending $\\rightarrow$ wing motor connectivity  ·  "
                 "geometric mean of input \\& VNC output fraction",
                 fontsize=12.5, fontweight="bold", y=0.985)
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

    vnc_connections = filter_to_vnc(connections)
    print(f"VNC-restriction    : {len(vnc_connections):,}/{len(connections):,} rows kept "
          f"(both component fractions computed from this same restricted set)")

    mn_total_input  = compute_mn_total_input(vnc_connections, mn_group_map.index.values)
    dn_total_output = compute_dn_total_output(vnc_connections, dn_group_map.index.values)

    long_df = compute_groupwise_geometric_mean(
        vnc_connections, dn_group_map, mn_group_map, mn_total_input, dn_total_output)

    matrix = long_df.pivot_table(index="dn_group", columns="muscle",
                                 values="geometric_mean_fraction", aggfunc="sum", fill_value=0.0)
    thresholded = long_df[long_df["geometric_mean_fraction"] >= THRESHOLD].copy()
    agreement = metric_agreement_summary(long_df)

    print(f"\n--- DIAGNOSTIC: metric agreement (of {len(long_df)} nonzero direct edges) ---")
    print(agreement.to_string(index=False))

    print(f"\n--- Top {min(15, len(long_df))} pathways by geometric mean ---")
    print(long_df.head(15)[["dn_group", "muscle", "input_fraction", "output_fraction",
                            "geometric_mean_fraction"]].to_string(index=False))

    matrix.to_csv(out_dir / "direct_dn_mn_geometric_mean_matrix.csv")
    long_df.to_csv(out_dir / "direct_dn_mn_geometric_mean_long.csv", index=False)
    thresholded.to_csv(out_dir / "direct_dn_mn_geometric_mean_thresholded.csv", index=False)
    agreement.to_csv(out_dir / "metric_agreement_summary.csv", index=False)

    fig_path = out_dir / "figure_direct_dn_mn_geometric_mean.png"
    make_figure(long_df, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Direct DN->MN groupwise geometric-mean fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold (display)          : {THRESHOLD}\n")
        fh.write(f"keep untyped DNs             : {KEEP_UNTYPED_DNS}\n")
        fh.write(f"DN neurons / groups          : {len(dn_group_map)} / {n_dn_groups}\n")
        fh.write(f"  (untyped singletons)       : {n_untyped}\n")
        fh.write(f"wing MNs / muscles           : {len(mn_group_map)} / "
                 f"{mn_group_map.nunique()}\n")
        fh.write(f"VNC rows kept                : {len(vnc_connections)} / {len(connections)}\n")
        fh.write(f"nonzero DN->muscle edges     : {len(long_df)}\n")
        fh.write(f"edges >= threshold (F_geom)  : {len(thresholded)}\n\n")
        fh.write("metric agreement:\n")
        fh.write(agreement.to_string(index=False))
        fh.write("\n\ntop 15 pathways by geometric mean:\n")
        fh.write(long_df.head(15)[["dn_group", "muscle", "input_fraction", "output_fraction",
                                   "geometric_mean_fraction"]].to_string(index=False))
        fh.write("\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return long_df


# =============================================================================
# SELF-TEST
# =============================================================================
def selftest():
    """
    Same synthetic connectome as step 02b (includes the one brain-tagged
    edge, to confirm the VNC filter still applies here too), with geometric
    mean hand-computed on top of the already-verified component fractions.

    From step 02b's hand calc (VNC-restricted, brain edge excluded):
      Denominators: input  -> muscleA=65, muscleB=50  (same as step 01 exactly,
                              since wing-MN input is 100% VNC anyway)
                    output -> DNx=20, DNy=20
      Numerators  : DNx->A = 20 , DNy->B = 20

    F_in(DNx->A)  = 20/65 = 0.307692...
    F_out(DNx->A) = 20/20 = 1.0
    F_geom(DNx->A) = sqrt(0.307692... * 1.0) = 0.554700 (6dp)

    F_in(DNy->B)  = 20/50 = 0.4
    F_out(DNy->B) = 20/20 = 1.0
    F_geom(DNy->B) = sqrt(0.4 * 1.0) = 0.632456 (6dp)

    Unlike steps 02/02b's self-test (where both toy edges landed at exactly
    1.0 because the synthetic DNs had no other output), this one exercises a
    genuinely different value on each side of the sqrt for DNx->A, since its
    F_in (0.308) and F_out (1.0) differ substantially.
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
    mn_tot = compute_mn_total_input(vnc_connections, mn_map.index.values)
    dn_tot = compute_dn_total_output(vnc_connections, dn_map.index.values)
    long_df = compute_groupwise_geometric_mean(vnc_connections, dn_map, mn_map, mn_tot, dn_tot)

    print("Self-test results:")
    print(long_df.to_string(index=False))

    got = {(r.dn_group, r.muscle): (round(r.input_fraction, 6),
                                     round(r.output_fraction, 6),
                                     round(r.geometric_mean_fraction, 6))
           for r in long_df.itertuples()}
    exp = {
        ("DNx", "muscleA"): (0.307692, 1.0, 0.554700),
        ("DNy", "muscleB"): (0.4, 1.0, 0.632456),
    }
    assert got == exp, f"\nMISMATCH\n got={got}\n exp={exp}"
    print("\n[OK] input fraction, output fraction, and geometric mean all match hand calc.")

    agreement = metric_agreement_summary(long_df)
    print("\nMetric agreement on toy data:")
    print(agreement.to_string(index=False))

    demo_dir = Path("/tmp/direct_dn_mn_geometric_mean_selftest")
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
