"""
================================================================================
 direct_dn_mn_input_fraction.py
================================================================================

PROJECT   : Wing motor-control connectome (BANC) — redesigned pipeline
STEP      : 01 — Direct DN -> MN connectivity, GROUPWISE, INPUT-FRACTION metric
STATUS    : Foundation block. No neurotransmitter signs, no multi-hop paths.
            Raw synapse counts are NOT used as the connection strength anywhere
            in the reported results; they are only ever a numerator that is
            immediately normalised.

--------------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES
--------------------------------------------------------------------------------
For every descending-neuron group (G_DN) and every wing motor-neuron group
(G_MN, i.e. a muscle / motor pool), the *groupwise input fraction*:

                    sum over d in G_DN, m in G_MN of  w(d -> m)
   F_in(G_DN->G_MN) = -------------------------------------------------
                          sum over m in G_MN of  TotalInput(m)

  * NUMERATOR   = all synapses from the DN group onto the MN group.
  * DENOMINATOR = the MN group's *total* input budget — every synapse onto
                  every motor neuron in the group, from ANY source in the WHOLE
                  connectome (not just from DNs, not just from our subset).

INTERPRETATION
   F_in answers "what fraction of this muscle's motor-neuron input is supplied
   *directly* by this DN type?" It is the target's-eye view of influence, which
   is the correct question for 'who controls this motor neuron'. Because the
   denominator is the MN's full input, summing F_in over ALL upstream groups
   for a given muscle is bounded by 1.0.

WHY INPUT FRACTION (and not raw counts, and not output fraction)
   * Raw counts conflate 'strong connection' with 'big neuron'. A 50-synapse
     input to an MN receiving 100 total (50%) matters far more than a
     50-synapse input to an MN receiving 10,000 (0.5%).
   * Output fraction (share of the DN's output budget) measures the DN's
     *dedication*, not its *influence*, and — critically for BANC — is NOT
     comparable across datasets, because a DN's output budget in a full-CNS
     dataset includes all its brain-side synapses that a VNC-only dataset lacks.
     Input fraction uses the MN's input, which is VNC-local in every dataset,
     so it stays comparable.  (Output fraction & geometric mean come in step 2.)

--------------------------------------------------------------------------------
THE ONE ASSUMPTION YOU MUST CHECK (the "subset trap")
--------------------------------------------------------------------------------
The denominator is only correct if `connections_princeton.csv` is the FULL BANC
edge list. If it were pre-filtered to (say) only DN->MN edges, every denominator
would be missing the MN's other inputs, and every fraction would be inflated —
in the extreme, the DN fractions to a muscle would sum to ~1.0.

This script therefore prints a DIAGNOSTIC: the summed direct-DN input fraction
per muscle. Cheong et al. 2024 (MANC) found DNs together supply only ~9-10% of
wing-MN input. If your BANC values land near there, the file is full AND you've
reproduced a known result. If they approach 1.0, the connections file is a
subset and the denominator is wrong — the script warns you.

--------------------------------------------------------------------------------
GROUPING (because we work GROUPWISE)
--------------------------------------------------------------------------------
  * DN groups  = shared 'Cell Type' (bilateral pairs / systematic types share a
                 type name). DNs with no Cell Type are kept as SINGLETON groups
                 labelled by Root ID so nothing is silently dropped; the count
                 is reported. Change KEEP_UNTYPED_DNS to False to exclude them.
  * MN groups  = muscles / motor pools, taken from your existing curated
                 motor_pools.csv (the pilot's muscle_mn_mapping.py output).

--------------------------------------------------------------------------------
INPUTS  (edit the PATHS block below if your layout differs)
--------------------------------------------------------------------------------
  data/connections_princeton.csv   columns: source, target, weight
  data/neurons.csv                 columns: 'Root ID', 'Super Class', 'Cell Type'
  data/processed/motor_pools/motor_pools.csv   columns: muscle, motor_neuron_ids
                                          (motor_neuron_ids = stringified list)

OUTPUTS  (written to results/direct_dn_mn_input_fraction/)
  direct_dn_mn_input_fraction_matrix.csv   DN-group x muscle matrix of F_in
  direct_dn_mn_input_fraction_long.csv     long form, non-zero edges only
  direct_dn_mn_thresholded.csv             edges with F_in >= THRESHOLD
  mn_group_input_totals.csv                per-muscle denominator + total DN frac
  figure_direct_dn_mn_input_fraction.png   heatmap + formula panel + sanity bar
  run_summary.txt                          counts, parameters, diagnostics

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  # normal run against your real data (from anywhere):
  python src/direct_dn_mn_input_fraction.py

  # verify the maths on a tiny synthetic connectome with known answers:
  python src/direct_dn_mn_input_fraction.py --selftest

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
RESULTS_DIR  = PROJECT_ROOT / "results" / "direct_dn_mn_input_fraction"

CONNECTIONS_CSV = DATA_DIR / "raw" / "connections_princeton.csv"
NEURONS_CSV     = DATA_DIR / "raw" / "neurons.csv"
MOTOR_POOLS_CSV = DATA_DIR / "processed" / "motor_pools" / "motor_pools.csv"


# =============================================================================
# COLUMN NAMES  — BANC / Princeton format uses spaces and capitals. Do not
#                 "fix" these to snake_case; that was a recurring pilot bug.
# =============================================================================
COL_ROOT_ID     = "Root ID"
COL_SUPER_CLASS = "Super Class"
COL_CELL_TYPE   = "Primary Cell Type"   # neurons.csv has no bare 'Cell Type' col;
                                        # this is the actual column (verified
                                        # against the real file 2026-08-21).

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
# "Substantial" direct connection threshold on the input fraction.
# Cheong et al. 2024 used >= 1% groupwise input. This is a display/reporting
# threshold only — the full continuous matrix is always saved unthresholded.
THRESHOLD = 0.01

# Keep DNs that have no Cell Type as their own singleton groups (True), or
# drop them entirely (False). Default True so nothing is silently lost.
KEEP_UNTYPED_DNS = True

# Cheong's reference: all DNs together supply ~9-10% of wing-MN input.
# Used only to annotate the sanity-check figure and warn if wildly off.
CHEONG_TOTAL_DN_FRACTION_REF = 0.10
SUBSET_TRAP_WARN_LEVEL = 0.50   # summed DN fraction above this => likely subset


# =============================================================================
# CORE COMPUTATION
# =============================================================================
def build_dn_groups(neurons: pd.DataFrame) -> pd.Series:
    """
    Return a Series mapping DN Root ID -> DN group label.

    Group label = 'Cell Type' when present; otherwise (if KEEP_UNTYPED_DNS)
    a singleton label 'untyped_<RootID>'. DNs are those whose 'Super Class'
    equals DN_SUPER_CLASS.
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

    motor_pools.csv has one row per muscle with a stringified list of MN ids
    in 'motor_neuron_ids'.
    """
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
    """
    DENOMINATOR builder. Total input synapses onto each MN, summed over the
    ENTIRE connectome (every source). This is the step whose correctness
    depends on `connections` being the full edge list.
    """
    # Sum weight over all edges grouped by target = total input per neuron.
    total_input_all = connections.groupby(CONN_TARGET)[CONN_WEIGHT].sum()
    # Restrict to our MNs; MNs with no input at all get 0 (avoids KeyErrors).
    return total_input_all.reindex(mn_ids).fillna(0.0)


def compute_groupwise_input_fraction(connections: pd.DataFrame,
                                     dn_group_map: pd.Series,
                                     mn_group_map: pd.Series,
                                     mn_total_input: pd.Series) -> pd.DataFrame:
    """
    Return a long-form DataFrame with one row per (dn_group, muscle) that has
    at least one direct synapse, containing:
        dn_group, muscle, numerator_synapses, denominator_input, input_fraction

    NUMERATOR   : groupwise synapses DN group -> MN group.
    DENOMINATOR : groupwise total input of the MN group (sum of member totals).
    """
    dn_ids = set(dn_group_map.index)
    mn_ids = set(mn_group_map.index)

    # Keep only DN -> wing-MN edges for the numerator.
    mask = connections[CONN_SOURCE].isin(dn_ids) & connections[CONN_TARGET].isin(mn_ids)
    dn_mn = connections.loc[mask, [CONN_SOURCE, CONN_TARGET, CONN_WEIGHT]].copy()

    # Map endpoints to their groups.
    dn_mn["dn_group"] = dn_mn[CONN_SOURCE].map(dn_group_map)
    dn_mn["muscle"]   = dn_mn[CONN_TARGET].map(mn_group_map)

    # NUMERATOR: sum synapses per (dn_group, muscle).
    numerator = (dn_mn.groupby(["dn_group", "muscle"])[CONN_WEIGHT]
                 .sum().rename("numerator_synapses").reset_index())

    # DENOMINATOR: total input per muscle = sum of member-MN total inputs.
    muscle_of_mn = mn_group_map
    denom_per_muscle = (mn_total_input.groupby(muscle_of_mn).sum()
                        .rename("denominator_input"))

    out = numerator.merge(denom_per_muscle, left_on="muscle", right_index=True, how="left")
    out["input_fraction"] = out["numerator_synapses"] / out["denominator_input"]
    out = out.sort_values("input_fraction", ascending=False).reset_index(drop=True)
    return out, denom_per_muscle


def summarise_per_muscle(long_df: pd.DataFrame,
                         denom_per_muscle: pd.Series) -> pd.DataFrame:
    """
    Per-muscle diagnostic table: total input (denominator), summed direct-DN
    input fraction across ALL DN groups, and number of DN groups above THRESHOLD.
    The summed DN fraction is the subset-trap check and the Cheong comparison.
    """
    total_dn_frac = (long_df.groupby("muscle")["input_fraction"].sum()
                     .rename("total_dn_input_fraction"))
    n_above = (long_df[long_df["input_fraction"] >= THRESHOLD]
               .groupby("muscle")["dn_group"].nunique()
               .rename(f"n_dn_groups_ge_{THRESHOLD:g}"))

    summary = pd.DataFrame({"denominator_input": denom_per_muscle})
    summary = summary.join(total_dn_frac).join(n_above)
    summary["total_dn_input_fraction"] = summary["total_dn_input_fraction"].fillna(0.0)
    summary[f"n_dn_groups_ge_{THRESHOLD:g}"] = \
        summary[f"n_dn_groups_ge_{THRESHOLD:g}"].fillna(0).astype(int)
    return summary.sort_values("total_dn_input_fraction", ascending=False)


# =============================================================================
# FIGURE  — must display the formula logic, per the project spec.
# =============================================================================
def make_figure(long_df: pd.DataFrame,
                per_muscle: pd.DataFrame,
                out_path: Path,
                demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Build DN-group x muscle matrix, keeping only DN groups that reach THRESHOLD
    # somewhere (otherwise the heatmap is mostly empty rows).
    mat = long_df.pivot_table(index="dn_group", columns="muscle",
                              values="input_fraction", aggfunc="sum", fill_value=0.0)
    keep_rows = mat.max(axis=1) >= THRESHOLD
    mat_shown = mat.loc[keep_rows]
    # Order rows by their strongest connection for readability.
    mat_shown = mat_shown.loc[mat_shown.max(axis=1).sort_values(ascending=False).index]

    n_rows = max(mat_shown.shape[0], 1)
    fig = plt.figure(figsize=(13, max(6.5, 0.28 * n_rows + 4.5)), dpi=150)
    gs = GridSpec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1.7, 6.0],
                  hspace=0.42, wspace=0.30)

    # --- Formula / methods panel (top-left) : shows the logic explicitly -----
    ax_formula = fig.add_subplot(gs[0, :])
    ax_formula.axis("off")
    formula = (
        r"$F_{\mathrm{in}}(G_{DN}\!\rightarrow\!G_{MN}) \;=\; "
        r"\dfrac{\sum_{d\in G_{DN}}\sum_{m\in G_{MN}} w(d\rightarrow m)}"
        r"{\sum_{m\in G_{MN}} \mathrm{TotalInput}(m)}$"
    )
    ax_formula.text(0.01, 0.92, "Groupwise direct DN$\\rightarrow$MN input fraction",
                    fontsize=14, fontweight="bold", va="top")
    ax_formula.text(0.02, 0.24, formula, fontsize=15, va="center")
    ax_formula.text(0.50, 0.24,
                    "numerator: all synapses DN type $\\rightarrow$ muscle\n"
                    "denominator: muscle's TOTAL input (whole connectome)",
                    fontsize=9, va="center", color="#444444")
    if demo:
        ax_formula.text(0.99, 0.98, "DEMO / SELF-TEST DATA — not real results",
                        fontsize=10, color="#B00020", fontweight="bold",
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
        ax.set_xlabel("Wing motor-neuron group (muscle)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"DN group  (max $F_{{\\mathrm{{in}}}}$ across muscles "
                      f"$\\geq$ {THRESHOLD:g})",
                      fontsize=10, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("input fraction $F_{\\mathrm{in}}$", fontsize=9)

    # --- Sanity-check bar (bottom-right) : total DN fraction per muscle ------
    axb = fig.add_subplot(gs[1, 1])
    pm = per_muscle.sort_values("total_dn_input_fraction")
    axb.barh(range(len(pm)), pm["total_dn_input_fraction"].values,
             color="#3B6EA5", edgecolor="black", linewidth=0.4)
    axb.set_yticks(range(len(pm)))
    axb.set_yticklabels(pm.index, fontsize=6)
    axb.set_xlabel("total DN input\nfraction (all DNs)", fontsize=8, fontweight="bold")
    axb.set_title("sanity check", fontsize=9)

    fig.suptitle("Direct descending $\\rightarrow$ wing motor connectivity  ·  "
                 "groupwise input fraction", fontsize=13, fontweight="bold")
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

    mn_total_input = compute_mn_total_input(connections, mn_group_map.index.values)

    long_df, denom_per_muscle = compute_groupwise_input_fraction(
        connections, dn_group_map, mn_group_map, mn_total_input)

    per_muscle = summarise_per_muscle(long_df, denom_per_muscle)

    # Wide matrix for saving.
    matrix = long_df.pivot_table(index="dn_group", columns="muscle",
                                 values="input_fraction", aggfunc="sum", fill_value=0.0)

    thresholded = long_df[long_df["input_fraction"] >= THRESHOLD].copy()

    # --- Diagnostics / subset-trap check ------------------------------------
    max_total = per_muscle["total_dn_input_fraction"].max() if len(per_muscle) else 0.0
    mean_total = per_muscle["total_dn_input_fraction"].mean() if len(per_muscle) else 0.0
    print("\n--- DIAGNOSTIC: summed direct-DN input fraction per muscle ---")
    print(f"    mean across muscles : {mean_total:.3f}")
    print(f"    max  across muscles : {max_total:.3f}")
    print(f"    Cheong (MANC) ref   : ~{CHEONG_TOTAL_DN_FRACTION_REF:.2f} "
          f"(all DNs -> wing MN)")
    if max_total > SUBSET_TRAP_WARN_LEVEL:
        print(f"    [WARNING] max summed DN fraction {max_total:.2f} exceeds "
              f"{SUBSET_TRAP_WARN_LEVEL:.2f}.\n"
              f"              connections_princeton.csv may be a SUBSET (e.g. only\n"
              f"              DN->MN edges), which would make the denominator too\n"
              f"              small and every input fraction too large. Verify it is\n"
              f"              the FULL BANC edge list before trusting these numbers.")
    else:
        print("    OK: fractions are in a plausible range for a full connectome.")

    # --- Save ---------------------------------------------------------------
    matrix.to_csv(out_dir / "direct_dn_mn_input_fraction_matrix.csv")
    long_df.to_csv(out_dir / "direct_dn_mn_input_fraction_long.csv", index=False)
    thresholded.to_csv(out_dir / "direct_dn_mn_thresholded.csv", index=False)
    per_muscle.to_csv(out_dir / "mn_group_input_totals.csv")

    fig_path = out_dir / "figure_direct_dn_mn_input_fraction.png"
    make_figure(long_df, per_muscle, fig_path, demo=demo)

    with open(out_dir / "run_summary.txt", "w") as fh:
        fh.write("Direct DN->MN groupwise input-fraction — run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"threshold (display)        : {THRESHOLD}\n")
        fh.write(f"keep untyped DNs           : {KEEP_UNTYPED_DNS}\n")
        fh.write(f"DN neurons / groups        : {len(dn_group_map)} / {n_dn_groups}\n")
        fh.write(f"  (untyped singletons)     : {n_untyped}\n")
        fh.write(f"wing MNs / muscles         : {len(mn_group_map)} / "
                 f"{mn_group_map.nunique()}\n")
        fh.write(f"nonzero DN->muscle edges   : {len(long_df)}\n")
        fh.write(f"edges >= threshold         : {len(thresholded)}\n")
        fh.write(f"mean summed DN frac/muscle : {mean_total:.4f}\n")
        fh.write(f"max  summed DN frac/muscle : {max_total:.4f}\n")

    print(f"\nSaved outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return long_df, per_muscle


# =============================================================================
# SELF-TEST  — synthetic connectome with hand-computed answers.
# =============================================================================
def selftest():
    """
    Tiny connectome with known answers (see hand calculation in comments).

    DN group DNx = {101,102}, DN group DNy = {103}
    muscleA = {201,202}, muscleB = {203}, plus an interneuron 999 as 'other' input.

    Total input per MN:
      201 : 101(10)+102(5)+999(30)          = 45
      202 : 101(5)+999(15)                  = 20
      203 : 103(20)+999(30)                 = 50
    Denominators:  muscleA=45+20=65 , muscleB=50
    Numerators:    DNx->A = 10+5+5 = 20 , DNy->B = 20 ; others = 0
    Expected F_in: DNx->A = 20/65 = 0.307692 , DNy->B = 20/50 = 0.40
    Total DN frac: muscleA = 20/65 = 0.307692 , muscleB = 20/50 = 0.40
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
    long_df, denom = compute_groupwise_input_fraction(connections, dn_map, mn_map, mn_tot)

    got = {(r.dn_group, r.muscle): round(r.input_fraction, 6)
           for r in long_df.itertuples()}
    exp = {("DNx", "muscleA"): 0.307692, ("DNy", "muscleB"): 0.40}

    print("Self-test results:")
    print(long_df.to_string(index=False))
    assert got == exp, f"\nMISMATCH\n got={got}\n exp={exp}"
    assert round(denom["muscleA"], 6) == 65.0
    assert round(denom["muscleB"], 6) == 50.0
    print("\n[OK] numerators, denominators and input fractions match hand calc.")

    # Also exercise the figure + full run end-to-end on the synthetic data.
    demo_dir = Path("/tmp/direct_dn_mn_selftest")
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
