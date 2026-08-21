# CLAUDE.md — Wing Motor-Control Connectome (redesigned pipeline)

Project instructions for Claude Code. Read this before working on anything in
this repo. It records decisions already made so you don't re-litigate them, and
flags what is still open so you don't silently assume an answer.

---

## 1. What this project is

A connectome analysis of **wing motor control in *Drosophila*** using the
**BANC** dataset (Brain And Nerve Cord — full CNS, adult female). Goal: map how
descending neurons (DNs) drive wing motor neurons (MNs), directly and via
premotor interneurons, and characterise the functional organisation.

There was an earlier **pilot analysis** (separate scripts, separate results).
The pilot is **superseded** — treat it as reference only. Do not build on its
methods. The redesign lives under `src/` and `results/`; keep it isolated from
pilot artifacts.

Closest prior work is **Cheong, Eichler, Stürner et al. 2024 (eLife, MANC)** —
male VNC. We are on BANC, which is female and full-CNS, so the brain side of DNs
is available to us in a way MANC's severed DNs are not. That brain-side access
is the intended source of novelty; the descriptive wing-premotor map is largely
replication of Cheong.

---

## 2. Core methodological decisions (do not silently change these)

- **Connection strength = input fraction, NOT raw synapse counts.** Raw counts
  are only ever a numerator that is immediately normalised. No result should
  report raw synapse counts as a strength.
- **Input fraction, target-normalised:**
  `weight(A→B) = synapses(A→B) / (B's TOTAL input synapses)`.
  The denominator is B's total input across the **whole connectome**, not just
  the edges inside our working subset. This is the target's-eye view of
  influence — the correct question for "who controls this motor neuron."
- **Why not output fraction (share of the source's output budget):** it measures
  a neuron's *dedication*, not its *influence*, and it is **not comparable across
  datasets** in BANC because a DN's output budget includes brain-side synapses
  that VNC-only datasets (MANC) lack. Input fraction stays comparable because MN
  input is VNC-local everywhere. (Output fraction + geometric mean are planned
  for a later step — see §6.)
- **Groupwise, not per-neuron.** Group before normalising.
  - DN groups = shared `Cell Type` (bilateral pairs / systematic types share a
    name). DNs with no type are kept as singleton groups (`untyped_<RootID>`) so
    nothing is dropped; report how many.
  - MN groups = muscles / motor pools, from the curated `motor_pools.csv`.
- **Not in scope yet:** neurotransmitter signs (E/I) — the connectome's NT
  predictions are not something we trust yet, so keep everything unsigned for
  now. Multi-hop / effective-connectivity ("hop") methods are **deprioritised** —
  focus on direct and shallow structure first.

### The one thing that can silently break correctness — the "subset trap"

The input-fraction denominator is only right if `connections_princeton.csv` is
the **full BANC edge list**. If it were pre-filtered (e.g. only DN→MN edges),
every denominator would be missing the MN's other inputs and every fraction
would be inflated toward 1.0. **Always verify this.** The step-1 script prints a
diagnostic: the summed direct-DN input fraction per muscle. Cheong found all DNs
together supply only **~9–10%** of wing-MN input. If our numbers land near there,
the file is full and we've reproduced a known result. If they approach 1.0, the
file is a subset and the denominator is wrong.

---

## 3. Data

Files (verified against the actual files on disk, 2026-08-21 — the paths and
column names below are ground truth, not aspirational):

- `data/raw/connections_princeton.csv` (~193MB) — raw Princeton/FlyWire-codex
  export. Columns: `pre_root_id`, `post_root_id`, `neuropil`, `syn_count`,
  `nt_type`. **Not** `source`/`target`/`weight` as earlier notes assumed — map
  `source=pre_root_id`, `target=post_root_id`, `weight=syn_count`. One row per
  (pre, post, neuropil), so a pre→post pair can span several rows; always
  `.sum()` weight when aggregating, never assume one row per edge. Confirmed
  to be the FULL connectome edge list (see subset trap above) — the real-data
  run's diagnostic landed at mean 8.1% / max 28.8% summed DN input fraction,
  in line with Cheong's ~9–10%, not near 1.0.
- `data/raw/neurons.csv` — columns include `Root ID`, `Super Class`,
  `Primary Cell Type` (also `Alternative Cell Type(s)`, unused so far).
  **Not** a bare `Cell Type` column as earlier notes assumed — use
  `Primary Cell Type`. 1313 DNs (`Super Class == "descending"`), 1310 of them
  have a `Primary Cell Type` (3 untyped).
- `data/processed/motor_pools/motor_pools.csv` — curated by the pilot,
  reused as input by the redesign. Columns: `muscle`, `muscle_type`,
  `muscle_group`, `n_motor_neurons`, `motor_neuron_ids` (a stringified Python
  list). 50 MNs / 18 muscles. Moved here from `results_old/motor_neurons/`
  during the 2026-08-21 directory cleanup (see §4) — it's an active input,
  not archived pilot output, so it doesn't live under `misc/`.

**Column-name gotcha (recurring pilot bug):** BANC columns have spaces and
capitals — `Root ID`, `Super Class`, `Primary Cell Type`. Do NOT rewrite them
to snake_case. DNs are rows where `Super Class == "descending"`. The
connections file is the one exception — it's already snake_case
(`pre_root_id`/`post_root_id`/`syn_count`) because it's a different (raw
codex) export, not the neurons file's format.

---

## 4. Directory conventions

```
<project root>/
├── CLAUDE.md                     # this file
├── data/
│   ├── raw/                       # raw BANC CSVs (connections_princeton.csv, neurons.csv)
│   │                               # gitignored — not committed, ~193MB + 15MB
│   └── processed/motor_pools/     # curated motor_pools.csv etc., reused as step-01 input
├── src/                           # redesigned pipeline scripts (all new work here)
├── results/                       # redesigned outputs; one subfolder per step
│   └── direct_dn_mn_input_fraction/   # step-01 output
├── docs/                          # loose reference docs (CLUSTER.docx, slide deck)
└── misc/pilot/                    # archived pilot: scripts/, notebooks/, config.yaml,
                                    # results_old/, Results_3min/ — reference only, see
                                    # misc/README.md. Do not extend, do not build on.
```

- New scripts go in `src/`. New results go in `results/<step_name>/`.
- Scripts assume project root is the parent of `src/` and use paths relative to
  that. Keep paths as clearly-labelled constants at the top of each script.
- Prefer simple, single-purpose, heavily-documented scripts over a config-driven
  master script (a config-driven approach was tried in the pilot and caused
  confusion). Each script should be runnable on its own and self-documenting.
- **Directory cleanup done 2026-08-21**: the pilot's `scripts/`, `notebooks/`,
  `config.yaml`, `results_old/`, and the duplicate `Results_3 min/` snapshot
  were archived into `misc/pilot/` (see `misc/README.md`). The one actively-used
  piece of pilot output, `motor_pools.csv`, was pulled out first and now lives
  at `data/processed/motor_pools/`. `data/raw/*.csv` were untracked from git
  (still on disk, just no longer committed — `.gitignore` was fixed, it was
  UTF-16-encoded and silently doing nothing before). Existing git history still
  has the old large blobs baked in (~222MB `.git`) — that was left alone
  deliberately (no history rewrite / force-push) rather than fixed.

---

## 5. What exists so far

- **`src/direct_dn_mn_input_fraction.py`** — Step 01. Groupwise direct DN→MN
  connectivity by input fraction. Produces the DN-group × muscle matrix, a
  long-form table, a thresholded table, a per-muscle diagnostic table, and a
  figure whose top panel shows the formula explicitly.
  - Display threshold: `THRESHOLD = 0.01` (1% input fraction, matching Cheong's
    "substantial" cutoff). This is display/reporting only — the full continuous
    matrix is always saved unthresholded.
  - Has a `--selftest` mode: runs on a tiny synthetic connectome with
    hand-computed answers. The math is **verified on synthetic data**.
  - **Run on real BANC data on 2026-08-21** (results in
    `results/direct_dn_mn_input_fraction/`). Required fixing three stale path/
    column assumptions in the script's constants — see §3 above for the
    corrected paths/columns (now also reflected in the script itself).
    Diagnostic: **mean summed DN input fraction = 8.1%, max = 28.8%** across
    18 muscles — no subset-trap warning, in line with Cheong's ~9–10% (MANC).
    544 DN groups (1313 DNs, 3 untyped singletons) × 18 muscles (50 MNs);
    429 nonzero DN→muscle edges, 37 at/above the 1% display threshold.
    Strongest direct edges: `DNg02→ps1` (5.5%), `DNp63→hg2` (4.3%),
    `DNg82→hg2` (3.6%); `DNp31` stands out as a broad hub reaching many
    muscles (DLM1/DVM1A/DVM2A/DVM3A/ps1) rather than one dominant target.
    Muscles with the most direct DN drive: hg2, ps1, hg1, b3. This is a first
    pass at face value — not yet cross-checked neuron-by-neuron against Cheong.

- **`src/direct_dn_mn_output_fraction.py`** — Step 02. Groupwise direct DN→MN
  connectivity by **output fraction** (mirror of step 01: same DN→MN edges,
  denominator flipped to the DN group's total output instead of the muscle's
  total input — the "dedication" view rather than the "influence" view; see
  the script's docstring for why it's NOT valid for cross-dataset comparison,
  unlike input fraction). Same outputs as step 01 (matrix/long/thresholded/
  per-group diagnostic table/figure), written to
  `results/direct_dn_mn_output_fraction/`.
  - Same `THRESHOLD = 0.01` display cutoff as step 01, for consistency.
  - `--selftest` reuses step 01's synthetic connectome so the two scripts are
    cross-checkable; hand-computed answer is F_out = 1.0 for both toy edges
    (the synthetic DNs have no targets besides the test MNs, so 100% of their
    output goes there — deliberately exercises the denominator grouping).
  - Subset-trap check here is a **heuristic, not a published reference**
    (Cheong's ~9–10% number was specifically about MN input share, doesn't
    apply to output share) — flags if >10% of DN groups dedicate >50% of
    their total output to the 18 wing muscles alone.
  - **Run on real BANC data on 2026-08-21** (results in
    `results/direct_dn_mn_output_fraction/`). No subset-trap warning: mean
    summed output fraction to wing muscles = 1.3%, median 0%, max 46.5%
    (`DNa08`), 0% of DN groups over the 50% heuristic flag. 92 DN groups
    reach the 1% threshold somewhere (vs. step 01's 23) — expected, since
    crossing 1% here only requires a small DN output budget, not a big shared
    muscle total. Most wing-dedicated DN groups: `DNa08` (46.5%, mostly→DLM1),
    `DNge015` (43.6%), `DNp31` (30.0% — the same broad hub from step 01, but
    now a *smaller* share of its own huge output budget than its outsized
    share of muscle input). `DNa08→DLM1` is the single strongest edge (26.3%
    of DNa08's entire output).

- **`src/direct_dn_mn_vnc_output_fraction.py`** — Step 02b. Variant of step
  02: same output-fraction math, but the denominator (and, for consistency,
  the numerator) is restricted to synapses physically located in a VNC
  neuropil, excluding a DN's brain-side output entirely. Answers "of this
  DN's *local VNC processing budget*, what share drives wing muscles?" rather
  than step 02's "of its *entire* output, brain included, what share?" — this
  is the version that would actually be comparable across BANC/MANC/maleCNS
  per the cross-dataset item below, since it matches how step 01's MN-input
  denominator was always VNC-local. VNC membership comes straight from the
  connections file's own `neuropil` column (`VNC_...` prefix, 114 distinct
  neuropil values, no separate neuron-level classification needed) — verified
  against the real file: VNC synapses are 36.6% of the whole connectome
  (8.14M / 22.2M), with a negligible ambiguous "neck" category (872 synapses,
  0.004%, excluded from both VNC and brain).
  - `--selftest` extends step 02's synthetic connectome with one extra
    brain-tagged edge and asserts the VNC filter drops exactly that row.
  - **Run on real BANC data on 2026-08-21** (results in
    `results/direct_dn_mn_vnc_output_fraction/`). No subset-trap warning.
    32.6% of connection rows / 36.6% of synapses are VNC-tagged; only 19/1313
    DNs (1.4%) show zero VNC output (plausible — truncated reconstructions —
    not a red flag). Mean summed VNC-output fraction to wing muscles = 1.8%,
    median 0%, max 50.6% (`DNge015`), only 0.4% of DN groups over the 50%
    heuristic flag. As expected, VNC-restricted fractions are uniformly ≥ the
    whole-connectome ones from step 02 (smaller denominator, same numerator)
    — ratio ranges from ~1.06× up to ~2.1× (`DNpe010`, `DNp03`): those two
    DNs' brain-side arbor is doing a lot to dilute their step-02 number even
    though, locally within the VNC, close to a third to a quarter of their
    output already goes to wing muscles.

- **`src/direct_dn_mn_geometric_mean.py`** — Step 03. Combines step 01
  (input fraction) and step 02b (VNC-restricted output fraction) as
  `F_geom = sqrt(F_in * F_out,VNC)`, per Cheong et al.'s pathway-exploration
  metric — self-contained (recomputes both component fractions from raw
  data rather than reading steps 01/02b's saved CSVs, same standalone
  convention as the rest of this pipeline). VNC-restricted output fraction
  was chosen over step 02's whole-connectome version because it's on the
  same VNC-local footing as step 01's denominator (verified: wing-MN input
  is 100% VNC-tagged, so this doesn't change F_in's value from step 01 at
  all — only adds the matching VNC-restricted F_out side). Unlike steps
  01/02/02b, this is fundamentally an **edge/pathway-level** metric, not a
  per-group total — its main output is a ranked pathway list, not a summed
  "sanity check" per DN group or muscle.
  - `--selftest` reuses step 02b's synthetic connectome; unlike 02/02b's toy
    tests (which both landed at a trivial 1.0), this one has genuinely
    different F_in (0.308) and F_out (1.0) for one edge, giving a
    non-trivial hand-computed F_geom = 0.5547 — a better exercise of the
    sqrt(a·b) math than a case where both inputs are already 1.
  - Also prints/saves a **metric-agreement table**: how many edges clear
    THRESHOLD under F_in alone, F_out,VNC alone, both individually, and
    F_geom — because `sqrt(a*b) >= T` does NOT strictly require both
    `a >= T` and `b >= T` (one very strong side can still pull the mean over
    the line), so this is checked empirically rather than assumed.
  - **Run on real BANC data on 2026-08-21** (by the user, per the write-not-run
    convention — results in `results/direct_dn_mn_geometric_mean/`). 429
    nonzero direct edges; 37 clear THRESHOLD by F_in alone, 207 by F_out,VNC
    alone, 35 by **both** individually, 88 by F_geom. Metric-agreement check
    confirms the math cleanly: **0** edges pass both individually but fail
    F_geom (mathematically guaranteed — sqrt(a·b) >= T whenever a,b >= T),
    while **53** edges pass F_geom without passing both sides individually —
    real confirmation that F_geom is not a strict AND, exactly as the
    docstring warned. Top pathway: `DNg02→ps1` (F_in=5.5%, F_out=14.2%,
    F_geom=8.8%). Notable case `DNa08→DLM1`: F_in only 0.6% (unremarkable by
    step 01 alone) but F_out=28.4% (DNa08 dedicates a lot of its VNC output
    here) → F_geom=4.1%, surfacing a pathway step 01 ranked much lower — the
    kind of edge this combined metric exists to catch.

- **`src/top_dns_per_muscle.py`** — Step 04. Not a new metric — a reporting
  view on top of step 01: for each wing muscle, its top `TOP_N=5` DN groups
  ranked by **input fraction** (never raw synapse count, per the project's
  core rule — raw counts are shown alongside for context only). Kept
  self-contained (recomputes step 01's input-fraction math from raw data)
  for the same standalone-script reason as every other step here, even
  though it duplicates step 01's core computation.
  - Outputs a long table (one row per muscle × rank) and a wide table (one
    row per muscle, columns rank1..rank5 + a `cumulative_top5_fraction`).
  - Figure is a small-multiples grid — one mini bar chart per muscle, all 18
    shown at once (6×3 grid). Unlike steps 01-03's heatmaps, this needed no
    `TOP_N_ROWS` truncation logic: the muscle count is small and fixed, so
    there's no unbounded-growth problem to cap.
  - `--selftest` reuses step 01's synthetic connectome, which happens to
    have only 1 contributing DN group per muscle — deliberately exercises
    the "fewer than N available" path (must show exactly what exists, no
    padding, no error).
  - **Run on real BANC data on 2026-08-21** (by the user — results in
    `results/top_dns_per_muscle/`). Only 1 of 18 muscles (`iii3`) has fewer
    than 5 contributing DN groups (it has 4). `DNp31` is the #1 or #2 DN for
    5 of the 6 flight-power muscles (DLM1/DLM5/DVM1A/DVM2A/DVM3A), confirming
    its broad-hub character from steps 01-03; steering muscles (b1-b3,
    hg1-hg4, i1-i2, iii1-iii4, ps1) each have a much more muscle-specific
    top-5 list with little overlap between them — consistent with steering
    muscles being individually controlled rather than sharing a few
    generalist DNs the way the power muscles seem to.

---

## 6. Open decisions — flag these, don't assume

- **Noise floor.** We have not settled a raw-synapse floor for reconstruction
  noise. (An earlier methods note wrongly cited a "3–4 synapse" field standard;
  Cheong actually pairs the 1% input fraction with a **≥50 raw-synapse groupwise**
  floor for calling a connection "strong.") Decide whether to add a raw floor
  alongside the input-fraction threshold, and at what value.
- **Untyped DNs.** Currently kept as singletons (`KEEP_UNTYPED_DNS = True`).
  Revisit whether to exclude them; report the count either way.
- **Next metric — done, pending your real-data run.** Step 03
  (`direct_dn_mn_geometric_mean.py`) is written, using VNC-restricted output
  fraction (step 02b) paired with input fraction (step 01), as reasoned in
  §5. Run it and check the metric-agreement table and top pathways before
  trusting the numbers, same as every other step here.
- **Cross-dataset replication (later goal).** Intent is to repeat across BANC /
  MANC / maleCNS. When we do: normalise on the **MN side** (VNC-local, comparable);
  DN-side normalisation is NOT comparable across full-CNS vs VNC-only datasets
  without restricting BANC's denominator to VNC inputs — **step 02b
  (`direct_dn_mn_vnc_output_fraction.py`) already does this restriction**, using
  the connections file's own `neuropil` column, and is the metric to use for this
  goal rather than step 02. Cell-type matching across datasets is still a real
  sub-project, not a given.

---

## 7. Working style for this project

- **As of 2026-08-21, Claude writes/edits pipeline scripts but does not run them**
  — the user runs scripts themselves and reviews results (a `python -m
  py_compile` syntax check is fine, that's not a run). This means new/edited
  scripts in this repo may be unverified against real data until the user
  runs them — don't assume a script "works" just because it was written
  carefully; check whether its docstring/CLAUDE.md notes say it's been run.
- Verify before trusting. Where a script's real-data diagnostics are already
  known (because it's been run), treat those as established; where they
  aren't yet, say so plainly rather than assuming the numbers are right.
- Be explicit about what is verified vs assumed, especially anything depending on
  the connections file being complete.
- Don't reach for multi-hop, signed, or effective-connectivity machinery yet —
  that's later. Keep the current step simple and correct.
- When comparing results to Cheong et al., cite the current version of that paper
  (it has multiple preprint/eLife versions with a changed title); numbers in the
  version-1 PDF may have been revised.
