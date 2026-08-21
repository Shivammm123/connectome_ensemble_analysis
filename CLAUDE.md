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

---

## 6. Open decisions — flag these, don't assume

- **Noise floor.** We have not settled a raw-synapse floor for reconstruction
  noise. (An earlier methods note wrongly cited a "3–4 synapse" field standard;
  Cheong actually pairs the 1% input fraction with a **≥50 raw-synapse groupwise**
  floor for calling a connection "strong.") Decide whether to add a raw floor
  alongside the input-fraction threshold, and at what value.
- **Untyped DNs.** Currently kept as singletons (`KEEP_UNTYPED_DNS = True`).
  Revisit whether to exclude them; report the count either way.
- **Next metrics (planned).** Output fraction, then the geometric mean of input
  and output fractions (favours edges strong from both sides — Cheong used this
  for pathway exploration). These come after direct input-fraction is solid.
- **Cross-dataset replication (later goal).** Intent is to repeat across BANC /
  MANC / maleCNS. When we do: normalise on the **MN side** (VNC-local, comparable);
  DN-side normalisation is NOT comparable across full-CNS vs VNC-only datasets
  without restricting BANC's denominator to VNC inputs. Cell-type matching across
  datasets is a real sub-project, not a given.

---

## 7. Working style for this project

- Verify before trusting. When a script can be run against real data, run it and
  check the diagnostics rather than assuming the numbers are right.
- Be explicit about what is verified vs assumed, especially anything depending on
  the connections file being complete.
- Don't reach for multi-hop, signed, or effective-connectivity machinery yet —
  that's later. Keep the current step simple and correct.
- When comparing results to Cheong et al., cite the current version of that paper
  (it has multiple preprint/eLife versions with a changed title); numbers in the
  version-1 PDF may have been revised.
