# Wing Motor-Control Connectome Analysis

Mapping how descending neurons (DNs) drive wing motor neurons (MNs) in
*Drosophila*, directly and via premotor interneurons, using the **BANC**
connectome (Brain And Nerve Cord — full CNS, adult female, Princeton/FlyWire).

This is a from-scratch **redesign** of an earlier pilot analysis. The pilot
is superseded and kept only for reference under [`misc/pilot/`](misc/pilot/)
— see [Project history](#project-history-pilot-vs-redesign) below. Everything
described in this README is the current, active pipeline.

---

## Table of contents

1. [Why this dataset, and what's new](#why-this-dataset-and-what-is-new)
2. [Core methodology](#core-methodology)
3. [Data](#data)
4. [Project structure](#project-structure)
5. [Pipeline steps](#pipeline-steps)
6. [Running the analysis](#running-the-analysis)
7. [Headline results so far](#headline-results-so-far)
8. [Project history: pilot vs. redesign](#project-history-pilot-vs-redesign)
9. [References](#references)

---

## Why this dataset, and what is new

The closest prior work is **Cheong, Eichler, Stürner et al. 2024 (eLife,
MANC)**, which mapped premotor circuits in the male *Drosophila* ventral
nerve cord. BANC differs in two important ways: it's **female**, and it's
**full-CNS** — DNs are captured with their brain-side arbor intact, not
severed at the neck the way MANC's are. That brain-side access is the
intended source of novelty here; the descriptive wing-premotor map itself is
largely a replication of Cheong's, run on a different, complementary dataset.

## Core methodology

- **Connection strength is always a *fraction*, never a raw synapse count.**
  Raw synapse counts conflate "strong connection" with "big neuron" — they
  only ever appear as an intermediate numerator, immediately normalised.
- **Groupwise, not per-neuron.** DNs are grouped by shared `Primary Cell
  Type` (bilateral pairs / systematic types share a name); untyped DNs are
  kept as singleton groups so nothing is silently dropped. Motor neurons are
  grouped into 18 muscles / motor pools.
- **Unsigned, direct, single-hop — for now.** Neurotransmitter sign
  predictions aren't trusted yet, and multi-hop / effective-connectivity
  methods are deprioritised in favour of getting the direct DN→MN structure
  right first.
- Four metrics are computed, each answering a different question about a
  direct DN-group → muscle edge:

  | Metric | Formula | Question it answers |
  |---|---|---|
  | **Input fraction** (`F_in`) | synapses(DN→MN) / muscle's *total* input (whole connectome) | "What share of this muscle's input comes from this DN type?" — influence, the target's-eye view. |
  | **Output fraction** (`F_out`) | synapses(DN→MN) / DN's *total* output (whole connectome) | "What share of this DN's entire output goes to this muscle?" — dedication, source's-eye view. Not comparable across datasets (BANC's denominator includes brain-side output MANC doesn't have). |
  | **VNC-restricted output fraction** (`F_out,VNC`) | synapses(DN→MN) / DN's total output, **VNC synapses only** | Same question as above, but restricted to the DN's local VNC arbor — puts it on the same VNC-local footing as `F_in`, and is the version that *would* be comparable across BANC/MANC/maleCNS. |
  | **Geometric mean** (`F_geom`) | `sqrt(F_in × F_out,VNC)` | Cheong et al.'s pathway-exploration metric — high only when an edge is strong from *both* sides at once, surfacing edges either fraction alone can miss or overrate. |

  A DN's VNC-vs-brain split matters here: wing-MN input in this dataset is
  confirmed **100% VNC-tagged** (motor-neuron dendrites are physically in the
  VNC), which is exactly why `F_in` and `F_out,VNC` — not `F_out` — are the
  pair combined into `F_geom`.

- **The "subset trap"**: every one of these denominators is only correct if
  the connections file is the *full* BANC edge list. Each script prints a
  diagnostic checking this (e.g. step 01 compares its summed DN-input-share
  per muscle against Cheong's published ~9–10% reference for MANC); see each
  script's docstring for details.

## Data

| File | Description |
|---|---|
| `data/raw/connections_princeton.csv` (~193MB, gitignored) | Raw Princeton/FlyWire-codex connectivity export. Columns: `pre_root_id`, `post_root_id`, `neuropil`, `syn_count`, `nt_type`. One row per (pre, post, neuropil) — a given edge can span several rows. |
| `data/raw/neurons.csv` (gitignored) | Neuron metadata. Columns include `Root ID`, `Super Class`, `Primary Cell Type`. DNs are rows where `Super Class == "descending"` (1,313 of them). |
| `data/processed/motor_pools/motor_pools.csv` | Curated wing motor-pool mapping (50 MNs → 18 muscles), reused from the pilot as the one piece of prior work still actively used. |

The two raw files aren't checked into git (they're large and not license-clear
to redistribute) — get them from the BANC/FlyWire codex release and place
them at the paths above.

**Column-name note:** BANC's own columns use spaces and capitals (`Root ID`,
`Super Class`, `Primary Cell Type`) — every script keeps them as-is rather
than converting to snake_case. The connections file is the one exception:
it's already snake_case because it's a different (raw codex) export format.

## Project structure

```
connectome_ensemble_analysis/
├── README.md                     # this file
├── CLAUDE.md                     # detailed working notes / agent instructions
├── requirement.txt                # Python dependencies
├── data/
│   ├── raw/                      # raw BANC CSVs (gitignored, see Data above)
│   └── processed/motor_pools/    # curated motor-pool mapping (active input)
├── src/                          # the current pipeline — all scripts here
├── results/                      # current pipeline's outputs, one folder per step
├── docs/                         # slide deck, misc reference docs
└── misc/pilot/                   # archived, superseded pilot analysis — see
                                   # misc/README.md. Reference only.
```

## Pipeline steps

Each script in `src/` is self-contained and independently runnable — it
recomputes everything it needs from the raw data rather than depending on
another step's output, so there's no fixed run order to remember.

| Step | Script | What it produces |
|---|---|---|
| 01 | [`direct_dn_mn_input_fraction.py`](src/direct_dn_mn_input_fraction.py) | DN-group × muscle input-fraction matrix, thresholded edges, per-muscle diagnostics, heatmap figure. |
| 02 | [`direct_dn_mn_output_fraction.py`](src/direct_dn_mn_output_fraction.py) | Same shape, output fraction (whole-connectome denominator). |
| 02b | [`direct_dn_mn_vnc_output_fraction.py`](src/direct_dn_mn_vnc_output_fraction.py) | Same, VNC-restricted denominator. |
| 03 | [`direct_dn_mn_geometric_mean.py`](src/direct_dn_mn_geometric_mean.py) | Ranked pathway list by `F_geom`, metric-agreement diagnostic. |
| 04 | [`top_dns_per_muscle.py`](src/top_dns_per_muscle.py) | Top-5 DN leaderboard per muscle (reporting view on step 01, not a new metric). |

Every script has a `--selftest` mode that verifies its math against a tiny
hand-computed synthetic connectome — run that first if you're not sure the
environment is set up correctly.

## Running the analysis

```bash
pip install -r requirement.txt   # numpy, pandas, scipy, matplotlib, etc.

# from the project root:
python src/direct_dn_mn_input_fraction.py --selftest   # verify the math
python src/direct_dn_mn_input_fraction.py               # run on real data
```

Repeat for each script in `src/`. Results land in `results/<step_name>/` —
a data matrix (CSV), a long-form non-zero-edges table, a thresholded table,
a diagnostics table, a figure (fixed at 16:9 so it drops onto a slide as-is),
and a `run_summary.txt`.

## Headline results so far

(Full detail and exact figures in [CLAUDE.md](CLAUDE.md) §5; summarised here.)

- Direct DNs together supply a **mean of 8.1%** of a wing muscle's input
  (max 28.8%, muscle `hg2`) — in line with Cheong's ~9–10% reference for
  MANC, confirming the connections file is the full edge list, not a subset.
- `DNp31` is a broad hub: the #1 or #2 direct input source for 5 of the 6
  flight-power muscles (DLM/DVM). Steering muscles (basalar, haltere,
  axillary, pleurosternal) each have much more muscle-specific top
  contributors with little overlap between them.
- The geometric-mean metric surfaces pathways neither component fraction
  alone would rank highly — e.g. `DNa08→DLM1` is unremarkable by input
  fraction alone (0.6%) but dedicates 28.4% of its VNC output there,
  giving a geometric mean (4.1%) that pulls it into view.

## Project history: pilot vs. redesign

An earlier pilot (interneuron hub clustering, circuit-modularity scoring,
DN→IN→MN pathway tracing) explored this same question with a different
methodology and is now superseded — its scripts, config, notebooks, and
output tables are archived under [`misc/pilot/`](misc/pilot/) (see
[`misc/README.md`](misc/README.md) for what's there and why). Nothing in the
current pipeline depends on it, except the one curated motor-pool mapping
described under [Data](#data) above.

## References

**Data source** — FlyWire/Princeton BANC connectome (Brain And Nerve Cord,
full adult female CNS).

**Closest prior work:**
```
Cheong, Eichler, Stürner et al. (2024). Transforming descending input into
behavior: The organization of premotor circuits in the Drosophila Male
Adult Nerve Cord connectome. eLife. https://doi.org/10.7554/eLife.96084
```
(Cite the current version — this paper has had multiple preprint/eLife
revisions with a changed title; numbers in an early PDF may since have been
revised.)
