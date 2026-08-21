# misc/ — archived pilot work (reference only)

Everything under here is from the **pilot analysis** that the current
redesign (see the project root [`CLAUDE.md`](../CLAUDE.md)) supersedes. It's
kept for provenance and reference, not as something to build on. Nothing in
`src/` or `results/` depends on anything in this folder.

Archived here on 2026-08-21 as part of a directory cleanup — this used to be
loose clutter at the project root (`scripts/`, `notebooks/`, `config.yaml`,
`results_old/`, `Results_3 min/`).

## What's here

- **`pilot/scripts/`** — the pilot's analysis scripts (cell-type
  classification, circuit modules, DN pathway mapping, Shrek-similarity
  variants, plotting scripts, etc.), plus `pilot/scripts/utils/` (its
  `data_loader.py` / `similarity.py` helpers).
- **`pilot/config.yaml`** — the pilot's config, read by most scripts in
  `pilot/scripts/` (`config_path: str = 'config.yaml'`, i.e. expected in the
  working directory). If you ever need to actually rerun one of these
  scripts, run it from inside `misc/pilot/` so that relative path resolves.
- **`pilot/notebooks/`** — the pilot's one interactive notebook.
- **`pilot/results_old/`** — the pilot's output tree, and the more complete
  and more recent of the two (182 files, last written 2026-03-17). This was
  the canonical pilot-results snapshot.
- **`pilot/Results_3min/`** — an older, partially-overlapping output snapshot
  (161 files, 2026-03-04) from an earlier pilot run. It predates
  `results_old/` and is superseded by it almost file-for-file; kept only
  because it holds one file `results_old/` doesn't —
  `candidate_prioritization/experimental_suggestions.csv`.

## What got pulled out before archiving

`pilot/results_old/motor_neurons/` (the curated `motor_pools.csv` +
`wing_mns_with_muscles.csv` + figure) was **not** archived — it's an active
input to the redesigned pipeline, not pilot output to shelve. It now lives at
[`data/processed/motor_pools/`](../data/processed/motor_pools/), which is
where [`src/direct_dn_mn_input_fraction.py`](../src/direct_dn_mn_input_fraction.py)
reads it from.
