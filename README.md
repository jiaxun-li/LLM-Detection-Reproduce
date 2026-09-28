# Supplement: robust detection of LLM-generated text under contamination

Implementation and numerical evidence for the accompanying paper. The three
studies have separate fitting and evaluation protocols.

| Study | Paper location | Distributed evidence |
|---|---|---|
| Finite-state simulation | Appendix C.3 | `simulation/original/results/` |
| Controlled contamination | Section 5.2, Appendix F | `reference/empirical/primary/` |
| RAID, four-scope TPR selection | Section 5.3, Appendix G | `reference/raid/` |

## Reproduce the paper's saved results

Use Python 3.11 or newer in a virtual environment, from this directory:

```sh
python -m pip install -r requirements/reproduce.txt
python reproduce.py verify
python reproduce.py figures --output outputs/paper-results
python reproduce.py raid-tables --output outputs/raid-tables
python reproduce.py simulation summarize --output outputs/simulation-summary
python reproduce.py simulation figure --output outputs/simulation-figure
```

Every output directory must be new. `figures` regenerates controlled-study
figures and tables and the current RAID table fragments. `raid-tables` selects
the RAID tables only. These commands use the paper's 119 RAID fits and 462
result rows; they do not select different bounds or perform model inference.
The paper-to-artifact map appears below.

`verify` checks the distributed hashes, current RAID fitting records, source
splits, and all displayed values in Tables 3 and 4. Its report separately states
provenance availability; passing integrity checks does not imply complete
score-level replay. [Provenance and scope](docs/PROVENANCE.md) lists remaining inputs.

For all 99 individual controlled-study PNGs:

```sh
python reproduce.py cell-plots --output outputs/controlled-cell-plots
```

The [simulation guide](simulation/README.md) gives small and full CPU runs.
Its figure command writes TeX; `--compile` additionally requires `pdflatex`,
TikZ, PGFPlots and the standalone class.

## Reproduce model experiments

Follow [EXPERIMENTS.md](docs/EXPERIMENTS.md). Exact recorded primary settings,
including model/tokenizer/dataset revisions and per-cell length guards, are in
`configs/paper_cells/`. Recorded environment and source provenance are in
`reference/provenance/`. Configure devices for available hardware.

The RAID reference includes the identities of all 12,871 selected sources,
their split and generation family, 154,452 machine-row identities, five-fold
tuning assignments, measured edit rates, fitted bounds, and final thresholds.
The original exclusion file, selected-source JSONL, controlled source identities,
and scoring-gate evidence are included; see [PROVENANCE.md](docs/PROVENANCE.md).
Restore exact controlled manifests with `python reproduce.py restore-sources --output NEW_SOURCE_CACHE`. Model weights and
token-score packs are not included. Do not substitute a new sample and label
its results as exact replay of the paper.

See the [technical reference](docs/TECHNICAL_REFERENCE.md) for implementation details.


## Paper-to-artifact map

All numbers refer to the accompanying submitted manuscript.

| Paper content | Frozen input | Reproduction output under `figures --output DIR` |
|---|---|---|
| Table 2; Figure 3; Appendix F | `reference/empirical/primary/` | `DIR/controlled/figures/` and `DIR/controlled/generated/` |
| Table 3 | `reference/raid/results.csv` | `DIR/raid/main_rates.tex` |
| Table 4 | Same | `DIR/raid/main_attacks.tex` |
| Tables 15–18 | Same | `DIR/raid/rates.tex` |
| Tables 19–25 | Same | `DIR/raid/attacks.tex` |
| Tables 26–32 | Same | `DIR/raid/costs.tex` |
| Tables 33–46 | Same | `DIR/raid/universal.tex` |
| Tables 47–53 | `reference/raid/selected_specs.json` | `DIR/raid/bounds.tex` |
| Table 14 | `reference/raid/binoculars_sanity.csv` | `DIR/raid/table_14_raw_baseline.csv` |
| Table 54 and truncation diagnostics | `reference/raid/contamination_records.csv` | `DIR/raid/table_54_rates_and_truncation.csv` |
| Tables 55–56 | `reference/raid/selected_sources.csv` | `DIR/raid/table_55_generators.csv`, `table_56_domains.csv` |
| Table 5; Figure 4 | `simulation/original/results/` | `simulation summarize` and `simulation figure` outputs |

The RAID builder uses the table-generation logic supplied with the research
manuscript. Captions/labels may differ from final editorial wording; numerical
precision and fitting scopes match. `verify` checks every main RAID table value
against the submitted values, plus the complete result and fitting inventories.

Saved plot copies are omitted; generate them with the commands above.
Code uses the MIT license; external resources retain their own terms as
described in [REUSE.md](REUSE.md).
