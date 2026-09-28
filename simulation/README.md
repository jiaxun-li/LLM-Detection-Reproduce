# Finite-state simulation

Run from the repository root with Python 3.11 or newer. No GPU or language-model
weights are needed. NumPy 2.3.5 is specified by the simulation package.

```sh
python -m pip install -r simulation/original/requirements.txt
python reproduce.py simulation verify
python reproduce.py simulation summarize --output outputs/simulation-summary
python reproduce.py simulation figure --output outputs/simulation-figure
```

Summary outputs are `results/report.md` and `results/plots.svg` under the chosen
output directory. Figure generation writes `figure_build/section4_alt_power.tex`.
Add `--compile` using a new output directory for PDF compilation, which requires
`pdflatex`, `standalone`, TikZ, and PGFPlots.

A small execution check:

```sh
python reproduce.py simulation run --output outputs/simulation-smoke --paths 20 --calibration 40 --lengths 16 32 --orders 10 --supports 500 --seed 20260906
```

The full simulation uses substantially more CPU time:

```sh
python reproduce.py simulation run --output outputs/simulation-full --paths 2000 --calibration 4000 --lengths 128 512 2048 4096 --orders 10 100 --supports 500 1000 --seed 20260906
```

All generated outputs must use fresh directories. The wrapper stages the
simulation implementation and runs against that copy, preserving included
reference results. A small run verifies execution, not the paper's estimates.
See [methods](../docs/TECHNICAL_REFERENCE.md) for scientific scope and uncertainty.

See the [technical details](original/TECHNICAL_DETAILS.md) and
[saved settings](original/results/config.json). The four execution-source hashes
and distribution inventory are checked by the wrapper. Use the wrapper to keep
reference algorithms and numerical results unchanged.
