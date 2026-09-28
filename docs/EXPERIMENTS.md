# Fresh model experiments

These commands run on computing resources supplied by the reader. Choose a
compatible PyTorch/CUDA installation and enough GPU memory for the target model
and Falcon scoring pair. Adjust batch sizes and device maps to your hardware;
multiple GPUs may be necessary. CPU-only figure and table regeneration is described
in the root README. Full generation/scoring was not rerun as a packaging check.

## Environment and inputs

Install the model-experiment dependencies in a separate environment:

```sh
python -m pip install -r requirements.txt
```

The dependency bounds are installation requirements, not a historical lockfile.
Record installed versions and resolved model, tokenizer, and dataset revisions
for your run. Acquire the public datasets and required model weights under their
respective access conditions. No model weights are included. Local RAID CSV or
JSONL input avoids the optional `raid-bench` downloader dependency.

Commands below use placeholders: replace `DATA_WORKSPACE` with your storage
directory and each `ABSOLUTE_CONFIG_PATH` with the full path to your configuration.
Use absolute config paths because relative paths resolve against the data
workspace. Keep generated data under this workspace and choose a new run ID when
changing settings. Resume only with the same ID and configuration.

## Controlled contamination

Use the corresponding recorded configuration in `configs/paper_cells/` for each
of the nine dataset-generator cells. These pin recovered model/tokenizer/dataset
revisions and the recorded generation, scoring and length-guard settings.
Configure device placement for your hardware. The observed package versions are
in `reference/provenance/primary/`; they are not a complete dependency lockfile.
Source identities, row indices, packing order, splits and text hashes are included.
Restore original manifests from the pinned public dataset revisions first:

```sh
python reproduce.py restore-sources --output DATA_WORKSPACE/source_manifests
```

The output must be new. Every source text and the entire original file hash are
verified before writing. Add `--original-directory ORIGINAL_SOURCE_DIRECTORY`
to use original JSONL files without downloads. Restored cache filenames match the
pinned configurations; original JSONL bytes are unchanged. Use the override below
to point experiments at this cache. Do not feed text-hash derivatives directly to
generation. A run without restored manifests remains a fresh replication.

For one example cell:

```sh
python -B run_experiment.py --workspace DATA_WORKSPACE --config ABSOLUTE_CONFIG_PATH --dataset xsum --model ibm-granite/granite-3.3-8b-base --run-id controlled-xsum-granite-01 --set paths.source_manifests=source_manifests --stage all
python -B tools/reanalysis/reevaluate_detector_revision.py --workspace DATA_WORKSPACE --source-run-id controlled-xsum-granite-01 --revision-id controlled-xsum-granite-corrected-01 --bootstrap-repetitions 2000
```

Repeat with unique IDs for the other eight cells. The second command is required:
the component-based analysis produces the final seven-detector definitions. It reuses saved token features and does not
repeat inference. Do not use `--skip-binoculars` for the paper's full detector set.

To repeat analysis after changing supported evaluation choices, retain the base
manifest and both `target_scores.jsonl` and `binoculars_scores.jsonl`, and give
the corrected analysis a new revision ID. Summary CSVs alone cannot refit scores.
Reduced sample sizes may be used for an execution check, but do not reproduce
the reference estimates.

## RAID: fresh data, scoring, and current analysis

Obtain the public labeled English RAID dataset and use `configs/raid.json` as a
configuration starting point. The Falcon model revisions are pinned to the recovered scoring record;
set CUDA device assignments for your installation. Use the original exclusion
file at `reference/provenance/raid/excluded_sources.json` for the paper sample.
Its 500 IDs do not overlap the 12,871 selected sources. Compare newly prepared
selections with `reference/provenance/raid/selected_sources.jsonl`. A different
exclusion list defines a fresh replication. Keep all variants of a source together. The implementation validates the
required exclusion count and recognizes specific release source totals; another
release requires a reviewed protocol update rather than bypassing validation.

Official origin scoring requires a separately obtained, reviewed upstream
Binoculars `metrics.py`. Obtain a pinned file with its license and record its
revision. Both `reference/provenance/raid/*-gate.json` files record the upstream
file SHA256; its contents are not bundled. Its numerical components are checked by the scoring gate. The upstream
project is [Binoculars](https://github.com/ahans30/Binoculars); moving upstream
code may be incompatible and should be reviewed before use.

Run the public preparation/scoring/current-evaluation sequence:

```sh
python -B RAID/run_raid.py --output outputs/raid-fresh --run-id raid-fresh --config configs/raid.json --data-path RAID_DATA_FILE --exclude-source-ids-path reference/provenance/raid/excluded_sources.json --reference-metrics REVIEWED_METRICS_PY --stage all
```

The stages are `prepare`, `score`, and `evaluate`. Preparation writes
`data.jsonl`; scoring writes `falcon_scores.jsonl` and `origin_scores.jsonl`;
current evaluation writes under `current-analysis/` in the run output. Scoring
loads the Falcon features and then the official origin pair. This portable path
uses serial scoring, without scheduler-specific launchers. One scoring process
does not imply one GPU: configure observer/performer devices for your hardware.
Resume only the same run identity, inputs, and configuration.

The current evaluator fits all four scopes using the cross-fitted TPR objective:
rate-specific and attack-specific are main; full-universal and eligible-universal
are secondary. Each rate bin must
have at least 250 attacked tuning rows, so arbitrarily tiny sample runs cannot
satisfy the full analysis contract.

To repeat current analysis on available compatible packs without new inference:

```sh
python -B RAID/analyze.py --falcon-scores outputs/raid-fresh/falcon_scores.jsonl --origin-scores outputs/raid-fresh/origin_scores.jsonl --run-id raid-tpr-repeat --output outputs/raid-tpr-repeat --bootstrap-repetitions 2000 --bootstrap-seed 481516
```

See [technical reference](TECHNICAL_REFERENCE.md) for shard/rate inputs,
fitting and output counts. The CPU table commands read the same current
four-scope results distributed under `reference/raid/`.

## Check completion

Inspect completion manifests and stage markers, row coverage, detector names,
metrics, and error output. A successful process exit alone is insufficient.
Use the corrected outputs for publication comparisons. Preserve your manifests,
resolved configuration, exclusions, token features, and environment record so
future analyses can reuse scoring. Current-code replications may differ from
reference results due to software, dataset, model, numerical, or sampling changes.
