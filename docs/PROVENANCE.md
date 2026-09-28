# Experiment provenance

## Source evidence

The recovered inputs identify the original paper samples. Public records are
covered by `reference/provenance/provenance.complete.json`. Original bytes and
transformed records are distinguished below. Inventory completeness does not
mean that all experiment inputs or score packs are distributed.

| Record under `reference/provenance/` | Representation |
|---|---|
| `raid/excluded_sources.json` | Original bytes, 500 unique development-source IDs |
| `raid/selected_sources.jsonl` | Original bytes, 12,871 sources, splits and selected attack-family IDs |
| `source_manifests/*.jsonl` | Three 3,000-source manifests with only `source_text` replaced by `source_text_sha256` |
| `source_manifests.json` | Original and distributed manifest hashes and sizes |
| `primary/*.json` | Nine anonymous run records, original hashes and preparation/scoring completion evidence |
| `raid/prepare.complete.json`, `raid/score.complete.json` | Original fields with private paths removed, plus original file hashes |
| `raid/origin-gate.json`, `raid/components-gate.json` | Both scoring parity records with original file hashes |
| `raid/*-shards.json` | Available scoring-shard completion/adoption evidence; not score packs |
| `analysis_evidence.json` | Original current-analysis hashes linked to distributed results |

The exclusion file has SHA256
`02e46d83e47a1decfe6a268190042bc0e9c2ff5e530bf6633bb1f9337299dc08`.
Its IDs do not overlap the selected RAID sources. The original selected-source
JSONL agrees with every source in `reference/raid/selected_sources.csv` and all
154,452 machine-row identities in `selected_machine_rows.csv`. Verification
checks these relationships and the 5,148/2,574/5,149 RAID split. The current
analysis manifest retains all tuning-source fold IDs.

## Restore controlled source manifests

The public manifests retain dataset revisions, original row indices, source IDs,
SQuAD packing order and component IDs, splits and generation seeds. Obtain raw
text separately under the source dataset terms. From the repository root:

```sh
python reproduce.py restore-sources --output DATA_WORKSPACE/source_manifests
```

This requires `datasets` and access to recorded dataset revisions. The tool uses
recorded row indices, applies the original whitespace normalization, joins packed
components in order, verifies every text hash, and verifies the complete JSONL
against the original file SHA256 and size. It refuses existing output directories
and mismatches. It does not reselect sources. Each dataset retains its original
500/500/2,000 tuning/calibration/test split.

Alternatively, add `--original-directory ORIGINAL_SOURCE_DIRECTORY` to use the
three original JSONL files without downloads. Original byte hashes are checked
before installation. Cache filenames match `configs/paper_cells/`, whose pinned
revision fields change cache signatures relative to the original configuration;
JSONL bytes remain unchanged. See [EXPERIMENTS.md](EXPERIMENTS.md) for cache use.

## Execution and software scope

`primary_cells.json` maps nine cells to recorded provenance and runnable configs.
They include resolved model/tokenizer/dataset revisions, generation/scoring
settings, seeds, length guards and observed versions. Omitted constructed-length
guards use the documented recorded-base-guard fallback.

`raid.json` records the dataset content hash, selection/split seeds, exclusions,
Falcon revisions, component revisions and original preparation/scoring software
versions. Both scoring gates record 314 same-logit upstream component parity
checks and passing tokenizer preflight with zero unscorable rows. These are
recorded checks, not newly rerun model checks. Derivatives retain reference hashes
and scoring settings; private paths and analysis-probe outputs are omitted.
Implementation fingerprints refer to original execution code, not a claim that
reorganized public code has identical bytes. Only exported shard records appear.

The current RAID analysis manifest has no independent software-environment
snapshot. Original scoring versions are not independently verified versions for
that later analysis. Dependency requirements are installation bounds, not a
recovered complete environment lockfile.

## Replay limits

Large token-score packs were intentionally not exported. Model weights,
prepared/generated text and upstream Binoculars `metrics.py` are not bundled.
The gates record the upstream file SHA256; neither contents nor a resolved
upstream commit can be recovered from a checksum alone. Obtain that input with
its license for new scoring.

Current RAID result, specification and calibration files remain byte-identical
to recovered originals. CPU table regeneration preserves their estimates; it
does not independently recompute intervals, refit clipping or repeat inference.
Source restoration establishes the original sample, not numerical identity of
a new model run. `availability.json` records these limits explicitly.
