"""Portable preparation, official scoring and current four-scope RAID analysis."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path

from experiment_core.infrastructure.io import atomic_write_json
from RAID.raid_data import RAID_ADVERSARIAL_ATTACKS, prepare_raid_data
from RAID.raid_scoring import RAIDFalconScorer, score_raid_jsonl
from RAID.binocular_origin import RAIDBinocularsOriginScorer

STAGES = ("prepare", "score", "evaluate")
EXPECTED_DETECTORS = ("log_likelihood", "rank", "log_rank", "lrr", "entropy", "entropy_gap", "binocular_origin")


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def load_raid_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config["dataset"]["required_attacks"]) != set(RAID_ADVERSARIAL_ATTACKS):
        raise ValueError("RAID requires all eleven official attacks")
    fractions = config["selection"]["split_fractions"]
    if fractions != {"clipping_tuning": .4, "calibration": .2, "test": .4}:
        raise ValueError("RAID requires the 40/20/40 source split")
    if config["selection"]["required_development_source_exclusions"] != 500:
        raise ValueError("RAID requires 500 development-source exclusions")
    if tuple(config["scoring"]["detectors"]) != EXPECTED_DETECTORS:
        raise ValueError("RAID requires six single-model detectors plus binocular_origin")
    if config["models"]["falcon"]["id"] != "tiiuae/falcon-7b":
        raise ValueError("single-model scorer must be Falcon-7B")
    pair = config["models"]["binoculars"]
    if pair["observer"] != "tiiuae/falcon-7b" or pair["performer"] != "tiiuae/falcon-7b-instruct":
        raise ValueError("origin requires the official Falcon pair")
    scoring = config["scoring"]
    if scoring["max_tokens"] != 512 or scoring["truncation_side"] != "right":
        raise ValueError("official scoring requires a right-truncated 512-token window")
    if scoring["dtype"] not in ("bf16", "bfloat16") or scoring.get("trust_remote_code", False):
        raise ValueError("official scoring requires native Falcon in BF16")
    evaluation = config["evaluation"]
    expected = {"target_fpr": .05, "clipping_quantiles": [.8,.85,.9,.95,.975,.99,.995],
                "attack_weight": .8, "clean_weight": .2, "crossfit_folds": 5,
                "crossfit_seed": 20260917, "rate_min_tuning_rows": 250}
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise ValueError(f"current four-scope protocol requires {key}={value}")
    if int(evaluation["bootstrap_repetitions"]) < 2:
        raise ValueError("at least two bootstrap repetitions required")
    return config


def load_source_exclusion_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("source_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values or any(not isinstance(v,str) or not v for v in values):
        raise ValueError("exclusions must be a nonempty list of source ID strings")
    if len(set(values)) != len(values):
        raise ValueError("duplicate exclusion IDs")
    if isinstance(payload, dict) and payload.get("source_count", len(values)) != len(values):
        raise ValueError("exclusion source_count mismatch")
    return set(values)


def prepare_stage(run_dir, config, *, data_path, source_exclusion_path, limit_sources=None):
    exclusions = load_source_exclusion_ids(source_exclusion_path)
    if len(exclusions) != config["selection"]["required_development_source_exclusions"]:
        raise ValueError("supply exactly 500 development exclusions, including for bounded preparation")
    summary = prepare_raid_data(Path(data_path), run_dir / "data.jsonl",
        run_dir / "selected_sources.jsonl", run_dir / "excluded_sources.json",
        run_dir / "raid_index.sqlite3", selection_seed=config["selection"]["selection_seed"],
        split_seed=config["selection"]["split_seed"], split_fractions=config["selection"]["split_fractions"],
        limit_sources=limit_sources, dataset_revision=config["dataset"].get("revision"),
        dataset_fingerprint=config["dataset"].get("fingerprint"), excluded_source_ids=exclusions)
    if limit_sources is None:
        if summary["missing_development_source_exclusions"]:
            raise ValueError("development exclusions are absent from the input release")
        recognized = config["dataset"].get("recognized_full_source_counts", [])
        before_exclusions = summary["selected_sources"] + summary["applied_development_source_exclusions"]
        if recognized and before_exclusions not in recognized:
            raise ValueError("unrecognized complete source count; review the dataset protocol")
    atomic_write_json(run_dir / "prepare.complete.json", {"status":"complete", **summary,
        "artifacts": {name:digest(run_dir/name) for name in ("data.jsonl","selected_sources.jsonl","excluded_sources.json")}})
    return summary


def validate_marker(run_dir, name):
    marker = json.loads((run_dir / name).read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"incomplete stage: {name}")
    for filename, expected in marker["artifacts"].items():
        if digest(run_dir / filename) != expected:
            raise ValueError(f"stage artifact changed: {filename}")
    return marker


def _release_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def score_stage(run_dir, config, *, reference_metrics):
    validate_marker(run_dir, "prepare.complete.json")
    scoring = config["scoring"]
    falcon = config["models"]["falcon"]
    scorer = RAIDFalconScorer({**scoring, "model_id":falcon["id"], "revision":falcon.get("revision"),
        "tokenizer_revision":falcon.get("tokenizer_revision"), "context_policy":"raid_output_only_512"})
    count = score_raid_jsonl(run_dir / "data.jsonl", run_dir / "falcon_scores.jsonl", scorer, scoring)
    del scorer
    _release_cuda()
    spec = importlib.util.spec_from_file_location("reviewed_binoculars_metrics", reference_metrics)
    if spec is None or spec.loader is None:
        raise ValueError("upstream metrics must be a loadable Python file")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    from RAID.revision_gate import preflight_inputs, run_pipeline_probe
    pair = config["models"]["binoculars"]
    scorer = RAIDBinocularsOriginScorer({**scoring, **pair}, reference_metrics=reference)
    preflight, selected = preflight_inputs(run_dir / "data.jsonl", [run_dir / "data.jsonl"],
                                         scorer.tokenizer, run_dir / "origin_preflight.json")
    probe = run_pipeline_probe(selected, scorer, run_dir)
    atomic_write_json(run_dir / "origin_gate.json", {"status":"pass", "preflight":preflight,
        "probe":probe, "reference_metrics":digest(reference_metrics)})
    # Keep the independent component check on every scored row, not just the probe.
    origin_count = score_raid_jsonl(run_dir / "data.jsonl", run_dir / "origin_scores.jsonl", scorer, scoring)
    del scorer
    _release_cuda()
    if origin_count != count:
        raise ValueError("Falcon and origin row counts differ")
    marker = {"status":"complete", "score_rows":count, "artifacts":{name:digest(run_dir/name) for name in
              ("falcon_scores.jsonl","origin_scores.jsonl","origin_gate.json","origin_preflight.json")}}
    atomic_write_json(run_dir / "score.complete.json", marker)
    return marker


def evaluate_stage(run_dir, config, *, run_id):
    validate_marker(run_dir, "prepare.complete.json")
    validate_marker(run_dir, "score.complete.json")
    from RAID.analyze import analyze_rows, load_score_rows
    paths = {"falcon_scores":run_dir / "falcon_scores.jsonl", "origin_scores":run_dir / "origin_scores.jsonl"}
    before = {name:digest(path) for name,path in paths.items()}
    rows = load_score_rows(paths["falcon_scores"], [paths["origin_scores"]])
    if before != {name:digest(path) for name,path in paths.items()}:
        raise ValueError("score inputs changed while loading")
    return analyze_rows(rows, output=run_dir / "current-analysis", run_id=run_id,
        repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
        seed=int(config["evaluation"]["bootstrap_seed"]), inputs=paths, input_fingerprints=before)
