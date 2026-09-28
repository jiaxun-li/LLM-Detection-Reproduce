"""Full-input tokenizer preflight and bounded real scoring/serialization gate."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from experiment_core.infrastructure.io import AppendSafeJsonlWriter, atomic_write_json, iter_jsonl
from RAID.raid_data import raid_row_key
from RAID.raid_data import RAID_ADVERSARIAL_ATTACKS


def source(row):
    return str(row["source_id"])


def domain(row):
    return str(row["domain"])


def split(row):
    return str(row["split"])


def human(row):
    return row["label"] == "human"


def condition(row):
    return str(row.get("attack", "none"))



def load_gate_tokenizer(config):
    from experiment_core.detectors.scoring import _transformers
    _, auto_tokenizer = _transformers()
    tokenizer = auto_tokenizer.from_pretrained(
        "tiiuae/falcon-7b", revision=config["tokenizer_revision"],
        trust_remote_code=False, use_fast=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("official tokenizer requires a pad/EOS ID")
    return tokenizer


def _record_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).digest()


def preflight_inputs(data_path, shard_paths, tokenizer, report_path):
    """Tokenize all inputs without inference; do not alter/drop invalid texts."""
    fingerprints = {}
    chosen_sources = {}
    selected = []
    families = {}
    source_groups = {}
    bad = []
    invalid_count = 0
    total = 0
    batch = []

    def check_batch(values):
        nonlocal invalid_count
        enc = tokenizer([r["text"] for r in values], padding=False, truncation=True,
                        max_length=512, return_token_type_ids=False)
        if len(enc["input_ids"]) != len(values):
            raise ValueError("tokenizer preflight returned a different row count")
        for row, ids in zip(values, enc["input_ids"]):
            if len(ids) < 2 or not any(i != tokenizer.pad_token_id for i in ids):
                invalid_count += 1
                if len(bad) < 20:
                    bad.append({"key":list(raid_row_key(row)), "input_tokens":len(ids)})

    for row in iter_jsonl(data_path, tolerate_partial_last_line=False):
        key = raid_row_key(row)
        if key in fingerprints:
            raise ValueError(f"duplicate prepared key: {key}")
        fingerprints[key] = _record_digest(row)
        sid = source(row)
        group = (split(row), domain(row))
        if sid in source_groups and source_groups[sid] != group:
            raise ValueError(f"source split/domain leakage: {sid}")
        source_groups[sid] = group
        families.setdefault(sid, Counter())["human" if human(row) else condition(row)] += 1
        chosen_sources.setdefault(group, sid)
        if chosen_sources[group] == sid:
            selected.append(row)
        batch.append(row)
        total += 1
        if len(batch) == 64:
            check_batch(batch)
            batch = []
        if total % 10000 == 0:
            print(f"origin tokenizer preflight: {total} rows", flush=True)
    if batch:
        check_batch(batch)
    # Validate source families as well as the three independent splits.
    if not families:
        raise ValueError("prepared data is empty")
    expected_family = Counter({"human": 1, "none": 1, **{a: 1 for a in RAID_ADVERSARIAL_ATTACKS}})
    if any(value != expected_family for value in families.values()):
        raise ValueError("incomplete source family in full prepared data")
    seen = set()
    for shard in shard_paths:
        for row in iter_jsonl(shard, tolerate_partial_last_line=False):
            key = raid_row_key(row)
            if key in seen or fingerprints.get(key) != _record_digest(row):
                raise ValueError(f"prepared shard duplicate/content mismatch: {key}")
            seen.add(key)
    if seen != set(fingerprints):
        raise ValueError("prepared shards do not cover the source data exactly")
    report = {"status":"pass" if invalid_count == 0 else "fail", "rows":total,
              "sources":len(families), "unscorable_rows":invalid_count,
              "unscorable_examples":bad, "shard_content_verified":True,
              "probe_sources":len(chosen_sources), "probe_rows":len(selected)}
    atomic_write_json(report_path, report)
    if invalid_count:
        raise ValueError(f"{invalid_count} unscorable official windows; see {report_path}; no texts changed")
    print(f"origin tokenizer preflight: PASS ({total} rows)", flush=True)
    return report, selected


def run_pipeline_probe(selected, scorer, destination):
    """Bounded official component parity, serialization and no-op resume check."""
    from RAID.raid_scoring import score_raid_jsonl
    from experiment_core.infrastructure.revision_artifacts import artifact_digest

    with tempfile.TemporaryDirectory(prefix="gate-probe-", dir=destination) as temp:
        root = Path(temp)
        prepared, scored = root / "data.jsonl", root / "origin.jsonl"
        with AppendSafeJsonlWriter(prepared) as writer:
            for row in selected:
                writer.write(row)
        options = {"batch_size": 1, "microbatch_size": 1, "checkpoint_interval": 10}
        calls_before = scorer.parity_checks
        count = score_raid_jsonl(prepared, scored, scorer, options)
        before, calls = artifact_digest(scored), scorer.parity_checks
        resumed = score_raid_jsonl(prepared, scored, scorer, options)
        if (count != len(selected) or resumed != count or before != artifact_digest(scored)
                or calls != scorer.parity_checks or calls - calls_before != count):
            raise ValueError("gate parity/serialization/resume check failed")
        return {"status": "pass", "score_rows": count, "resume_unchanged": True,
                "component_parity_checks": count}
