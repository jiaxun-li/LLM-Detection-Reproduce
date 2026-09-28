"""Read-only score-pack helpers for current RAID analysis."""
from pathlib import Path
import json
from experiment_core.infrastructure.io import iter_jsonl

def identifier(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value) or value in {".", ".."}:
        raise ValueError("run IDs must be single safe directory names")
    return value

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def compact_rows(path, include_token_ids=False):
    import numpy as np
    from experiment_core.analysis.evaluation import EVALUATION_TOKEN_FEATURES
    retained = EVALUATION_TOKEN_FEATURES | ({"scored_token_ids", "full_scored_token_ids", "token_ids"} if include_token_ids else set())
    for row in iter_jsonl(path, tolerate_partial_last_line=False):
        yield {**{k:v for k,v in row.items() if k not in {"text","prompt","token_features","document_features"}},
            "token_features": {k:np.asarray(v,dtype=float) for k,v in row["token_features"].items()
                               if k in retained}}

def scorer_config(source):
    first = next(iter_jsonl(source / "binoculars_scores.jsonl"))
    observer = first["binoculars_observer_revision"]
    performer = first["binoculars_performer_revision"]
    # Require resolved immutable revisions, never silently fall back to main.
    for value in (observer, performer):
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("source pair must record resolved model commits")
    return {"observer_revision": observer, "performer_revision": performer,
        "tokenizer_revision": first["binoculars_tokenizer_revision"],
        "performer_tokenizer_revision": first["binoculars_performer_tokenizer_revision"],
        "dtype": "bf16", "max_tokens": 512,
        "observer_device": "cuda:0", "performer_device": "cuda:1",
        "trust_remote_code": False}
