"""Primary revision identity utilities, independent of RAID orchestration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def identifier(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value) or value in {".", ".."}:
        raise ValueError("run IDs must be single safe directory names")
    return value


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def implementation_fingerprint():
    names = (
        "experiment_core/detectors/detector_revision.py",
        "experiment_core/analysis/evaluation.py",
        "experiment_core/detectors/scoring.py",
        "experiment_core/infrastructure/revision_artifacts.py",
        "tools/reanalysis/reevaluate_detector_revision.py",
        "tools/reanalysis/primary_revision_io.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
