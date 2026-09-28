#!/usr/bin/env python3
"""Prepare, officially score, and evaluate a fresh four-scope RAID replication."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_core.infrastructure.io import atomic_write_json
from RAID.raid_pipeline import STAGES, digest, load_raid_config, prepare_stage, score_stage, evaluate_stage, validate_marker
from RAID.score_io import identifier


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", type=identifier, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/raid.json")
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--exclude-source-ids-path", type=Path, required=True)
    p.add_argument("--reference-metrics", type=Path, required=True)
    p.add_argument("--stage", choices=(*STAGES,"all"), default="all")
    p.add_argument("--limit-sources", type=int)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_raid_config(args.config)
    if args.limit_sources is not None and args.limit_sources < 40:
        raise ValueError("bounded preparation requires at least 40 sources")
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "outputs").resolve()) or out == (ROOT / "outputs").resolve():
        raise ValueError("run output must be below repository outputs/")
    identity = {"run_id":args.run_id, "configuration":config, "limit_sources":args.limit_sources,
        "input_hashes":{name:digest(path) for name,path in (("data",args.data_path),
            ("exclusions",args.exclude_source_ids_path),("reference_metrics",args.reference_metrics))},
        "implementation_hashes":{str(p.relative_to(ROOT)):digest(p) for p in sorted([*(ROOT/"RAID").glob("*.py"), *(ROOT/"experiment_core").rglob("*.py")])}}
    marker_path = out / "manifest.json"
    if out.exists():
        if not marker_path.is_file():
            raise ValueError("existing output has no run identity; choose a new output directory")
        manifest = json.loads(marker_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k,v in identity.items()):
            raise ValueError("resume identity/configuration/input/code mismatch; use a new output")
    else:
        from tools.reproduce.verify import new_output
        new_output(out)
        out.mkdir(parents=True)
        manifest = {**identity, "completion_status":"running", "completed_stages":[]}
        atomic_write_json(marker_path, manifest)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage in manifest["completed_stages"]:
            if stage in ("prepare","score"):
                validate_marker(out, stage + ".complete.json")
            else:
                complete=json.loads((out/"current-analysis/focused.complete.json").read_text(encoding="utf-8"))
                if (complete.get("status") != "complete" or complete.get("specifications") != 119
                        or complete.get("result_rows") != 462):
                    raise ValueError("incomplete four-scope analysis marker")
                for name, expected in complete["artifacts"].items():
                    if digest(out/"current-analysis"/name) != expected:
                        raise ValueError("analysis artifact changed")
            continue
        manifest["completion_status"] = "running"
        atomic_write_json(marker_path, manifest)
        try:
            if stage == "prepare":
                prepare_stage(out,config,data_path=args.data_path,
                    source_exclusion_path=args.exclude_source_ids_path,limit_sources=args.limit_sources)
            elif stage == "score":
                score_stage(out,config,reference_metrics=args.reference_metrics)
            else:
                evaluate_stage(out,config,run_id=args.run_id)
        except Exception as error:
            manifest["completion_status"]="failed"
            manifest["failure"]={"stage":stage,"type":type(error).__name__,"message":str(error)}
            atomic_write_json(marker_path,manifest)
            raise
        manifest["completed_stages"].append(stage)
        manifest.pop("failure",None)
        manifest["completion_status"]="complete" if set(STAGES)<=set(manifest["completed_stages"]) else "running"
        atomic_write_json(marker_path,manifest)
    print(f"RAID {manifest['completion_status']}: {out}")


if __name__ == "__main__":
    main()
