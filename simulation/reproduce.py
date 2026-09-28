"""Preservation-safe entry point for the frozen finite-state simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE / "original"


def verify() -> int:
    """Verify distributed original byte hashes, including the execution source identity."""
    expected = json.loads((ORIGINAL / "distribution_checksums.json").read_text())
    config = json.loads((ORIGINAL / "results/config.json").read_text())
    checks = [(name.replace("\\", "/"), digest) for name, digest in expected.items()]
    checks.extend(config["source_sha256"].items())
    for name, digest in checks:
        path = ORIGINAL / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen simulation artifact does not match: {name}")
    print(f"Verified {len(expected)} distribution hashes and {len(config['source_sha256'])} execution-source hashes.")
    return len(checks)


def fresh_output(value: str) -> Path:
    out = Path(value).expanduser().resolve()
    protected = (PACKAGE, PACKAGE.parent / "paper", PACKAGE.parent / "downloads", PACKAGE.parent / "reference")
    if any(out.is_relative_to(path) for path in protected):
        raise ValueError("Output must be outside simulation/, paper/, and downloads/ to protect reference materials.")
    if out.exists():
        raise ValueError(f"Output already exists; choose a new directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    return out


def execute(out: Path, script: str, *args: str) -> None:
    subprocess.run([sys.executable, "-B", str(out / script), *args], cwd=out, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check original source and frozen result bytes")
    for name in ("summarize", "figure", "run"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, help="New directory outside simulation/")
        if name == "figure":
            command.add_argument("--compile", action="store_true", help="Compile PDF with pdflatex")
        if name == "run":
            command.add_argument("--paths", type=int, default=2000)
            command.add_argument("--calibration", type=int, default=4000)
            command.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 4096])
            command.add_argument("--orders", type=int, nargs="+", default=[10, 100])
            command.add_argument("--supports", type=int, nargs="+", default=[500, 1000])
            command.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    verify()
    if args.command == "verify":
        return
    if args.command == "run":
        if min(args.paths, args.calibration, *args.lengths, *args.orders) <= 0:
            parser.error("Paths, calibration paths, lengths, and orders must be positive")
        if any(s not in (500, 1000) for s in args.supports):
            parser.error("Supported study support sizes are 500 and 1000")
        if args.seed < 0:
            parser.error("Seed must be nonnegative")
    if args.command == "figure" and args.compile and shutil.which("pdflatex") is None:
        parser.error("PDF compilation requires pdflatex with standalone, TikZ and PGFPlots")
    out = fresh_output(args.output)
    for script in ORIGINAL.glob("*.py"):
        shutil.copyfile(script, out / script.name)
    if args.command in ("summarize", "figure"):
        shutil.copytree(ORIGINAL / "results", out / "results")
    if args.command == "run":
        forwarded = []
        for key in ("paths", "calibration", "lengths", "orders", "supports", "seed"):
            value = getattr(args, key)
            forwarded.extend([f"--{key}", *(str(v) for v in (value if isinstance(value, list) else [value]))])
        execute(out, "run_large.py", *forwarded, "--output", "results")
        execute(out, "summarize.py", "--output", "results")
    elif args.command == "summarize":
        execute(out, "summarize.py", "--output", "results")
    else:
        execute(out, "build_paper_figure.py")
        if args.compile:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory=figure_build", "figure_build/section4_alt_power.tex"],
                           cwd=out, check=True)
    print(f"Reproduction outputs: {out}")


if __name__ == "__main__":
    main()
