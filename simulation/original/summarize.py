"""Generate a portable report and SVG from completed large-state simulations."""

import argparse
import csv
import html
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def yes(value):
    return str(value).lower() == "true"


def fmt(value):
    return f"{float(value):.4f}"


def plot(data, destination):
    selected = [r for r in data if r["profile"] == "mild" and int(r["support_size"]) == 1000 and yes(r["theorem_selected"]) and r["attack"] == "minimum_force_zero" and r["calibration"] == "calibrated_5pct"]
    detectors = list(dict.fromkeys(r["detector"] for r in selected))
    orders = sorted({int(r["K"]) for r in selected})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="760" viewBox="0 0 1320 760">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#243347}.title{font-size:20px;font-weight:bold}.panel{font-size:14px;font-weight:bold}</style>', '<text x="25" y="30" class="title">10,000-token Markov model: clipped versus raw power</text>', '<text x="25" y="52">Mild history variation · support 1,000 · detector-specific certified epsilon · adaptive force-zero attack</text>', '<text x="1010" y="30" style="fill:#2563eb">Raw (blue)</text><text x="1140" y="30" style="fill:#ea580c">Clipped (orange)</text>']
    for row_index, order in enumerate(orders):
        for column, detector in enumerate(detectors):
            subset = [r for r in selected if int(r["K"]) == order and r["detector"] == detector]
            lengths = sorted({int(r["n"]) for r in subset})
            if not lengths:
                continue
            x0, y0, pw, ph = 55 + 325 * column, 120 + 305 * row_index, 245, 215
            lmin, lmax = math.log2(min(lengths)), math.log2(max(lengths))
            x = lambda n: x0 + pw * (math.log2(n) - lmin) / max(1, lmax - lmin)
            y = lambda p: y0 + ph * (1 - float(p))
            svg.append(f'<text x="{x0}" y="{y0-27}" class="panel">K={order} · {html.escape(detector)}</text>')
            svg.append(f'<text x="{x0}" y="{y0-9}">epsilon={float(subset[0]["epsilon"]):.4f}</text>')
            for tick in (0, .25, .5, .75, 1):
                svg.append(f'<path d="M{x0},{y(tick):.2f}h{pw}" stroke="#e2e8f0"/><text x="{x0-30}" y="{y(tick)+4:.2f}">{tick:g}</text>')
            for aggregation, color in (("raw", "#2563eb"), ("clipped", "#ea580c")):
                series = sorted((r for r in subset if r["aggregation"] == aggregation), key=lambda r: int(r["n"]))
                points = " ".join(f'{x(int(r["n"])):.2f},{y(r["tpr"]):.2f}' for r in series)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                for r in series:
                    px = x(int(r["n"]))
                    svg.append(f'<path d="M{px:.2f},{y(r["tpr_low"]):.2f}V{y(r["tpr_high"]):.2f}" stroke="{color}" opacity=".6"/><circle cx="{px:.2f}" cy="{y(r["tpr"]):.2f}" r="3" fill="{color}"/>')
            for n in lengths:
                svg.append(f'<text x="{x(n):.2f}" y="{y0+ph+22}" text-anchor="middle">{n}</text>')
    svg += ['<text x="25" y="700">Horizontal axis: generated sequence length (log2 scale). K is the actual memory length, not the generated length.</text>', '<text x="25" y="723">Separately calibrated 5% FPR tests. Bars: 95% Wilson intervals conditional on fitted thresholds; no simultaneous-coverage claim.</text>', '</svg>']
    destination.write_text("\n".join(svg), encoding="utf-8")


def summarize(folder):
    audit, exact, mc = (rows(folder / name) for name in ("assumption_audit.csv", "exact_means.csv", "monte_carlo.csv"))
    coverage = json.loads((folder / "coverage.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    json.loads((folder / "completion.json").read_text(encoding="utf-8"))
    lines = ["# Large-vocabulary Section 4 simulation", "", "The construction uses 10,000 actual token identities, conditional supports of 500 or 1,000 tokens, and actual history lengths K=10 or K=100. The simulator maintains the full rolling history. Conditional score distributions depend on the parity of all K identity bits, enabling exact calculations using two score regimes without enumerating the exponentially large history space. This is structured compression, not a claim that arbitrary large-state Markov models are easy to analyze.", "", "The reference [A Statistical Framework of Watermarks for Large Language Models](https://arxiv.org/pdf/2404.01245) motivates comparison with a concentrated synthetic probability model; this simulation studies clipping, not watermark detection, and does not reproduce that paper's experiment. Full human and target profiles are saved in [profiles.json](profiles.json).", "", "## Exact assumption checks", "", "Strong means the simpler clean-distribution sufficient inequality. Certified means the sharper contamination interval is nonempty. Every count below is out of four additive detectors. The same clean distribution calculations apply at both history orders, but finite-length behavior can differ.", "", "| Profile | Conditional support | K | Strong | Certified |", "|---|---:|---:|---:|---:|"]
    for profile, support, order in sorted({(r["profile"], int(r["support_size"]), int(r["K"])) for r in audit}):
        subset = [r for r in audit if r["profile"] == profile and int(r["support_size"]) == support and int(r["K"]) == order]
        lines.append(f'| {profile} | {support} | {order} | {sum(yes(r["strong_condition"]) for r in subset)}/{len(subset)} | {sum(yes(r["certified"]) for r in subset)}/{len(subset)} |')
    certified = {(r["profile"], r["support_size"], r["K"], r["detector"]) for r in audit if yes(r["certified"])}
    chosen = [r for r in exact if yes(r["theorem_selected"]) and (r["profile"], r["support_size"], r["K"], r["detector"]) in certified]
    lines += ["", "The tables use detector-specific selected contamination levels; they do not imply a single contamination value works for all four detectors. See [assumption_audit.csv](assumption_audit.csv) for endpoints and the common-interval flag.", "", "## Exact means and robustness", "", "The attack-specific raw and clipped means in [exact_means.csv](exact_means.csv) are exact stationary means for the listed attacks. They are **not global worst-policy minima**. Separately, the uniform adaptive clipped lower gap is the theorem's guarantee against every admissible adaptive contamination strategy."]
    if chosen:
        gaps = [float(r["uniform_adaptive_clipped_lower_gap"]) for r in chosen]
        lines += ["", f"Across {len(chosen)} certified selected attack rows, the uniform adaptive clipped lower gap ranges from {min(gaps):.6g} to {max(gaps):.6g}. These gaps are in detector-specific score units and should not be compared in magnitude across detectors."]
        raw = [float(r["raw_gap"]) for r in chosen if r["attack"] in {"minimum_zero", "minimum_force_zero"}]
        lines.append(f"Among the {len(raw)} corresponding minimum-score attack rows, {sum(g < 0 for g in raw)} have strictly negative exact raw mean gaps. A negative attained raw gap demonstrates a failing allowed attack; the positive uniform clipped bound establishes robust asymptotic separation.")
    maxn = max(int(r["n"]) for r in mc)
    lines += ["", "## Power and actual FPR at the largest length", "", f"Length {maxn}, mild profile, detector-specific selected epsilon, minimum-force-zero attack. Raw and clipped tests use separate independent human calibrations.", "", "| Support | K | Detector | Calibration | Epsilon | Raw power | Clipped power | Raw FPR | Clipped FPR |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    selected = [r for r in mc if r["profile"] == "mild" and yes(r["theorem_selected"]) and r["attack"] == "minimum_force_zero" and int(r["n"]) == maxn]
    for support, order, detector, calibration in sorted({(int(r["support_size"]), int(r["K"]), r["detector"], r["calibration"]) for r in selected}):
        pair = {r["aggregation"]: r for r in selected if int(r["support_size"]) == support and int(r["K"]) == order and r["detector"] == detector and r["calibration"] == calibration}
        if "raw" in pair and "clipped" in pair:
            raw, clip = pair["raw"], pair["clipped"]
            lines.append(f'| {support} | {order} | {detector} | {calibration} | {fmt(raw["epsilon"])} | {fmt(raw["tpr"])} | {fmt(clip["tpr"])} | {fmt(raw["fpr"])} | {fmt(clip["fpr"])} |')
    lines += ["", "![Power curves](plots.svg)", "", "## Coverage and limits", ""]
    if coverage:
        counts = [int(r["visited_token_count"]) for r in coverage]
        lines += [f"Across {len(coverage)} simulated batches, the number of visited token identities ranges from {min(counts)} to {max(counts)} out of 10,000; {sum(c == 10000 for c in counts)} batches visit all 10,000. All identities are reachable by construction, but a finite batch need not visit them all.", "", f'Full-history parity checks pass in {sum(bool(r["final_history_parity_check"]) for r in coverage)}/{len(coverage)} batches. Coverage metadata are in [coverage.json](coverage.json).']
    lines += ["", f'- Evaluation paths: {config["paths"]}; calibration paths: {config["calibration"]}; lengths: {config["lengths"]}; seed: {config["seed"]}.', "- K is the number of previous tokens used by the source, not the continuation length. All paths start from K zero tokens, so short sequences include initialization effects.", "- Empirical 5% calibration targets a fixed FPR; it does not establish FPR tending to zero. Fixed-theorem thresholds provide the separate vanishing-error construction.", "- Wilson 95% intervals in monte_carlo.csv are conditional on fitted calibration thresholds. They exclude calibration-sample uncertainty and are not simultaneous intervals across all cells.", "- Exact human means concern the specified human kernel in each profile. No trained-LLM or real-human-distribution claim is made.", "- Only four additive detectors are included. Ratio-based detectors and entropy require separate arguments.", "- Failed sufficient conditions do not prove clipping cannot help. Selected-attack simulations do not exhaust adaptive attacks; the theorem lower bound supplies that protection where positive.", "", "Source hashes and execution settings: [config.json](config.json). Completion status: [completion.json](completion.json)."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(mc, folder / "plots.svg")
    print(f"Generated report.md and plots.svg in {folder.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    folder = (base / args.output).resolve()
    if not folder.is_relative_to(base):
        raise ValueError("output must remain inside simulation_section4_large")
    summarize(folder)
