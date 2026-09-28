"""Rebuild primary report tables and figures from the immutable publication CSVs.

Run from any directory with the bundled scientific Python environment. No model
inference, detector fitting, or bootstrap recomputation is performed. All means
across cells are descriptive equal-cell means, not new inferential estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO / "reference/empirical"
DETECTORS = ["log_likelihood", "rank", "log_rank", "lrr", "entropy", "entropy_gap", "binocular_origin"]
LABELS = {"log_likelihood": "Log likelihood", "rank": "Rank", "log_rank": "Log rank", "lrr": "LRR", "entropy": "Entropy", "entropy_gap": "Entropy gap", "binocular_origin": "Binoculars"}
DATASETS = ["xsum", "squad", "writingprompts"]
DATASET_LABELS = {"xsum": "XSum", "squad": "SQuAD", "writingprompts": "WritingPrompts"}
MODELS = ["ibm-granite/granite-3.3-8b-base", "mistralai/Mistral-Small-24B-Base-2501", "Qwen/Qwen2.5-32B"]
MODEL_LABELS = dict(zip(MODELS, ["Granite 8B", "Mistral 24B", "Qwen 32B"]))
RATIOS = [0, .05, .1, .2, .3, .4, .5]
STYLES = {("random", "raw"): ("#2166ac", "--", "o"), ("random", "clipped"): ("#2166ac", "-", "s"), ("tail", "raw"): ("#b2182b", "--", "o"), ("tail", "clipped"): ("#b2182b", "-", "s")}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def num(row, key):
    value = float(row[key])
    assert math.isfinite(value), (key, value)
    return value


def key(row):
    return (row["dataset"], row["model"], row["detector"], row["contamination_mode"], row["aggregation"], num(row, "target_fpr"), num(row, "requested_contamination_ratio"))


def macro_name(detector):
    return "".join(x.title() for x in detector.split("_"))


def draw_curves(c, records, dataset, model, detector, rect, target=.05, bands=False, xlabels=True, ylabels=True):
    left, bottom, width, height = rect
    tx = lambda x: left + x / .5 * width
    ty = lambda y: bottom + y * height
    c.setLineWidth(.3)
    c.setStrokeColor(Color(.88, .88, .88))
    for v in (0, .5, 1):
        c.line(left, ty(v), left + width, ty(v))
    for v in RATIOS[::1]:
        if v != .05:
            c.line(tx(v), bottom, tx(v), bottom + height)
    c.setFillColor(Color(.15, .15, .15))
    c.setFont("Helvetica", 6.6)
    if ylabels:
        for v in (0, .5, 1):
            c.drawRightString(left - 3, ty(v) - 2.2, f"{v:g}")
    if xlabels:
        for v in (0, .1, .2, .3, .4, .5):
            c.drawCentredString(tx(v), bottom - 9, f"{100*v:.0f}")
    for (mode, aggregation), (color, style, marker) in STYLES.items():
        rows = [records[(dataset, model, detector, mode, aggregation, target, ratio)] for ratio in RATIOS]
        y = [num(row, "tpr") for row in rows]
        if bands:
            c.saveState()
            c.setFillColor(HexColor(color))
            c.setFillAlpha(.09)
            path = c.beginPath()
            path.moveTo(tx(RATIOS[0]), ty(num(rows[0], "tpr_ci_low")))
            for x, row in zip(RATIOS, rows):
                path.lineTo(tx(x), ty(num(row, "tpr_ci_low")))
            for x, row in reversed(list(zip(RATIOS, rows))):
                path.lineTo(tx(x), ty(num(row, "tpr_ci_high")))
            path.close()
            c.drawPath(path, stroke=0, fill=1)
            c.restoreState()
        c.setStrokeColor(HexColor(color))
        c.setFillColor(HexColor(color))
        c.setLineWidth(1)
        c.setDash(3, 2) if style == "--" else c.setDash()
        path = c.beginPath()
        path.moveTo(tx(RATIOS[0]), ty(y[0]))
        for x, val in zip(RATIOS[1:], y[1:]):
            path.lineTo(tx(x), ty(val))
        c.drawPath(path)
        c.setDash()
        for x, val in zip(RATIOS, y):
            if marker == "o":
                c.circle(tx(x), ty(val), 1.1, stroke=0, fill=1)
            else:
                c.rect(tx(x)-1.1, ty(val)-1.1, 2.2, 2.2, stroke=0, fill=1)


def figure_legend(c, y):
    for i, ((mode, aggregation), (color, style, marker)) in enumerate(STYLES.items()):
        x = 58 + i * 101
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1)
        c.setDash(3, 2) if style == "--" else c.setDash()
        c.line(x, y, x + 15, y)
        c.setDash()
        c.setFillColor(Color(.1,.1,.1))
        c.setFont("Helvetica", 7)
        c.drawString(x+19, y-2, f"{mode.title()} / {aggregation}")


def finish_figure(c, target):
    c.showPage()
    c.save()
    if png:
        subprocess.run([shutil.which("pdftoppm"), "-singlefile", "-r", "180", "-png", str(target.with_suffix(".pdf")), str(target)], check=True, capture_output=True)


def build(bundle, output, previews=False):
    global png
    png = previews
    from types import SimpleNamespace
    args = SimpleNamespace(bundle=bundle)

    source = args.bundle / "primary"
    generated = output / "generated"
    figures = output / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    source_paths = [source / name for name in ("primary_all_metrics.csv", "primary_robustness_auc.csv", "primary_cell_inventory.csv")]
    metrics, robustness, inventory = map(read_csv, source_paths)
    assert (len(metrics), len(robustness), len(inventory)) == (3528, 504, 9)
    assert set(row["detector"] for row in metrics) == set(DETECTORS)
    records = {key(row): row for row in metrics}
    assert len(records) == len(metrics), "Duplicate metric key"
    assert all(int(row["bootstrap_repetitions"]) == 2000 for row in metrics)
    assert all(row["analysis"] == "primary_frozen_mixture" for row in metrics)
    auc_records = {}
    for row in robustness:
        auc_key = (row["dataset"], row["model"], row["detector"], row["contamination_mode"], row["aggregation"], num(row, "target_fpr"))
        assert auc_key not in auc_records
        auc_records[auc_key] = row
        ys = [num(records[auc_key + (ratio,)], "tpr") for ratio in RATIOS]
        calculated = float(np.trapezoid(ys, RATIOS) / .5)
        assert math.isclose(calculated, num(row, "robustness_auc_tpr_vs_contamination"), rel_tol=0, abs_tol=1e-12), auc_key

    cell_results = []
    summaries = []
    for target in (.01, .05):
        for detector in DETECTORS:
            by_detector = []
            for dataset in DATASETS:
                for model in MODELS:
                    raw = records[(dataset, model, detector, "random", "raw", target, 0)]
                    clipped = records[(dataset, model, detector, "random", "clipped", target, 0)]
                    item = {"dataset": dataset, "model": model, "detector": detector, "target_fpr": target,
                            "clean_tpr_raw": num(raw, "tpr"), "clean_tpr_clipped": num(clipped, "tpr"), "clean_tpr_difference": num(clipped, "tpr") - num(raw, "tpr"),
                            "held_out_fpr_raw": num(raw, "actual_fpr"), "held_out_fpr_clipped": num(clipped, "actual_fpr"), "held_out_fpr_difference": num(clipped, "actual_fpr") - num(raw, "actual_fpr"),
                            "clean_auroc_raw": num(raw, "auroc"), "clean_auroc_clipped": num(clipped, "auroc"), "clean_partial_auroc_raw": num(raw, "partial_auroc_0_5_fpr"), "clean_partial_auroc_clipped": num(clipped, "partial_auroc_0_5_fpr"), "direction": int(raw["direction"]), "clipping_specification": clipped["clipping_specification"]}
                    for mode in ("random", "tail"):
                        for aggregation in ("raw", "clipped"):
                            arow = auc_records[(dataset, model, detector, mode, aggregation, target)]
                            item[f"{mode}_robustness_{aggregation}"] = num(arow, "robustness_auc_tpr_vs_contamination")
                            item[f"{mode}_attacked_mean_auroc_{aggregation}"] = mean(num(records[(dataset, model, detector, mode, aggregation, target, ratio)], "auroc") for ratio in RATIOS[1:])
                            item[f"{mode}_attacked_mean_partial_auroc_{aggregation}"] = mean(num(records[(dataset, model, detector, mode, aggregation, target, ratio)], "partial_auroc_0_5_fpr") for ratio in RATIOS[1:])
                        item[f"{mode}_robustness_difference"] = item[f"{mode}_robustness_clipped"] - item[f"{mode}_robustness_raw"]
                    cell_results.append(item)
                    by_detector.append(item)
            summary = {"detector": detector, "target_fpr": target, "n_cells": len(by_detector)}
            for field in ("clean_tpr_raw", "clean_tpr_clipped", "clean_tpr_difference", "held_out_fpr_raw", "held_out_fpr_clipped", "held_out_fpr_difference", "random_robustness_raw", "random_robustness_clipped", "random_robustness_difference", "tail_robustness_raw", "tail_robustness_clipped", "tail_robustness_difference"):
                summary[field] = mean(row[field] for row in by_detector)
            for mode in ("random", "tail"):
                summary[f"{mode}_positive_cells"] = sum(row[f"{mode}_robustness_difference"] > 1e-12 for row in by_detector)
                summary[f"{mode}_negative_cells"] = sum(row[f"{mode}_robustness_difference"] < -1e-12 for row in by_detector)
            summaries.append(summary)
    write_csv(generated / "primary_cell_results.csv", cell_results)
    write_csv(generated / "primary_summary.csv", summaries)

    # Compact main table: area values and differences use percentage-point scale.
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"& \multicolumn{2}{c}{Random area} & \multicolumn{2}{c}{Tail area} & \multicolumn{2}{c}{Clean change} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Detector & Raw & Clip & Raw & Clip & TPR & FPR \\", r"\midrule"]
    for row in (s for s in summaries if s["target_fpr"] == .05):
        vals = [100 * row[f"{mode}_robustness_{agg}"] for mode in ("random", "tail") for agg in ("raw", "clipped")]
        textvals = [f"{v:.1f}" for v in vals]
        textvals += [f"{100*row['clean_tpr_difference']:+.1f}", f"{100*row['held_out_fpr_difference']:+.2f}"]
        lines.append(LABELS[row["detector"]] + " & " + " & ".join(textvals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_text(generated / "primary_area_table.tex", "\n".join(lines))

    # Point TPRs: equal weight for each dataset-generator cell, clean counted once.
    point_results = []
    for detector in DETECTORS:
        item = {"detector": detector, "target_fpr": .05, "n_cells": 9}
        for label, mode, ratio in (("clean", "random", 0),
                                   ("random20", "random", .2), ("tail20", "tail", .2)):
            for aggregation in ("raw", "clipped"):
                selected = [records[(dataset, model, detector, mode, aggregation, .05, ratio)]
                            for dataset in DATASETS for model in MODELS]
                assert len(selected) == 9
                item[f"{label}_{aggregation}"] = mean(num(row, "tpr") for row in selected)
                if ratio == 0:
                    for dataset in DATASETS:
                        for model in MODELS:
                            assert num(records[(dataset, model, detector, "random", aggregation, .05, 0)], "tpr") == num(records[(dataset, model, detector, "tail", aggregation, .05, 0)], "tpr")
        point_results.append(item)
    write_csv(generated / "primary_point_tpr_summary.csv", point_results)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{0\% contamination} & \multicolumn{4}{c}{20\% contamination} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r"& \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Random} & \multicolumn{2}{c}{Tail} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Detector & Raw & Clipped & Raw & Clipped & Raw & Clipped \\", r"\midrule"]
    for item in point_results:
        values = [f"{100*item[f'{label}_{aggregation}']:.1f}"
                  for label in ("clean", "random20", "tail20") for aggregation in ("raw", "clipped")]
        lines.append(LABELS[item["detector"]] + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_text(generated / "primary_main_table.tex", "\n".join(lines))

    s5 = {r["detector"]: r for r in summaries if r["target_fpr"] == .05}
    eg, orig = s5["entropy_gap"], s5["binocular_origin"]
    qwen_squad_origin = next(r for r in cell_results if r["dataset"] == "squad" and r["model"] == MODELS[2] and r["detector"] == "binocular_origin" and r["target_fpr"] == .05)
    write_text(generated / "primary_area_interpretation.tex", (
        "Clipping improves the mean contamination-curve area for all seven detectors, "
        "but the magnitude differs substantially. The largest random-splice gains occur "
        f"for entropy gap ({100*eg['random_robustness_difference']:.1f} percentage points) "
        f"and Binoculars ({100*orig['random_robustness_difference']:.1f} points); "
        f"their tail-splice gains are {100*eg['tail_robustness_difference']:.1f} and {100*orig['tail_robustness_difference']:.1f} points. "
        "Rank and LRR also gain materially, whereas likelihood, log rank, and entropy "
        "show smaller average changes. These are descriptive equal-cell averages, not "
        "pooled-source estimates or simultaneous significance claims. The trade-off "
        "is visible on clean text: clipping reduces mean clean TPR for five methods. "
        "Separate calibration does not force identical held-out FPR, so the per-cell "
        "calibration results remain essential to interpreting these gains."
    ))
    point_by_detector = {r["detector"]: r for r in point_results}
    paragraphs = []
    for detector in ("entropy_gap", "binocular_origin"):
        row = point_by_detector[detector]
        paragraphs.append(
            f"For {LABELS[detector] if detector == 'binocular_origin' else LABELS[detector].lower()}, clipping changes mean TPR at 20\\% replacement "
            f"from {100*row['random20_raw']:.1f}\\% to {100*row['random20_clipped']:.1f}\\% under random replacement "
            f"and from {100*row['tail20_raw']:.1f}\\% to {100*row['tail20_clipped']:.1f}\\% under tail replacement. "
            f"Its clean TPR changes from {100*row['clean_raw']:.1f}\\% to {100*row['clean_clipped']:.1f}\\%."
        )
    paragraphs.append("These equal-cell averages summarize one contamination level, not the full dose-response curve. "
                      "Separate calibration does not ensure identical achieved test FPRs; per-cell operating points "
                      "and the full-range area summary are reported in Appendix~\\ref{app:primary-data}.")
    write_text(generated / "primary_main_interpretation.tex", " ".join(paragraphs))
    write_text(generated / "primary_cell_interpretation.tex", (
        "Figure~\\ref{fig:primary-binoculars} shows that, although detection performance varies "
        "across datasets and generators, the robustness benefit of clipping is consistent across "
        "all nine settings: the area under the TPR--contamination curve increases under both "
        "random and tail replacement. Clipping limits extreme token contributions and generally "
        "slows the loss of detection power as contamination increases. This consistency concerns "
        "overall robustness, not an improvement at every individual contamination rate. "
        f"The mean achieved test FPR remains below 5\\% for both raw ({100*orig['held_out_fpr_raw']:.2f}\\%) "
        f"and clipped ({100*orig['held_out_fpr_clipped']:.2f}\\%) scores. "
        "This is an average across settings, not a per-setting guarantee; detailed operating "
        "points are reported in Appendix~\\ref{app:primary-data}."
    ))
    write_text(generated / "primary_setting_limitations.tex", (
        "For Qwen--SQuAD, Binoculars' achieved test FPR changes from "
        f"{100*qwen_squad_origin['held_out_fpr_raw']:.2f}\\% to {100*qwen_squad_origin['held_out_fpr_clipped']:.2f}\\%. "
        "That cell therefore cannot establish a matched-FPR gain by itself. "
        "SQuAD also exposes a limitation of the target-model heuristics: likelihood, "
        "rank, log rank, and LRR often have near-zero contaminated TPR. The corrected "
        "Qwen--SQuAD LRR keeps the published direction and a positive denominator cap; "
        "its low detection rate is retained, rather than removed by choosing a "
        "test-dependent direction. Clipping does not rescue every underlying detector."
    ))

    # A full cell-level table at each operating target. Longtable can span pages.
    for target in (.01, .05):
        suffix = "1fpr" if target == .01 else "5fpr"
        lines = [r"\begin{longtable}{llrrrrrr}", r"\toprule", r"Cell & Detector & \multicolumn{2}{c}{Clean TPR} & \multicolumn{2}{c}{Actual FPR} & \multicolumn{2}{c}{Area change} \\", r"& & Raw & Clip & Raw & Clip & Random & Tail \\", r"\midrule\endfirsthead", r"\toprule", r"Cell & Detector & \multicolumn{2}{c}{Clean TPR} & \multicolumn{2}{c}{Actual FPR} & \multicolumn{2}{c}{Area change} \\", r"& & Raw & Clip & Raw & Clip & Random & Tail \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
        for dataset in DATASETS:
            for model in MODELS:
                rows = [r for r in cell_results if r["dataset"] == dataset and r["model"] == model and r["target_fpr"] == target]
                for i, row in enumerate(rows):
                    label = DATASET_LABELS[dataset] + " / " + MODEL_LABELS[model] if i == 0 else ""
                    label_detector = LABELS[row["detector"]]
                    vals = [f"{100*row[k]:.1f}" for k in ("clean_tpr_raw", "clean_tpr_clipped", "held_out_fpr_raw", "held_out_fpr_clipped")]
                    vals += [f"{100*row[f'{mode}_robustness_difference']:+.1f}" for mode in ("random", "tail")]
                    lines.append(label + " & " + label_detector + " & " + " & ".join(vals) + r" \\")
                if (dataset, model) != (DATASETS[-1], MODELS[-1]):
                    lines.append(r"\addlinespace[3pt]")
        lines.append(r"\end{longtable}")
        write_text(generated / f"primary_cells_{suffix}.tex", "\n".join(lines))

    # Saved parameters, deduplicated across operating targets.
    lines = [r"\begin{longtable}{lllrl}", r"\toprule", r"Dataset & Model & Detector & $d$ & Selected bound(s) \\", r"\midrule\endfirsthead", r"\toprule", r"Dataset & Model & Detector & $d$ & Selected bound(s) \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    for row in (r for r in cell_results if r["target_fpr"] == .05):
        spec = json.loads(row["clipping_specification"])
        if not spec:
            spec_text = "None"
        elif row["detector"] == "lrr":
            spec_text = f"$U_{{-\\log p}}={spec['nll_upper']:.4g},\\ U_{{\\log r}}={spec['log_rank_upper']:.4g}$"
        else:
            cap = next(iter(spec.values()))
            spec_text = ("$U=" if row["detector"] == "binocular_origin" else "$L=") + f"{cap:.4g}$"
        lines.append(" & ".join([DATASET_LABELS[row["dataset"]], MODEL_LABELS[row["model"]], LABELS[row["detector"]], str(row["direction"]), spec_text]) + r" \\")
    lines.append(r"\end{longtable}")
    write_text(generated / "primary_selected_bounds.tex", "\n".join(lines))

    auroc_summary = []
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"& \multicolumn{2}{c}{Clean AUROC} & \multicolumn{2}{c}{Random AUROC} & \multicolumn{2}{c}{Tail AUROC} \\", r"Detector & Raw & Clip & Raw & Clip & Raw & Clip \\", r"\midrule"]
    for detector in DETECTORS:
        dr = [r for r in cell_results if r["detector"] == detector and r["target_fpr"] == .05]
        row = {"detector": detector}
        for field in ("clean_auroc_raw", "clean_auroc_clipped", "random_attacked_mean_auroc_raw", "random_attacked_mean_auroc_clipped", "tail_attacked_mean_auroc_raw", "tail_attacked_mean_auroc_clipped"):
            row[field] = mean(r[field] for r in dr)
        auroc_summary.append(row)
        lines.append(LABELS[detector] + " & " + " & ".join(f"{100*v:.1f}" for k,v in row.items() if k != "detector") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_csv(generated / "primary_auroc_summary.csv", auroc_summary)
    write_text(generated / "primary_auroc_summary.tex", "\n".join(lines))

    partial_summary = []
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"& \multicolumn{2}{c}{Clean pAUROC} & \multicolumn{2}{c}{Random pAUROC} & \multicolumn{2}{c}{Tail pAUROC} \\", r"Detector & Raw & Clip & Raw & Clip & Raw & Clip \\", r"\midrule"]
    for detector in DETECTORS:
        dr = [r for r in cell_results if r["detector"] == detector and r["target_fpr"] == .05]
        row = {"detector": detector}
        for field in ("clean_partial_auroc_raw", "clean_partial_auroc_clipped", "random_attacked_mean_partial_auroc_raw", "random_attacked_mean_partial_auroc_clipped", "tail_attacked_mean_partial_auroc_raw", "tail_attacked_mean_partial_auroc_clipped"):
            row[field] = mean(r[field] for r in dr)
        partial_summary.append(row)
        lines.append(LABELS[detector] + " & " + " & ".join(f"{100*v:.1f}" for k,v in row.items() if k != "detector") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_csv(generated / "primary_partial_auroc_summary.csv", partial_summary)
    write_text(generated / "primary_partial_auroc_summary.tex", "\n".join(lines))

    def percent_interval(row, estimate, low, high, signed=False):
        val, lo, hi = (100*num(row, col) for col in (estimate, low, high))
        assert lo <= hi, (low, high)
        return f"{val:+.1f} ({lo:+.1f}, {hi:+.1f})" if signed else f"{val:.1f} ({lo:.1f}, {hi:.1f})"

    end_rows = []
    ci_lines = [r"\begin{longtable}{lrrr}", r"\toprule", r"Detector & Raw FPR (95\% CI) & Clip FPR (95\% CI) & Clean $\GPU host$TPR (95\% CI) \\", r"\midrule\endfirsthead", r"\toprule", r"Detector & Raw FPR (95\% CI) & Clip FPR (95\% CI) & Clean $\GPU host$TPR (95\% CI) \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    end_lines = [r"\begin{longtable}{lrr}", r"\toprule", r"Detector & Random $\GPU host$TPR (95\% CI) & Tail $\GPU host$TPR (95\% CI) \\", r"\midrule\endfirsthead", r"\toprule", r"Detector & Random $\GPU host$TPR (95\% CI) & Tail $\GPU host$TPR (95\% CI) \\", r"\midrule\endhead", r"\bottomrule\endfoot"]
    for dataset in DATASETS:
        for model in MODELS:
            group_label = DATASET_LABELS[dataset]+" / "+MODEL_LABELS[model]
            ci_lines.append(r"\multicolumn{4}{l}{\textbf{" + group_label + r"}} \\*")
            end_lines.append(r"\multicolumn{3}{l}{\textbf{" + group_label + r"}} \\*")
            for i, detector in enumerate(DETECTORS):
                label = LABELS[detector]
                raw = records[(dataset, model, detector, "random", "raw", .05, 0)]
                clipped = records[(dataset, model, detector, "random", "clipped", .05, 0)]
                vals = [percent_interval(row, "actual_fpr", "actual_fpr_ci_low", "actual_fpr_ci_high") for row in (raw,clipped)]
                vals.append(percent_interval(clipped, "clipped_minus_raw_tpr", "clipped_minus_raw_tpr_ci_low", "clipped_minus_raw_tpr_ci_high", True))
                row_end = r" \\*" if i < len(DETECTORS)-1 else r" \\"
                ci_lines.append(" & ".join([label, *vals]) + row_end)
                vals = []
                for mode in ("random", "tail"):
                    row = records[(dataset, model, detector, mode, "clipped", .05, .5)]
                    vals.append(percent_interval(row, "clipped_minus_raw_tpr", "clipped_minus_raw_tpr_ci_low", "clipped_minus_raw_tpr_ci_high", True))
                    end_rows.append({"dataset":dataset,"model":model,"detector":detector,"contamination_mode":mode,"requested_contamination_ratio":.5,"target_fpr":.05,"paired_tpr_difference":num(row,"clipped_minus_raw_tpr"),"paired_tpr_difference_ci_low":num(row,"clipped_minus_raw_tpr_ci_low"),"paired_tpr_difference_ci_high":num(row,"clipped_minus_raw_tpr_ci_high")})
                end_lines.append(" & ".join([label,*vals]) + row_end)
            if (dataset, model) != (DATASETS[-1], MODELS[-1]):
                ci_lines.append(r"\addlinespace[3pt]")
                end_lines.append(r"\addlinespace[3pt]")
    ci_lines.append(r"\end{longtable}")
    end_lines.append(r"\end{longtable}")
    write_text(generated / "primary_calibration_intervals.tex", "\n".join(ci_lines))
    write_text(generated / "primary_endpoint_effects.tex", "\n".join(end_lines))
    write_csv(generated / "primary_endpoint_effects.csv", end_rows)

    target_path = figures / "primary_binoculars_grid"
    c = canvas.Canvas(str(target_path.with_suffix(".pdf")), pagesize=(468, 266.4), invariant=1)
    figure_legend(c, 257)
    for i, dataset in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            x, y, w, h = 57 + 137 * j, 31 + 70 * (2-i), 125, 56
            draw_curves(c, records, dataset, model, "binocular_origin", (x,y,w,h), bands=False, xlabels=i==2, ylabels=j==0)
            c.setFillColor(Color(.1,.1,.1))
            if i == 0:
                c.setFont("Helvetica-Bold", 8)
                c.drawCentredString(x+w/2, y+h+8, MODEL_LABELS[model])
            if j == 0:
                c.saveState()
                c.translate(12, y+h/2)
                c.rotate(90)
                c.setFont("Helvetica", 7)
                c.drawCentredString(0, 0, DATASET_LABELS[dataset])
                c.restoreState()
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(259, 5, "Requested replacement (%)")
    c.saveState()
    c.translate(31, 136)
    c.rotate(90)
    c.drawCentredString(0, 0, "TPR")
    c.restoreState()
    finish_figure(c, target_path)

    # Complete curve atlas: one readable page per dataset and operating target.
    for dataset in DATASETS:
        for target in (.01, .05):
            target_path = figures / f"primary_all_detectors_{dataset}_{int(100*target)}fpr"
            c = canvas.Canvas(str(target_path.with_suffix(".pdf")), pagesize=(468, 580), invariant=1)
            c.setFont("Helvetica-Bold", 10)
            c.drawCentredString(234, 568, f"{DATASET_LABELS[dataset]}: TPR at {int(100*target)}% target FPR")
            figure_legend(c, 552)
            for i, detector in enumerate(DETECTORS):
                for j, model in enumerate(MODELS):
                    x, y, w, h = 64 + j * 135, 30+(6-i)*70, 120, 56
                    draw_curves(c, records, dataset, model, detector, (x,y,w,h), target, bands=True, xlabels=i==6, ylabels=j==0)
                    c.setFillColor(Color(.1,.1,.1))
                    if i == 0:
                        c.setFont("Helvetica-Bold", 8)
                        c.drawCentredString(x+w/2, y+h+9, MODEL_LABELS[model])
                    if j == 0:
                        label = LABELS[detector].replace("Log likelihood", "Log\nlikelihood")
                        c.saveState()
                        c.translate(14, y+h/2)
                        c.rotate(90)
                        c.setFont("Helvetica", 7)
                        for k, text in enumerate(label.splitlines()):
                            c.drawCentredString(0, -k*8, text)
                        c.restoreState()
            c.setFont("Helvetica", 7.5)
            c.drawCentredString(259, 5, "Requested replacement (%)")
            finish_figure(c, target_path)

    headlines = {"source_bundle": args.bundle.name, "scope": "controlled contamination study across nine dataset-generator combinations; seven publication detectors", "source_counts": {p.name: len(r) for p, r in zip(source_paths, [metrics, robustness, inventory])}, "aggregation": "Equal arithmetic mean of nine dataset-generator point estimates; no aggregate confidence interval calculated.", "units": "Stored values are proportions; LaTeX tables display percentage-point scale.", "checks": {"unique_metric_keys": len(records), "all_robustness_areas_match_independent_trapezoid": True, "bootstrap_repetitions": 2000}, "summary": summaries, "source_hashes": {p.relative_to(args.bundle).as_posix(): {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths}}
    origin_cells = [r for r in cell_results if r["detector"] == "binocular_origin" and r["target_fpr"] == .05]
    headlines["binoculars_5fpr_cells"] = origin_cells
    headlines["held_out_fpr_ranges_5fpr"] = {agg: [min(r[f"held_out_fpr_{agg}"] for r in cell_results if r["target_fpr"] == .05), max(r[f"held_out_fpr_{agg}"] for r in cell_results if r["target_fpr"] == .05)] for agg in ("raw", "clipped")}
    write_text(generated / "primary_findings.json", json.dumps(headlines, indent=2))
    print(json.dumps({"status": "pass", "metric_rows": len(metrics), "cell_summary_rows": len(cell_results), "figure_pairs": 7, "summary_5fpr": [r for r in summaries if r["target_fpr"] == .05]}, indent=2))
