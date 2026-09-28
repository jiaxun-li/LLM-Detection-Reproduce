"""Analyze explicit RAID score packs with the current four-scope TPR protocol."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RAID import metrics as base
from RAID.score_io import compact_rows, identifier
from RAID.evaluation import METHODS, PROTOCOL, fit_and_evaluate
from RAID.raid_data import RAID_ADVERSARIAL_ATTACKS
from experiment_core.infrastructure.io import atomic_write_json
from experiment_core.infrastructure.revision_artifacts import artifact_digest


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def attach_rates(rows, records):
    cached = {(str(r['source_id']), r['split'], r['attack']): r for r in records}
    expected = {(base.source(r), base.split(r), base.condition(r)) for r in rows if not base.human(r)}
    if len(cached) != len(records) or set(cached) != expected:
        raise ValueError('Contamination cache does not match score families')
    for r in rows:
        if not base.human(r):
            value = float(cached[(base.source(r), base.split(r), base.condition(r))]['realized_contamination_rate'])
            if not 0 <= value <= 1:
                raise ValueError('Invalid contamination rate')
            r['contamination_rate'] = value


def verify_baselines(rows, frozen, accepted, rates=None):
    """Replay every accepted raw/full/eligible attack point, not the old oracle."""
    humans = base._rows(rows, 'test', True)
    machine = base._rows(rows, 'test', False)
    checked = []
    for d in METHODS:
        f = frozen['detectors'][d]
        for scope, prefix in [('full_universal', ''), ('eligible_universal', 'eligible_universal_')]:
            spec = f[prefix + 'clipping_specification']
            thresholds = f[prefix + 'classification_thresholds']
            raw, clipped = base._cache(humans + machine, d, f['direction'], spec)
            expected = [r for r in accepted if r['detector'] == d]
            if len(expected) != 12 or len({r['condition'] for r in expected}) != 12:
                raise ValueError('Expected 12 accepted attack conditions')
            for record in expected:
                mr = [r for r in machine if base.condition(r) == record['condition']]
                point = base._point(humans, mr, raw, clipped, thresholds)
                for name, value in point.items():
                    key = name if name.startswith('raw_') else prefix + name
                    if not math.isclose(value, float(record[key]), rel_tol=0, abs_tol=1e-10):
                        raise ValueError(f'Baseline mismatch: {d}/{scope}/{record["condition"]}/{key}')
                checked.append({'detector': d, 'scope': scope, 'condition': record['condition'], **point})
            if rates is not None:
                rate_rows = [r for r in rates if r['detector'] == d]
                if len(rate_rows) != 4:
                    raise ValueError('Expected four accepted rate bins')
                bounds = (0, .05, .1, .2, .5)
                for record in rate_rows:
                    index = int(record['contamination_bin_index'])
                    mr = [r for r in machine if bounds[index-1] < base.rho(r) <= bounds[index]]
                    point = base._point(humans, mr, raw, clipped, thresholds)
                    for name, value in point.items():
                        key = name if name.startswith('raw_') else prefix + name
                        if not math.isclose(value, float(record[key]), rel_tol=0, abs_tol=1e-10):
                            raise ValueError(f'Rate baseline mismatch: {d}/{scope}/{index}/{key}')
                    checked.append({'detector': d, 'scope': scope,
                                    'condition': record['contamination_bin'], **point})
    return checked



def load_score_rows(falcon_scores, origin_scores, binoculars_scores=None, contamination_records=None):
    """Merge explicit score packs and attach tokenizer-aligned contamination rates."""
    keep_tokens = contamination_records is None
    rows = list(compact_rows(Path(falcon_scores), include_token_ids=keep_tokens))
    if binoculars_scores is not None:
        rows = base.merge_score_rows(rows, list(compact_rows(Path(binoculars_scores))))
    origin = []
    for path in origin_scores:
        origin.extend(compact_rows(Path(path)))
    if not origin:
        raise ValueError('Official origin score pack(s) are required')
    rows = base.merge_score_rows(rows, origin)
    if contamination_records is not None:
        attach_rates(rows, read_csv(Path(contamination_records)))
    else:
        base.attach_contamination_rates(rows)
    return rows


def analyze_rows(rows, output, run_id, repetitions=2000, seed=481516,
                 inputs=None, frozen=None, baseline_directory=None, input_fingerprints=None):
    """Fit/evaluate fresh or saved score rows; historical baseline audit is optional.

    inputs maps public logical labels to source paths for pre/post SHA checks.
    Directions are always learned on clean tuning; an optional frozen direction
    record must agree. Rates may be precomputed with the scored-token alignment
    protocol, or computed here from scored (and, when available, full) token IDs.
    """
    from RAID import crossfit
    run_id = identifier(run_id)
    if repetitions < 1:
        raise ValueError('At least one bootstrap repetition is required')
    output = Path(output).resolve()
    output_root = (ROOT / 'outputs').resolve()
    if output == output_root or not output.is_relative_to(output_root):
        raise ValueError('Analysis output must be a new directory below outputs/')
    if output.exists():
        raise ValueError('Output already exists; choose a new output directory')
    paths = {str(label): Path(path).resolve() for label, path in (inputs or {}).items()}
    if baseline_directory is not None:
        baseline_directory = Path(baseline_directory).resolve()
        for name in ('frozen_specs.json', 'attack_summary.csv', 'contamination_summary.csv'):
            paths['baseline/' + name] = baseline_directory / name
    if any(path.is_relative_to(output) for path in paths.values()):
        raise ValueError('Output overlaps an input path')
    fingerprints = {label: artifact_digest(path) for label, path in paths.items()}
    if input_fingerprints is not None and any(fingerprints.get(label) != digest
                                            for label, digest in input_fingerprints.items()):
        raise ValueError('Input changed since score rows were loaded')
    _, counts = base._validate(rows, expected=RAID_ADVERSARIAL_ATTACKS)
    source_domains = {}
    for row in rows:
        source_id, domain = base.source(row), base.domain(row)
        if source_id in source_domains and source_domains[source_id] != domain:
            raise ValueError('Source family contains mixed domains')
        source_domains[source_id] = domain
    machine = [r for r in rows if not base.human(r)]
    if any('contamination_rate' not in r for r in machine):
        base.attach_contamination_rates(rows)
    if any(not math.isfinite(base.rho(r)) or not 0 <= base.rho(r) <= 1 for r in machine):
        raise ValueError('Invalid contamination rate')
    humans = base._rows(rows, 'clipping_tuning', True)
    clean = base._rows(rows, 'clipping_tuning', False, 'none')
    learned = {'detectors': {d: {'direction': base.learn_direction(d, humans, clean)} for d in METHODS}}
    if frozen is not None:
        for detector in METHODS:
            if frozen['detectors'][detector]['direction'] != learned['detectors'][detector]['direction']:
                raise ValueError('Supplied direction differs from clean tuning: ' + detector)
    frozen = learned
    baseline = None
    if baseline_directory is not None:
        baseline = verify_baselines(rows, read_json(baseline_directory / 'frozen_specs.json'),
            read_csv(baseline_directory / 'attack_summary.csv'),
            read_csv(baseline_directory / 'contamination_summary.csv'))
    fold_plan = crossfit.make_folds(humans)
    provenance = {'protocol': PROTOCOL, 'run_id': run_id, 'selector': 'crossfit_tpr',
        'crossfit': fold_plan, 'include_universal': True,
        'attacked_weighting': 'pooled documents within fitting scope',
        'inputs': fingerprints, 'population': counts,
        'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
        'detectors': list(METHODS), 'directions': learned,
        'objective': '0.8 out-of-fold pooled attacked TPR + 0.2 out-of-fold clean TPR',
        'clean_loss_constraint': None, 'rate_bins': [0, .05, .1, .2, .5],
        'baseline_parity': 'not_requested' if baseline is None else 'pass',
        'code': {str(path.relative_to(ROOT)): artifact_digest(path) for path in
                 [Path(__file__).resolve(), ROOT/'RAID/evaluation.py', ROOT/'RAID/crossfit.py',
                  ROOT/'RAID/metrics.py', ROOT/'RAID/score_io.py',
                  ROOT/'experiment_core/detectors/detector_revision.py',
                  ROOT/'experiment_core/analysis/evaluation.py']}}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'manifest.json', {**provenance, 'status': 'running'})
    specs, results, calibration = fit_and_evaluate(rows, frozen, repetitions, seed, fold_plan=fold_plan)
    if len(specs) != 119 or len(results) != 462:
        raise ValueError('Four-scope result count mismatch')
    atomic_write_json(output / 'selected_specs.json', {'protocol': PROTOCOL, 'specifications': specs})
    base._csv(output / 'results.csv', results)
    base._csv(output / 'calibration.csv', calibration)
    names = ['manifest.json', 'selected_specs.json', 'results.csv', 'calibration.csv']
    if baseline is not None:
        base._csv(output / 'baseline_parity.csv', baseline)
        names.append('baseline_parity.csv')
    for label, path in paths.items():
        if artifact_digest(path) != fingerprints[label]:
            raise ValueError('Input changed during evaluation: ' + label)
    atomic_write_json(output / 'manifest.json', {**provenance, 'status': 'complete'})
    artifacts = {name: artifact_digest(output / name) for name in names}
    atomic_write_json(output / 'focused.complete.json', {'status': 'complete', 'artifacts': artifacts,
        'specifications': len(specs), 'result_rows': len(results),
        'baseline_parity': provenance['baseline_parity']})
    return {'status': 'complete', 'output': str(output), 'specifications': len(specs), 'result_rows': len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--falcon-scores', type=Path, required=True)
    parser.add_argument('--origin-scores', type=Path, action='append', required=True,
                        help='Official output-only origin scores; repeat for disjoint shards')
    parser.add_argument('--binoculars-scores', type=Path,
                        help='Optional legacy pair score pack, for archived data layouts')
    parser.add_argument('--contamination-records', type=Path,
                        help='Optional alignment-rate CSV; otherwise compute from scored token IDs')
    parser.add_argument('--baseline-directory', type=Path,
                        help='Optional historical frozen_specs and attack/rate summaries for parity audit')
    parser.add_argument('--run-id', type=identifier, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=481516)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1:
        parser.error('At least one bootstrap repetition is required')
    paths = {'falcon_scores': args.falcon_scores,
             **{f'origin_scores/{i}': path for i, path in enumerate(args.origin_scores)}}
    for label in ('binoculars_scores', 'contamination_records'):
        if getattr(args, label) is not None:
            paths[label] = getattr(args, label)
    before = {label: artifact_digest(path) for label, path in paths.items()}
    rows = load_score_rows(args.falcon_scores, args.origin_scores, args.binoculars_scores,
                           args.contamination_records)
    for label, path in paths.items():
        if artifact_digest(path) != before[label]:
            raise ValueError('Input changed while loading: ' + label)
    print(json.dumps(analyze_rows(rows, args.output, args.run_id,
        repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
        inputs=paths, baseline_directory=args.baseline_directory, input_fingerprints=before), indent=2))


if __name__ == '__main__':
    main()
