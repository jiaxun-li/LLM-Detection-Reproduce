"""Current four-scope RAID evaluation with cross-fitted TPR selection."""
from collections import Counter
import numpy as np
from RAID import metrics as base

METHODS = ('log_likelihood', 'rank', 'log_rank', 'lrr', 'entropy',
           'entropy_gap', 'binocular_origin')
PROTOCOL = 'raid-focused-crossfit-tpr-four-scope-v2'


def paired_fpr_interval(humans, raw, clipped, thresholds, repetitions, seed):
    """Pair each human's decisions and stratify resampling by domain."""
    humans = sorted(humans, key=base.source)
    domains = sorted({base.domain(r) for r in humans})
    differences = []
    for domain in domains:
        differences.append(np.array([
            int(clipped[id(r)] >= thresholds[domain]['clipped']) -
            int(raw[id(r)] >= thresholds[domain]['raw'])
            for r in humans if base.domain(r) == domain]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        total = 0
        for values in differences:
            weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)))
            total += np.dot(weights, values)
        samples.append(total / len(humans))
    return base.percentile_interval(samples)


def fit_and_evaluate(rows, frozen, repetitions=2000, seed=481516, min_rate_rows=250,
                     fold_plan=None):
    humans = base._rows(rows, 'clipping_tuning', True)
    clean = base._rows(rows, 'clipping_tuning', False, 'none')
    calibration = base._rows(rows, 'calibration', True)
    test_h = base._rows(rows, 'test', True)
    test_clean = base._rows(rows, 'test', False, 'none')
    from RAID import crossfit
    fold_plan = fold_plan if fold_plan is not None else crossfit.make_folds(humans)
    attacks = sorted({base.condition(r) for r in rows if not base.human(r)} - {'none'})
    rate_groups = [(f'({lo:g},{hi:g}]', lambda r, lo=lo, hi=hi: lo < base.rho(r) <= hi)
                   for lo, hi in zip((0, .05, .1, .2), (.05, .1, .2, .5))]
    scopes = [
            ('attack_specific', [(a, lambda r, a=a: base.condition(r) == a) for a in attacks]),
            ('rate_specific', rate_groups),
    ]
    scopes.extend([
        ('full_universal', [('all_attacks', lambda r: True)]),
        ('eligible_universal', [('eligible_attacks', lambda r: 0 < base.rho(r) <= .5)]),
    ])
    specifications, results, calibration_rows = [], [], []
    for detector in METHODS:
        direction = frozen['detectors'][detector]['direction']
        for scope, groups in scopes:
            for group, belongs in groups:
                train = [r for r in rows if base.split(r) == 'clipping_tuning'
                         and not base.human(r) and base.condition(r) != 'none' and belongs(r)]
                test = [r for r in rows if base.split(r) == 'test'
                        and not base.human(r) and base.condition(r) != 'none' and belongs(r)]
                if not train or not test or (scope == 'rate_specific' and len(train) < min_rate_rows):
                    raise ValueError(f'Insufficient data for {detector}/{scope}/{group}')
                spec, diagnostics = crossfit.select_specification(
                    detector, direction, humans, clean, train, fold_plan)
                selected = next(r for r in diagnostics if r['specification'] == spec)
                thresholds, cr = base._thresholds(calibration, detector, direction, spec, scope)
                for row in cr:
                    row['group'] = group
                calibration_rows.extend(cr)
                specifications.append({'detector': detector, 'scope': scope, 'group': group,
                    'selector': 'crossfit_tpr',
                    'direction': direction, 'specification': spec, 'thresholds': thresholds,
                    'selection_objective': selected['objective'],
                    'selection_pooled_attack_auroc': selected['pooled_attack_auroc'],
                    'selection_clean_auroc': selected['clean_auroc'],
                    'n_tuning_machine': len(train), 'attack_counts': dict(Counter(map(base.condition, train))),
                    'candidate_diagnostics': diagnostics})
                specifications[-1].update({k: selected[k] for k in
                    ('cv_attacked_tpr', 'cv_clean_tpr', 'cv_human_fpr')})
                comparisons = [('attacked', test), ('clean', test_clean)]
                if scope in ('full_universal', 'eligible_universal'):
                    # The fitting population differs; the evaluation populations do not.
                    all_test = [r for r in rows if base.split(r) == 'test'
                                and not base.human(r) and base.condition(r) != 'none']
                    comparisons = [
                        ('clean', test_clean), ('all_attacks', all_test),
                        ('eligible_attacks', [r for r in all_test if 0 < base.rho(r) <= .5]),
                        *[('attack:' + a, [r for r in all_test if base.condition(r) == a])
                          for a in attacks],
                        *[('rate:' + label, [r for r in all_test if predicate(r)])
                          for label, predicate in rate_groups],
                    ]
                if any(not mr for _, mr in comparisons):
                    raise ValueError('Empty test comparison')
                score_rows = {id(r): r for r in test_h + [r for _, mr in comparisons for r in mr]}
                raw, clipped = base._cache(list(score_rows.values()), detector, direction, spec)
                for record in cr:
                    hs = [r for r in test_h if base.domain(r) == record['domain']]
                    record['raw_test_fpr'] = base.actual_fpr([raw[id(r)] for r in hs], record['raw_threshold'])
                    record['clipped_test_fpr'] = base.actual_fpr([clipped[id(r)] for r in hs], record['clipped_threshold'])
                for condition, mr in comparisons:
                    result = {'detector': detector, 'scope': scope, 'group': group,
                              'condition': condition, 'target_fpr': base.TARGET_FPR,
                              'n_test_machine': len(mr),
                              'n_test_human': len(test_h),
                              **base._point(test_h, mr, raw, clipped, thresholds)}
                    base._add_ci(result, base._bootstrap(test_h, mr, raw, clipped, thresholds,
                                 repetitions, base._seed(seed, detector, scope, group, condition)))
                    result['paired_fpr_difference'] = result['clipped_test_fpr'] - result['raw_test_fpr']
                    low, high = paired_fpr_interval(test_h, raw, clipped, thresholds, repetitions,
                                                   base._seed(seed, detector, scope, group, condition))
                    result['paired_fpr_difference_ci_low'] = low
                    result['paired_fpr_difference_ci_high'] = high
                    results.append(result)
                print(f'Completed {detector} {scope} {group}', flush=True)
    return specifications, results, calibration_rows
