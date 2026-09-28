"""Source-group, domain-stratified cross-fitted TPR bound selection.

Only threshold estimation is cross-fitted. Candidate bounds and directions
are fixed from the complete clean tuning set, as in the AUROC protocol.
The final calibration and test sets are never used here.
"""
from collections import Counter
import numpy as np
from RAID import metrics as base

FOLDS = 5
FOLD_SEED = 20260917


def make_folds(humans, folds=FOLDS, seed=FOLD_SEED):
    if folds < 2 or any(base.split(r) != 'clipping_tuning' for r in humans):
        raise ValueError('Cross-fitting requires tuning humans and at least two folds')
    ids = [base.source(r) for r in humans]
    if len(set(ids)) != len(ids):
        raise ValueError('Expected one tuning human per source')
    assignment, counts = {}, []
    for domain in sorted({base.domain(r) for r in humans}):
        members = sorted(base.source(r) for r in humans if base.domain(r) == domain)
        if len(members) < folds:
            raise ValueError(f'Domain {domain} has fewer sources than folds')
        rng = np.random.default_rng(base._seed(seed, domain))
        members = [members[i] for i in rng.permutation(len(members))]
        for i, sid in enumerate(members):
            assignment[sid] = i % folds
        for fold in range(folds):
            n_valid = sum(assignment[s] == fold for s in members)
            n_train = len(members) - n_valid
            counts.append({'domain': domain, 'fold': fold, 'threshold_humans': n_train,
                           'validation_humans': n_valid,
                           'nominal_allowed_exceedances': int(np.floor(.05 * n_train))})
    return {'folds': folds, 'seed': seed, 'source_folds': assignment, 'domain_counts': counts}


def crossfit_metrics(humans, clean, attacked, hs, cs, attacked_scores, plan):
    assignment = plan['source_folds']
    if any(base.split(r) != 'clipping_tuning' for r in humans + clean + attacked):
        raise ValueError('Cross-fitting may only use tuning rows')
    if len(humans) != len(hs) or len(clean) != len(cs) or len(attacked) != len(attacked_scores):
        raise ValueError('Score and row lengths differ')
    domains = {base.source(r): base.domain(r) for r in humans}
    if set(domains) != set(assignment) or len(domains) != len(humans):
        raise ValueError('Fold assignment does not match tuning humans')
    if Counter(map(base.source, clean)) != Counter(domains.keys()):
        raise ValueError('Expected one clean machine text per tuning source')
    for r in clean + attacked:
        if base.source(r) not in domains or base.domain(r) != domains[base.source(r)]:
            raise ValueError('Machine row has no matching tuning human/domain')
    totals = {'human': [0, 0], 'clean': [0, 0], 'attacked': [0, 0]}
    fold_records = []
    for fold in range(plan['folds']):
        thresholds = {}
        for domain in sorted(set(domains.values())):
            values = [score for r, score in zip(humans, hs)
                      if base.domain(r) == domain and assignment[base.source(r)] != fold]
            if not values:
                raise ValueError('Empty threshold-training domain')
            thresholds[domain] = base.calibration_threshold(values, base.TARGET_FPR)
        record = {'fold': fold, 'thresholds': thresholds}
        for name, rows, scores in [('human', humans, hs), ('clean', clean, cs),
                                   ('attacked', attacked, attacked_scores)]:
            decisions = [score >= thresholds[base.domain(r)] for r, score in zip(rows, scores)
                         if assignment[base.source(r)] == fold]
            count, hits = len(decisions), sum(decisions)
            totals[name][0] += hits
            totals[name][1] += count
            record[name + '_n'] = count
            record[name + '_hits'] = hits
            record[name + '_rate'] = hits / count if count else None
        fold_records.append(record)
    if any(n == 0 for _, n in totals.values()):
        raise ValueError('Cross-fit comparison is empty')
    rates = {name: hits/n for name, (hits, n) in totals.items()}
    return {'cv_attacked_tpr': rates['attacked'], 'cv_clean_tpr': rates['clean'],
            'cv_human_fpr': rates['human'], 'cv_folds': fold_records,
            'objective': .8 * rates['attacked'] + .2 * rates['clean']}


def select_specification(detector, direction, humans, clean, attacked, plan):
    if not humans or not clean or not attacked:
        raise ValueError('TPR selection requires nonempty tuning groups')
    if any(base.split(r) != 'clipping_tuning' for r in humans + clean + attacked):
        raise ValueError('Only tuning rows may select bounds')
    best, best_value, diagnostics = {}, -np.inf, []
    for spec in base.candidate_specifications(detector, direction, humans + clean):
        record = {'specification': spec}
        if detector == 'lrr' and not base.valid_lrr_clipping_spec(spec):
            diagnostics.append({**record, 'rejected': 'nonpositive_lrr_cap'})
            continue
        score = lambda r: base.oriented_document_score(r, detector, direction, spec)
        hs, cs = list(map(score, humans)), list(map(score, clean))
        if spec and (base.constant_clean_scores(hs, cs) or
                     base.structurally_constant_candidate(humans + clean, detector, direction, spec)):
            diagnostics.append({**record, 'rejected': 'constant_clean_scores'})
            continue
        ms = list(map(score, attacked))
        metrics = crossfit_metrics(humans, clean, attacked, hs, cs, ms, plan)
        diagnostics.append({**record, **metrics, 'clean_auroc': base.auroc(hs, cs),
                            'pooled_attack_auroc': base.auroc(hs, ms)})
        if metrics['objective'] > best_value + 1e-12:
            best, best_value = dict(spec), metrics['objective']
    return best, diagnostics
