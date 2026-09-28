"""Validate the saved four-scope evidence against the submitted paper's tables."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

METHODS = ('log_likelihood', 'rank', 'log_rank', 'lrr', 'entropy', 'entropy_gap', 'binocular_origin')
RATES = ('(0,0.05]', '(0.05,0.1]', '(0.1,0.2]', '(0.2,0.5]')
# Table 3, then Table 4: displayed (raw TPR %, gain in percentage points).
TABLE_3 = {
    'log_likelihood': ((49.17,.32),(55.62,0),(30.47,0)),
    'rank': ((29.48,25.23),(35.05,26.40),(24.87,15.96)),
    'log_rank': ((53.24,.74),(59.38,.68),(35.79,.28)),
    'lrr': ((59.51,2.28),(64.48,4.21),(47.60,.73)),
    'entropy': ((26.11,.09),(35.07,.43),(18.50,.01)),
    'entropy_gap': ((66.55,1.60),(71.64,1.13),(46.77,1.02)),
    'binocular_origin': ((74.27,.22),(82.79,.28),(60.88,.87)),
}
TABLE_4 = {
    'log_likelihood': ((44.96,1.15),(27.91,0),(59.22,0)),
    'rank': ((22.02,31.37),(27.89,7.79),(39.50,23.83)),
    'log_rank': ((50.15,2.21),(31.87,-.10),(62.11,.19)),
    'lrr': ((59.16,3.52),(38.80,9.30),(65.04,4.87)),
    'entropy': ((24.20,.43),(11.87,.10),(33.52,.54)),
    'entropy_gap': ((65.43,3.42),(75.78,0),(72.54,.85)),
    'binocular_origin': ((74.07,.43),(80.83,0),(78.71,.31)),
}

def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def require(value, message):
    if not value: raise ValueError(message)

def validate(root):
    root = Path(root)
    result = rows(root / 'results.csv')
    specs = json.loads((root / 'selected_specs.json').read_text())['specifications']
    sources = rows(root / 'selected_sources.csv')
    manifest = json.loads((root / 'manifest.json').read_text())
    require(len(result)==462 and len(specs)==119, 'Paper expects 462 results / 119 fits')
    require(manifest['selector']=='crossfit_tpr' and manifest['bootstrap_repetitions']==2000,
            'Wrong paper selector/bootstrap configuration')
    require(set(r['detector'] for r in result)==set(METHODS), 'Wrong detector set')
    require(Counter(s['scope'] for s in specs)=={'attack_specific':77,'rate_specific':28,
            'full_universal':7,'eligible_universal':7}, 'Wrong four-scope fit counts')
    look = {(r['detector'],r['scope'],r['group'],r['condition']):r for r in result}
    require(len(look)==462,'Duplicate result key')
    for s in specs:
        candidates = [c for c in s['candidate_diagnostics'] if 'rejected' not in c]
        # The recorded objective must equal the fixed 0.8/0.2 weighting and best candidate.
        expected = .8*s['cv_attacked_tpr']+.2*s['cv_clean_tpr']
        require(abs(s['selection_objective']-expected)<1e-12,'Objective weighting mismatch')
        require(abs(max(c['objective'] for c in candidates)-s['selection_objective'])<=1e-12,
                'Selected candidate is not optimal under recorded objective')
        require(any(c['specification']==s['specification'] for c in candidates),'Missing selected candidate')
    for r in result:
        require(float(r['target_fpr'])==.05 and int(r['n_test_human'])==5149,'Wrong evaluation population')
        require(abs(float(r['clipped_tpr'])-float(r['raw_tpr'])-float(r['paired_tpr_difference']))<1e-12,
                'TPR difference mismatch')
        for k,v in r.items():
            if k.endswith(('_ci_low','_ci_high')):require(math.isfinite(float(v)), 'Nonfinite interval')
    for scope, groups, expected in [
        ('rate_specific',RATES[:3],TABLE_3),
        ('attack_specific',('upper_lower','paraphrase','alternative_spelling'),TABLE_4)]:
        for d, values in expected.items():
            for group, (raw,gain) in zip(groups,values):
                r=look[d,scope,group,'attacked']
                require(abs(100*float(r['raw_tpr'])-raw)<.0051,f'Paper raw mismatch: {d}/{group}')
                require(abs(100*float(r['paired_tpr_difference'])-gain)<.0051,f'Paper gain mismatch: {d}/{group}')
    require(len(sources)==12871 and len({r['source_id'] for r in sources})==12871,'Source identities mismatch')
    require(Counter(r['split'] for r in sources)=={'clipping_tuning':5148,'calibration':2574,'test':5149},
            'Source split sizes mismatch')
    require(set(manifest['crossfit']['source_folds'])=={r['source_id'] for r in sources if r['split']=='clipping_tuning'},
            'Cross-fitting sources do not match source manifest')
    calibration=rows(root/'calibration.csv')
    require(len(calibration)==952, 'Expected eight calibration domains per fit')
    cal_lookup={(r['detector'],r['clipping_mode'],r['group'],r['domain']):r for r in calibration}
    require(len(cal_lookup)==952,'Duplicate calibration key')
    for s in specs:
        for domain,thresholds in s['thresholds'].items():
            row=cal_lookup[s['detector'],s['scope'],s['group'],domain]
            for kind in ['raw','clipped']:
                require(float(row[kind+'_threshold'])==thresholds[kind],'Saved calibration threshold mismatch')
    return {'fits':119,'results':462,'paper_tables_3_and_4':'matched','sources':12871}
