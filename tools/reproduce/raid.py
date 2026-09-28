"""Paper RAID table builder; frozen cross-fitted TPR results, no inference."""
import csv
import hashlib
import json
import re
from pathlib import Path

from .verify import verify_inventory, new_output, absolute_path
SOURCE = None
OUT = None
METHODS = ['log_likelihood', 'rank', 'log_rank', 'lrr', 'entropy', 'entropy_gap', 'binocular_origin']
NAMES = dict(zip(METHODS, ['Log likelihood', 'Rank', 'Log rank', 'LRR', 'Entropy', 'Entropy gap', 'Binoculars']))
RATES = ['(0,0.05]', '(0.05,0.1]', '(0.1,0.2]', '(0.2,0.5]']
ATTACKS = ['alternative_spelling', 'article_deletion', 'homoglyph', 'insert_paragraphs', 'number', 'paraphrase', 'perplexity_misspelling', 'synonym', 'upper_lower', 'whitespace', 'zero_width_space']
ATTACK_NAMES = dict(zip(ATTACKS, ['Alternative spelling', 'Article deletion', 'Homoglyph', 'Paragraph insertion', 'Number change', 'Paraphrase', 'Misspelling', 'Synonym', 'Case change', 'Whitespace', 'Zero-width space']))


def pct(x, signed=False):
    value = 100 * float(x)
    if abs(value) < .005:
        value = 0.
    return format(value, '+.2f' if signed else '.2f')


def change(row, kind='tpr', stack=False):
    key = 'paired_' + kind + '_difference'
    delta = pct(row[key], True)
    interval = '[' + pct(row[key + '_ci_low']) + ', ' + pct(row[key + '_ci_high']) + ']'
    return r'\shortstack{' + delta + r'\\{}' + interval + '}' if stack else delta + ' ' + interval


def table(caption, label, heads, body, columns, long=False):
    if long:
        header = ' & '.join(heads) + r' \\'
        return '\n'.join([r'\begingroup\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{@{}' + columns + '@{}}',
            r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\toprule', header,
            r'\midrule\endfirsthead', r'\multicolumn{' + str(len(heads)) + r'}{l}{\textit{Continued}}\\',
            r'\toprule', header, r'\midrule\endhead', r'\bottomrule\endfoot', *body, r'\end{longtable}', r'\endgroup', ''])
    return '\n'.join([r'\begin{table}[htbp]\centering', r'\caption{' + caption + r'}\label{' + label + '}',
        r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}' + columns + '@{}}', r'\toprule', *heads,
        r'\midrule', *body, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def build(source, output, png=False):
    global SOURCE, OUT
    SOURCE = absolute_path(source)
    OUT = new_output(output, (SOURCE,))
    if not __debug__:
        raise RuntimeError("Do not use -O: table validation requires assertions")
    verify_inventory(SOURCE, "focused.complete.json", "artifacts", None)
    marker = json.loads((SOURCE / 'focused.complete.json').read_text())
    assert marker['status'] == 'complete' and marker['specifications'] == 119 and marker['result_rows'] == 462
    for name, record in marker['artifacts'].items():
        data = (SOURCE / name).read_bytes()
        assert len(data) == record['size'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
    with (SOURCE / 'results.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    specs = json.loads((SOURCE / 'selected_specs.json').read_text())['specifications']
    assert len(rows) == 462 and len(specs) == 119
    lookup = {(r['detector'], r['scope'], r['group'], r['condition']): r for r in rows}
    assert len(lookup) == 462
    def get(d, scope, group, condition='attacked'):
        return lookup[d, scope, group, condition]
    OUT.mkdir(parents=True, exist_ok=False)
    output = {}
    main_rates = RATES[:3]
    body = []
    for d in METHODS:
        cells = [NAMES[d]]
        for rate in main_rates:
            r = get(d, 'rate_specific', rate)
            cells += [pct(r['raw_tpr']), pct(r['paired_tpr_difference'], True)]
        body.append(' & '.join(cells) + r' \\[3pt]')
    output['main_rates.tex'] = table(
        r'Rate-specific RAID detection at target 5\% FPR. Raw TPR is a percentage; change is clipped minus raw in percentage points. The three intervals contain 19,894, 4,682, and 7,137 attacked test texts, respectively.',
        'tab:raid-rate-specific', [r'& \multicolumn{2}{c}{\(0<\rho\leq0.05\)} & \multicolumn{2}{c}{\(0.05<\rho\leq0.1\)} & \multicolumn{2}{c}{\(0.1<\rho\leq0.2\)} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}', r'Detector & Raw & Change & Raw & Change & Raw & Change \\'], body, 'lrrrrrr')
    body = []
    for d in METHODS:
        cells = [NAMES[d]]
        for attack in ['upper_lower', 'paraphrase', 'alternative_spelling']:
            r = get(d, 'attack_specific', attack)
            cells += [pct(r['raw_tpr']), change(r, stack=True)]
        body.append(' & '.join(cells) + r' \\[3pt]')
    output['main_attacks.tex'] = table(
        r'Attack-specific RAID detection at target 5\% FPR. Each attack has 5,149 test texts. Raw TPR is a percentage; change is clipped minus raw in percentage points, with paired 95\% confidence intervals.',
        'tab:raid-attack-specific', [r'& \multicolumn{2}{c}{Case change} & \multicolumn{2}{c}{Paraphrase} & \multicolumn{2}{c}{Alternative spelling} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}', r'Detector & Raw & Change [95\% CI] & Raw & Change [95\% CI] & Raw & Change [95\% CI] \\'], body, 'lrrrrrr')
    output['rates.tex'] = ''
    for i, rate in enumerate(RATES):
        body = []
        for d in METHODS:
            r = get(d, 'rate_specific', rate)
            body.append(' & '.join([NAMES[d], pct(r['raw_tpr']), pct(r['clipped_tpr']), change(r)]) + r' \\')
        output['rates.tex'] += table(r'Rate-specific detection for \(\rho\in' + rate + r'\), with ' + str(r['n_test_machine']) + ' attacked test texts.',
            'tab:raid-rate-' + str(i+1), [r'Detector & Raw TPR & Clipped TPR & Change [95\% CI] \\'], body, 'lrrr')
    output['attacks.tex'] = ''
    output['costs.tex'] = ''
    output['universal.tex'] = ''
    output['bounds.tex'] = ''
    for d in METHODS:
        body = []
        for a in ATTACKS:
            r = get(d, 'attack_specific', a)
            body.append(' & '.join([ATTACK_NAMES[a], pct(r['raw_tpr']), pct(r['clipped_tpr']), change(r)]) + r' \\')
        output['attacks.tex'] += table(NAMES[d] + ': attack-specific detection, with 5,149 test texts per attack.', 'tab:raid-attacks-' + d.replace('_', '-'),
            ['Attack', 'Raw TPR', 'Clipped TPR', r'Change [95\% CI]'], body, 'lrrr', True)
        body = []
        for scope, groups in [('rate_specific', RATES), ('attack_specific', ATTACKS)]:
            for g in groups:
                r = get(d, scope, g, 'clean')
                group = r'\(' + g + r'\)' if scope == 'rate_specific' else ATTACK_NAMES[g]
                body.append(' & '.join([group, pct(r['raw_tpr']) + '/' + pct(r['clipped_tpr']), change(r),
                    pct(r['raw_test_fpr']) + '/' + pct(r['clipped_test_fpr']), change(r, 'fpr')]) + r' \\')
        output['costs.tex'] += table(NAMES[d] + ': clean detection and false positives for each specific fit. Each entry Raw/Clip gives both operating points.',
            'tab:raid-costs-' + d.replace('_', '-'), ['Fitting group', 'Clean TPR', r'Change [95\% CI]', 'Human FPR', r'Change [95\% CI]'], body, 'lrrrr', True)
        body = []
        for cond in ['all_attacks', 'eligible_attacks'] + ['rate:' + r for r in RATES]:
            rs = [get(d, s, g, cond) for s, g in [('full_universal', 'all_attacks'), ('eligible_universal', 'eligible_attacks')]]
            name = {'all_attacks': 'All attacks', 'eligible_attacks': r'\(0<\rho\leq0.5\)'}.get(cond, r'\(' + cond.removeprefix('rate:') + r'\)')
            body.append(' & '.join([name, pct(rs[0]['raw_tpr']), pct(rs[0]['clipped_tpr']), change(rs[0], stack=True), pct(rs[1]['clipped_tpr']), change(rs[1], stack=True)]) + r' \\[3pt]')
        output['universal.tex'] += table(NAMES[d] + ': secondary universal rules evaluated on matched test populations.', 'tab:raid-universal-' + d.replace('_','-'),
            ['Test population', 'Raw', 'Full', r'Change [95\% CI]', 'Eligible', r'Change [95\% CI]'], body, 'lrrrrr', True)
        body = []
        for s, g, name in [('full_universal', 'all_attacks', 'Full'), ('eligible_universal', 'eligible_attacks', 'Eligible')]:
            r = get(d, s, g, 'clean')
            body.append(' & '.join([name, pct(r['raw_tpr']) + '/' + pct(r['clipped_tpr']), change(r), pct(r['raw_test_fpr']) + '/' + pct(r['clipped_test_fpr']), change(r, 'fpr')]) + r' \\')
        output['universal.tex'] += table(NAMES[d] + ': clean TPR and human FPR for the two universal fits (Raw/Clip).', 'tab:raid-universal-cost-' + d.replace('_','-'),
            ['Fit', 'Clean TPR', r'Change [95\% CI]', 'Human FPR', r'Change [95\% CI]'], body, 'lrrrr', True)
        body = []
        ordered = sorted([s for s in specs if s['detector'] == d], key=lambda s: (['rate_specific','attack_specific','full_universal','eligible_universal'].index(s['scope']), s['group']))
        for s in ordered:
            g = s['group']
            name = ATTACK_NAMES.get(g, {'all_attacks': 'Full universal', 'eligible_attacks': 'Eligible universal'}.get(g, r'\(' + g + r'\)'))
            spec = s['specification']
            values = [format(float(spec[k]), '.6g') if k in spec else '--' for k in ['lower','nll_upper','log_rank_upper']]
            body.append(' & '.join([name, str(s['n_tuning_machine']), *values]) + r' \\')
        direction = ordered[0]['direction']
        assert all(s['direction'] == direction for s in ordered)
        output['bounds.tex'] += table(NAMES[d] + r': selected bounds; direction \(d=' + ('+1' if direction == 1 else '-1') + r'\).',
            'tab:raid-bounds-' + d.replace('_','-'), ['Fitting group', 'Tuning texts', 'Floor', 'NLL cap', 'Log-rank cap'], body, 'lrrrr', True)
    # Keep confidence intervals in the appendix; main tables show point changes.
    for name in ['main_rates.tex', 'main_attacks.tex']:
        output[name] = re.sub(r'\\shortstack\{([+-][0-9.]+)\\\\\{\}\[[^\]]+\]\}', r'\1', output[name])
        output[name] = output[name].replace(r'Change [95\% CI]', 'Change').replace(r'\\[3pt]', r'\\')
        output[name] = output[name].replace(r'\tabcolsep}{3pt}', r'\tabcolsep}{6pt}')
        output[name] = output[name].replace(r'TPRs are percentages; changes are percentage points with paired 95\% confidence intervals.', 'TPRs are percentages; changes are clipped minus raw in percentage points.')
        output[name] = output[name].replace(r'in percentage points, with paired 95\% confidence intervals.', 'in percentage points.')
    for name, content in output.items():
        (OUT / name).write_text(content, encoding='utf-8')
    from .raid_diagnostics import build_diagnostics
    build_diagnostics(SOURCE, OUT)
    (OUT / 'provenance.json').write_text(json.dumps({'source': 'reference/raid', 'input_artifacts': marker['artifacts'], 'specifications':119, 'result_rows':462,
        'generated': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in output.items()}}, indent=2) + '\n')
    print('PASS: current input hashes; 119 fits; 462 unique result rows; ' + str(len(output)) + ' generated table fragments')


def main():
    import argparse
    from .verify import ROOT
    parser = argparse.ArgumentParser(description="Regenerate the paper's four-scope RAID tables.")
    parser.add_argument('--source', type=Path, default=ROOT / 'reference/raid')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)

if __name__ == '__main__':
    main()
