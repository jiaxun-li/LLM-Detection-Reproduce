"""Regenerate publication summary PDF figures and CSV/LaTeX/JSON tables on CPU."""
import argparse
import contextlib
import json
import shutil
from pathlib import Path
from .verify import DEFAULT_BUNDLE, DEFAULT_RAID, new_output, verify_inventory, digest, implementation_hashes, absolute_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--raid', type=Path, default=DEFAULT_RAID)
    parser.add_argument('--study', choices=['all', 'controlled', 'raid'], default='all')
    parser.add_argument('--png', action='store_true', help='Also render PNG previews; requires pdftoppm on PATH')
    args = parser.parse_args()
    args.bundle, args.raid = absolute_path(args.bundle), absolute_path(args.raid)
    if not __debug__:
        raise RuntimeError('Run without -O: source-table validation assertions are required')
    out = new_output(args.output, (args.bundle, args.raid))
    if args.png and not shutil.which('pdftoppm'):
        raise RuntimeError('--png requires Poppler pdftoppm on PATH')
    verify_inventory(args.bundle, 'bundle.complete.json', 'artifact_hashes', None)
    inputs = {'bundle.complete.json': digest(args.bundle / 'bundle.complete.json')}
    if args.study in ('all', 'raid'):
        verify_inventory(args.raid, 'focused.complete.json', 'artifacts', None)
        from .paper_contract import validate
        validate(args.raid)
        inputs['focused.complete.json'] = digest(args.raid / 'focused.complete.json')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'build.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        if args.study in ('all', 'controlled'):
            from .controlled import build
            build(args.bundle.resolve(), out / 'controlled', args.png)
        if args.study in ('all', 'raid'):
            from .raid import build
            build(args.raid.resolve(), out / 'raid', args.png)
    verify_inventory(args.bundle, 'bundle.complete.json', 'artifact_hashes', None)
    after = {'bundle.complete.json': digest(args.bundle / 'bundle.complete.json')}
    if args.study in ('all', 'raid'):
        verify_inventory(args.raid, 'focused.complete.json', 'artifacts', None)
        after['focused.complete.json'] = digest(args.raid / 'focused.complete.json')
    if after != inputs:
        raise ValueError('Reference inventories changed during reproduction')
    files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'reproduction.complete.json').write_text(json.dumps({
        'status': 'complete', 'study': args.study, 'inference_performed': False, 'reference_inventories': inputs,
        'raid_result_scope': 'Paper four-scope cross-fitted TPR; 119 fits, 462 results',
        'implementation_hashes': implementation_hashes(),
        'description': 'Summary presentation regenerated from validated frozen results; no refitting or new intervals.',
        'output_hashes': files}, indent=2) + '\n', encoding='utf-8')
    print(f'Reproduced {len(files)} files in {out}')


if __name__ == '__main__':
    main()
