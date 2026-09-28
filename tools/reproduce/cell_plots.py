"""Regenerate the 99 controlled-study cell PNGs; RAID is reported as tables."""
import argparse
import json
import os
import shutil
from pathlib import Path
from .verify import DEFAULT_BUNDLE, new_output, verify_inventory, digest, implementation_hashes, absolute_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=['controlled'], default='controlled')
    args = parser.parse_args()
    args.bundle = absolute_path(args.bundle)
    out = new_output(args.output, (args.bundle,))
    verify_inventory(args.bundle, 'bundle.complete.json', 'artifact_hashes', None)
    marker_digest = digest(args.bundle / 'bundle.complete.json')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(out / '.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    if args.study in ('all', 'controlled'):
        import plot_tpr_contamination as plotter
        from tools.reproduce.controlled import DETECTORS
        cells = sorted((args.bundle / 'primary/primary_cells').iterdir())
        if len(cells) != 9:
            raise ValueError('Expected exactly nine primary cells')
        for source in cells:
            destination = out / 'controlled' / source.name
            destination.mkdir(parents=True)
            rows = plotter.read_metrics(source / 'metrics.csv', 'primary_frozen_mixture')
            detectors = list(DETECTORS)
            plotter.plot(rows, detectors, destination / 'all_detectors.png', True)
            plotter.plot(rows, detectors, destination / 'detectors_2x7.png', True, layout='detector-columns')
            for fpr in (.01, .05):
                plotter.plot(rows, detectors, destination / f'detectors_row_fpr_{fpr * 100:.0f}pct.png',
                             True, layout='detector-columns', target_fprs=(fpr,))
            for detector in detectors:
                plotter.plot(rows, [detector], destination / f'{detector}.png', True)
    verify_inventory(args.bundle, 'bundle.complete.json', 'artifact_hashes', None)
    if digest(args.bundle / 'bundle.complete.json') != marker_digest:
        raise ValueError('Reference inventory changed during reproduction')
    files = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    count = sum(name.endswith('.png') for name in files)
    expected = 99
    if count != expected:
        raise ValueError(f'Expected {expected} PNGs, got {count}')
    (out / 'reproduction.complete.json').write_text(json.dumps({
        'status': 'complete', 'study': 'controlled contamination',
        'png_count': count, 'matplotlib_version': matplotlib.__version__,
        'reference_inventory': marker_digest, 'output_hashes': files,
        'implementation_hashes': implementation_hashes(('plot_tpr_contamination.py',)),
        'note': 'Original plotting APIs; pixel identity depends on rendering environment.'}, indent=2) + '\n', encoding='utf-8')
    print(f'Reproduced {count} PNG figures in {out}')


if __name__ == '__main__':
    main()
