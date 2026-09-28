"""Check immutable reference artifact sizes and SHA256 (standard library only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def absolute_path(path):
    """Canonical absolute path with Windows extended-length IO support."""
    resolved = Path(path).resolve()
    if os.name != 'nt':
        return resolved
    value = str(resolved)
    if value.startswith('\\\\?\\'):
        return resolved
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)


ROOT = absolute_path(__file__).parents[2]
DEFAULT_BUNDLE = ROOT / 'reference/empirical'
DEFAULT_RAID = ROOT / 'reference/raid'
DEFAULT_PROVENANCE = ROOT / 'reference/provenance'


def digest(path):
    path = absolute_path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return {'size': path.stat().st_size, 'sha256': h.hexdigest()}


def implementation_hashes(extra=()):
    paths = [*absolute_path(__file__).parent.glob('*.py'), *(ROOT / name for name in extra)]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def verify_inventory(root, marker_name, key, count):
    root = absolute_path(root)
    marker = json.loads((root / marker_name).read_text(encoding='utf-8'))
    entries = marker[key]
    if marker.get('status') != 'complete' or (count is not None and len(entries) != count):
        raise ValueError(f'Unexpected completion marker: {marker_name}')
    for name, expected in entries.items():
        path = absolute_path(root / name)
        if not path.is_relative_to(root):
            raise ValueError(f'Artifact escapes reference root: {name}')
        if digest(path) != expected:
            raise ValueError(f'Artifact size/hash mismatch: {name}')
    return marker


def verify(bundle=DEFAULT_BUNDLE, raid=DEFAULT_RAID, provenance=DEFAULT_PROVENANCE):
    verify_inventory(bundle, 'bundle.complete.json', 'artifact_hashes', None)
    marker = verify_inventory(raid, 'focused.complete.json', 'artifacts', None)
    verify_inventory(provenance, 'provenance.complete.json', 'artifacts', None)
    from .paper_contract import validate
    validate(raid)
    from .provenance import validate as validate_provenance
    sources = validate_provenance(provenance, raid)
    availability = json.loads((absolute_path(provenance) / 'availability.json').read_text())
    return {'status': 'pass', 'controlled_artifacts': len(json.loads((absolute_path(bundle) / 'bundle.complete.json').read_text())['artifact_hashes']),
            'raid_fits': marker['specifications'], 'raid_result_rows': marker['result_rows'],
            'provenance_status': availability['status'],
            'recovered_sources': sources,
            'scope': 'Distribution integrity and paper-result consistency; not fresh inference or score-level replay.'}


def new_output(path, inputs=()):
    path = absolute_path(path)
    protected = [ROOT / 'reference', ROOT / 'paper', ROOT / 'downloads', ROOT / 'simulation', *inputs]
    if path.exists():
        raise ValueError('Output must be a NEW directory; existing outputs are never overwritten')
    for source in protected:
        source = absolute_path(source)
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError(f'Output overlaps protected input: {source}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--raid', type=Path, default=DEFAULT_RAID)
    parser.add_argument('--provenance', type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.raid, args.provenance), indent=2))


if __name__ == '__main__':
    main()
