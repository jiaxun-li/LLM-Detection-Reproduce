"""Check recovered source membership and scoring provenance, without inference."""
import csv
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(provenance, raid):
    provenance, raid = Path(provenance), Path(raid)
    excluded = read(provenance / 'raid/excluded_sources.json')['source_ids']
    require(len(excluded) == len(set(excluded)) == 500, 'Expected 500 unique exclusions')
    sources = jsonl(provenance / 'raid/selected_sources.jsonl')
    selected = {row['source_id']: row for row in sources}
    require(len(selected) == len(sources) == 12871, 'Unexpected RAID source population')
    require(not set(excluded).intersection(selected), 'Excluded source overlaps RAID selection')
    require(Counter(r['split'] for r in sources) == {'clipping_tuning': 5148, 'calibration': 2574, 'test': 5149}, 'Unexpected RAID splits')
    with (raid / 'selected_sources.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 12871 and {r['source_id'] for r in rows} == selected.keys(), 'Source CSV coverage mismatch')
    for row in rows:
        require(all(str(selected[row['source_id']].get(k)) == v for k, v in row.items()), 'Source CSV identity mismatch')
    expected = set()
    for row in sources:
        expected.add((row['source_id'], 'none', row['base_generation_id']))
        expected.update((row['source_id'], attack, identity) for attack, identity in row['attack_raid_ids'].items())
    with (raid / 'selected_machine_rows.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    actual = {(r['source_id'], r['attack'], r['raid_id']) for r in rows}
    require(len(rows) == len(actual) == 154452 and actual == expected, 'Machine-row identity mismatch')
    index = read(provenance / 'source_manifests.json')
    require(set(index) == {'squad', 'writingprompts', 'xsum'}, 'Missing controlled dataset')
    for dataset, entry in index.items():
        rows = jsonl(provenance / entry['file'])
        require(len(rows) == len({r['source_id'] for r in rows}) == 3000, 'Controlled source count/uniqueness mismatch')
        require(Counter(r['split'] for r in rows) == {'clipping_tuning': 500, 'calibration': 500, 'test': 2000}, 'Controlled split mismatch')
        require(all(r['dataset'] == dataset and len(r['source_text_sha256']) == 64 for r in rows), 'Controlled identity metadata mismatch')
        records = [read(p) for p in (provenance / 'primary').glob(dataset + '--*.json')]
        require(len(records) == 3, 'Expected three controlled model cells')
        for record in records:
            require(record['original_source_manifest'] == entry['original_file'], 'Original source manifest hash mismatch')
            require(all(r['dataset_resolved_revision'] == record['dataset']['resolved_revision'] for r in rows), 'Dataset revision mismatch')
    for name in ['origin-gate.json', 'components-gate.json']:
        gate = read(provenance / 'raid' / name)['record']
        require(gate['status'] == 'pass' and gate['parity_checks'] == 314, 'Scoring gate did not pass')
        require(gate['parity_scope'] == 'same_logits_upstream_mean_components', 'Unexpected gate parity scope')
        require(gate['source_exclusions']['sha256'] == read(provenance / 'raid.json')['source_exclusions']['sha256'], 'Gate exclusion hash mismatch')
        require(gate['tokenizer_preflight']['sources'] == 12871 and gate['tokenizer_preflight']['unscorable_rows'] == 0, 'Gate population mismatch')
    require(read(provenance / 'availability.json')['analysis_environment_snapshot']['available'] is False,
            'Unverified analysis environment must remain unavailable')
    return {'raid_exclusions': 500, 'raid_sources': 12871, 'controlled_sources': 9000, 'scoring_gates': 2}
