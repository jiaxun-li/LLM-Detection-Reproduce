"""Restore exact controlled source manifests from original files or pinned datasets.

No models are loaded. Outputs must be new; reference inputs are never changed.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from experiment_core.preparation.data import stable_int
from .provenance import jsonl, read, require
from .verify import DEFAULT_PROVENANCE, ROOT, digest, new_output, verify_inventory


def encoded(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf-8')


def restore_rows(identities, raw_rows, fields):
    needed = {i for row in identities for i in (row['source_index'] if isinstance(row['source_index'], list) else [row['source_index']])}
    texts = {}
    for index, raw in enumerate(raw_rows):
        if index not in needed:
            continue
        text = None
        for field in fields:
            value = raw.get(field)
            if isinstance(value, str):
                candidate = re.sub(r'\s+', ' ', value).strip()
                if len(candidate.split()) >= 20:
                    text = candidate
                    break
        require(text is not None, 'Dataset row cannot supply the recorded source text')
        texts[index] = text
        if len(texts) == len(needed):
            break
    require(texts.keys() == needed, 'Pinned dataset is missing recorded row indices')
    restored = []
    for identity in identities:
        row = dict(identity)
        indices = row['source_index']
        text = '\n\n'.join(texts[i] for i in indices) if isinstance(indices, list) else texts[indices]
        expected = row.pop('source_text_sha256')
        require(hashlib.sha256(text.encode('utf-8')).hexdigest() == expected, 'Source text hash mismatch; refusing substituted data')
        row['source_text'] = text
        restored.append(row)
    return restored


def cache_name(config, dataset):
    spec = config['datasets'][dataset]
    signature = stable_int(json.dumps(spec, sort_keys=True), json.dumps(config['splits'], sort_keys=True),
                           json.dumps(config['generation']['seeds']), bits=31)
    count = sum(config['splits'][key] for key in ['clipping_tuning', 'calibration', 'test'])
    return f"{dataset}-{spec['split']}-{count}-seed{config['splits']['selection_seed']}-{signature:08x}.jsonl"


def restore(output, original_directory=None):
    verify_inventory(DEFAULT_PROVENANCE, 'provenance.complete.json', 'artifacts', None)
    inputs = [DEFAULT_PROVENANCE]
    if original_directory is not None:
        inputs.append(Path(original_directory))
    output = new_output(output, inputs=inputs)
    index = read(DEFAULT_PROVENANCE / 'source_manifests.json')
    pending = {}
    summary = {}
    for dataset, entry in index.items():
        name = Path(entry['file']).name
        configs = [read(p) for p in sorted((ROOT / 'configs/paper_cells').glob(dataset + '--*.json'))]
        require(len(configs) == 3, 'Expected three configurations for each dataset')
        if original_directory is not None:
            original = Path(original_directory) / name
            require(digest(original) == entry['original_file'], 'Original manifest hash mismatch')
            data = original.read_bytes()
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError('Install datasets for pinned dataset restoration, or supply --original-directory') from exc
            identities = jsonl(DEFAULT_PROVENANCE / entry['file'])
            first = identities[0]
            kwargs = dict(path=first['dataset_id'], split=first['dataset_split'],
                          revision=first['dataset_resolved_revision'], streaming=True)
            if first['dataset_config'] is not None:
                kwargs['name'] = first['dataset_config']
            rows = restore_rows(identities, load_dataset(**kwargs), configs[0]['datasets'][dataset]['text_fields'])
            data = encoded(rows)
        require({'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == entry['original_file'],
                'Restored original manifest bytes do not match the recorded hash')
        names = sorted({cache_name(config, dataset) for config in configs})
        for target in names:
            pending[target] = data
        summary[dataset] = {'original_name': name, 'restored_cache_names': names, **entry['original_file']}
    output.mkdir(parents=True)
    for name, data in pending.items():
        (output / name).write_bytes(data)
    (output / 'source-recovery.complete.json').write_text(json.dumps({'status': 'complete', 'datasets': summary}, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New source-manifest cache directory')
    parser.add_argument('--original-directory', type=Path, help='Optional directory of original source JSONL files; avoids download')
    args = parser.parse_args()
    print(json.dumps(restore(args.output, args.original_directory), indent=2))


if __name__ == '__main__':
    main()
