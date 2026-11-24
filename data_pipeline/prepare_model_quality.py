"""Seal disjoint development and held-out manifests for the fixed MIDV-2020 split."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


LAYOUTS = {
    'train': {'aze_passport', 'grc_passport', 'esp_id', 'fin_id', 'svk_id'},
    'validation': {'est_id', 'srb_passport'},
    'test': {'alb_id', 'lva_passport'},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, strict=True)
        rows = list(reader)
        columns = reader.fieldnames
    for split, layouts in LAYOUTS.items():
        if {row['template_id'] for row in rows if row['split'] == split} != layouts:
            raise ValueError(f'Fixed layout assignment changed for {split}')
    counts = Counter((row['split'], row['label']) for row in rows)
    if counts != {('train', 'ID_Card'): 900, ('train', 'Passport'): 600,
                  ('validation', 'ID_Card'): 300, ('validation', 'Passport'): 300,
                  ('test', 'ID_Card'): 300, ('test', 'Passport'): 300}:
        raise ValueError('Expected the full seed-42 MIDV-2020 layout sample')
    if len({row['image_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    args.output.mkdir()
    summary = {'splits': {split: {label: counts[split, label] for label in ('ID_Card', 'Passport')}
                          for split in LAYOUTS}, 'layouts': {key: sorted(value) for key, value in LAYOUTS.items()}}
    for name, held_out in [('development.csv', False), ('held-out.csv', True)]:
        path = args.output / name
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(row for row in rows if (row['split'] == 'test') == held_out)
        summary[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = json.dumps(summary, indent=2)+'\n'
    (args.output / 'split.json').write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()
