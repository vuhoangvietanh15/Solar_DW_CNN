import argparse
import json
from pathlib import Path
from .data import DEFAULT_DATA, DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST, prepare, read_day


def main():
    p = argparse.ArgumentParser(description='Audit CSVs or prepare disjoint observing-day splits')
    p.add_argument('--source', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output', type=Path, default=Path('data/prepared'))
    p.add_argument('--train-days', nargs='+', default=DEFAULT_TRAIN)
    p.add_argument('--val-days', nargs='+', default=DEFAULT_VAL)
    p.add_argument('--test-days', nargs='+', default=DEFAULT_TEST)
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    if a.audit_only:
        report = []
        for day in sorted(a.source.iterdir()):
            if not day.is_dir() or not list(day.glob('*_PSD.csv')):
                continue
            b = read_day(a.source, day.name)
            v, f = b['values'], b['frequency']
            report.append({'day': day.name, 'files': len(v), 'bins': len(f),
                           'first_mhz': float(f[0]), 'last_mhz': float(f[-1]),
                           'recorded_db_min': float(v.min()), 'recorded_db_max': float(v.max()),
                           'frequency_axis_verified': False})
        if not report:
            raise ValueError(f'No observing-day folders in {a.source}')
        print(json.dumps(report, indent=2))
    else:
        meta = prepare(a.source, a.output, a.train_days, a.val_days, a.test_days)
        print(json.dumps(meta, indent=2)[:1200])
        print('Prepared:', a.output.resolve())


if __name__ == '__main__':
    main()
