"""Synthetic CSV adapter example. All units and time evidence are explicit."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import numpy as np
from motion_foundation import Channel
from motion_foundation.adapters import sha256
from motion_foundation.corpus import Builder


def convert_csv(source, out, subject, split):
    source = Path(source)
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['time_s', 'acc_x', 'acc_y', 'acc_z']:
            raise ValueError('expected time_s and acceleration xyz in m/s^2')
        rows = [[float(row[k]) if row[k] else np.nan for k in reader.fieldnames] for row in reader]
    data = np.asarray(rows)
    group = dict(cols=[0, 1, 2], modality='imu', channels=[asdict(Channel('acc', 'm/s^2', axis=a)) for a in 'xyz'])
    builder = Builder(out)
    builder.put('synthetic_csv', subject, source.name, data[:, 1:], data[:, 0], [group], split=split,
                time_evidence='measured', dependency=0., evidence='synthetic CSV generator; seconds and m/s^2', source_sha=sha256(source))
    return builder.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    target = Path(args.out)
    target.mkdir(exist_ok=False)
    source = target / 'synthetic.csv'
    times = np.arange(200) / 50.
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'acc_x', 'acc_y', 'acc_z'])
        writer.writerows(zip(times, np.sin(times), np.cos(times), np.sin(2 * times)))
    print(convert_csv(source, target, 'synthetic:participant', 'train'))


if __name__ == '__main__':
    main()
