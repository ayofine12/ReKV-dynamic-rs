import argparse
import json
from pathlib import Path


def read_acc(path):
    data = json.loads(path.read_text())
    if 'acc' not in data:
        raise KeyError(f'{path} does not contain acc')
    return float(data['acc'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench')
    parser.add_argument('--sample_fps', default='1.0')
    parser.add_argument('--rs', nargs='+', type=int, default=[16, 64])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = []
    for rs in args.rs:
        result_path = base_dir / f'{rs}-{args.sample_fps}' / 'result.json'
        acc = read_acc(result_path)
        rows.append((rs, acc, result_path))

    for rs, acc, result_path in rows:
        print(f'rs={rs}: acc={acc * 100:.2f}% ({result_path})')

    if len(rows) >= 2:
        first_rs, first_acc, _ = rows[0]
        last_rs, last_acc, _ = rows[-1]
        print(f'delta rs{last_rs}-rs{first_rs}: {(last_acc - first_acc) * 100:+.2f} pp')


if __name__ == '__main__':
    main()
