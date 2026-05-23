import argparse
import json
from pathlib import Path

import pandas as pd


def safe_acc(row, col):
    try:
        return float(row[col]) >= 50.0
    except Exception:
        return False


def load_result(path):
    with open(path) as f:
        return json.load(f)


def load_csv(save_dir):
    csv_path = Path(save_dir) / 'results.csv'
    if not csv_path.exists():
        csv_path = Path(save_dir) / '1_0.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df['uid'] = df['uid'].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench')
    parser.add_argument('--ablation_subdir', default='layer_ablation_base16_high64')
    parser.add_argument('--sample_fps', default='1.0')
    parser.add_argument('--base_rs', type=int, default=16)
    parser.add_argument('--high_rs', type=int, default=64)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    ablation_root = base_dir / args.ablation_subdir
    base_save = base_dir / f'{args.base_rs}-{args.sample_fps}'
    high_save = base_dir / f'{args.high_rs}-{args.sample_fps}'

    base_result = load_result(base_save / 'result.json')
    high_result = load_result(high_save / 'result.json')
    base_acc = float(base_result.get('acc', float('nan')))
    high_acc = float(high_result.get('acc', float('nan')))

    base_df = load_csv(base_save)
    high_df = load_csv(high_save)
    if base_df is not None:
        base_df = base_df[['uid', 'qa_acc']].rename(columns={'qa_acc': f'qa_acc_rs{args.base_rs}'})
    if high_df is not None:
        high_df = high_df[['uid', 'qa_acc']].rename(columns={'qa_acc': f'qa_acc_rs{args.high_rs}'})

    rows = []
    for result_path in sorted(ablation_root.glob(f'*-{args.sample_fps}/result.json')):
        save_dir = result_path.parent
        label = save_dir.name.removesuffix(f'-{args.sample_fps}')
        result = load_result(result_path)
        acc = float(result.get('acc', float('nan')))
        row = {
            'label': label,
            'save_dir': str(save_dir),
            'acc': acc,
            'delta_vs_base_rs': acc - base_acc,
            'delta_vs_high_rs': acc - high_acc,
            f'base_rs{args.base_rs}_acc': base_acc,
            f'high_rs{args.high_rs}_acc': high_acc,
        }
        for key, value in result.items():
            if key != 'acc':
                row[f'acc_{key}'] = value

        ablation_df = load_csv(save_dir)
        if ablation_df is not None and base_df is not None and high_df is not None:
            ablation_df = ablation_df[['uid', 'qa_acc']].rename(columns={'qa_acc': 'qa_acc_ablation'})
            merged = ablation_df.merge(base_df, on='uid', how='inner').merge(high_df, on='uid', how='inner')
            base_col = f'qa_acc_rs{args.base_rs}'
            high_col = f'qa_acc_rs{args.high_rs}'
            base_right = merged[base_col] >= 50.0
            high_right = merged[high_col] >= 50.0
            abl_right = merged['qa_acc_ablation'] >= 50.0

            row['num_questions'] = int(len(merged))
            row['ablation_right_base_wrong'] = int((abl_right & ~base_right).sum())
            row['ablation_wrong_base_right'] = int((~abl_right & base_right).sum())
            row['net_vs_base_questions'] = row['ablation_right_base_wrong'] - row['ablation_wrong_base_right']
            row['recovers_rs64_gains'] = int((~base_right & high_right & abl_right).sum())
            row['misses_rs64_gains'] = int((~base_right & high_right & ~abl_right).sum())
            row['avoids_rs64_losses'] = int((base_right & ~high_right & abl_right).sum())
            row['matches_high_rs'] = int((abl_right == high_right).sum())
            row['matches_base_rs'] = int((abl_right == base_right).sum())

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values('delta_vs_base_rs', ascending=False)
    out_path = ablation_root / 'layer_ablation_summary.csv'
    summary.to_csv(out_path, index=False)

    print(f'base rs{args.base_rs}: {base_acc:.6f}')
    print(f'high rs{args.high_rs}: {high_acc:.6f}')
    print(f'summary: {out_path}')
    if not summary.empty:
        cols = ['label', 'acc', 'delta_vs_base_rs', 'delta_vs_high_rs', 'net_vs_base_questions', 'recovers_rs64_gains', 'avoids_rs64_losses']
        cols = [col for col in cols if col in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == '__main__':
    main()
