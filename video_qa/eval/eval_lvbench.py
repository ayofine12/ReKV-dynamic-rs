import argparse
import ast
import json
import os
from collections import defaultdict

import pandas as pd


def _as_types(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (SyntaxError, ValueError):
            pass
        return [value]
    return [str(value)]


def _pred_choice(value):
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return ""
    if ")" in value:
        idx = value.index(")")
        if idx > 0:
            return value[idx - 1].upper()
    first = value[0].upper()
    if first in "ABCDEFGH":
        return first
    for char in value.upper():
        if char in "ABCDEFGH":
            return char
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--anno_path', type=str, default='data/lvbench/full_mc.json')
    args = parser.parse_args()

    if args.results_path is None:
        results_path = os.path.join(args.save_dir, 'results.csv')
    else:
        results_path = args.results_path
        if args.save_dir is None:
            args.save_dir = os.path.dirname(results_path)

    df = pd.read_csv(results_path)
    if 'pred_choice' not in df.columns:
        df['pred_choice'] = df['pred_answer'].map(_pred_choice)
    else:
        df['pred_choice'] = df['pred_choice'].map(_pred_choice)

    if 'correct_choice' in df.columns:
        df['qa_acc_recomputed'] = (df['pred_choice'] == df['correct_choice']).astype(float) * 100.0
    elif 'qa_acc' not in df.columns:
        raise ValueError('results must contain either correct_choice or qa_acc')

    metric_col = 'qa_acc_recomputed' if 'qa_acc_recomputed' in df.columns else 'qa_acc'
    print(f'#Samples: {len(df)}')
    print(f'Average qa_acc: {df[metric_col].mean():.2f}')

    if 'task' in df.columns:
        category_total = defaultdict(int)
        category_right = defaultdict(float)
        for _, row in df.iterrows():
            for category in _as_types(row['task']):
                category_total[category] += 1
                category_right[category] += float(row[metric_col]) / 100.0
        category_acc = {
            category: category_right[category] / total
            for category, total in category_total.items()
            if total > 0
        }
        for category in sorted(category_acc):
            print(f'{category}: {category_acc[category] * 100:.2f} ({category_total[category]})')
    else:
        category_acc = {}

    if 'uid' in df.columns:
        answers = {
            str(row['uid']): row['pred_choice']
            for _, row in df.iterrows()
            if not pd.isna(row['uid'])
        }
        answers_path = os.path.join(args.save_dir, 'answers.json')
        with open(answers_path, 'w') as f:
            json.dump(answers, f, indent=2)
        print(f'answers: {answers_path}')

    result = {k: v for k, v in category_acc.items()}
    result['acc'] = float(df[metric_col].mean() / 100.0)
    result_path = os.path.join(args.save_dir, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'result: {result_path}')
    print(f'save_dir: {args.save_dir}')


if __name__ == '__main__':
    main()
