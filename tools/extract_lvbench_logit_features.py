import argparse
import math
from pathlib import Path

import pandas as pd
import torch


def safe_float(value):
    if value is None:
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def load_results(save_dir, rs):
    csv_path = Path(save_dir) / 'results.csv'
    if not csv_path.exists():
        csv_path = Path(save_dir) / '1_0.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'No results.csv or 1_0.csv found in {save_dir}')

    df = pd.read_csv(csv_path)
    df['uid'] = df['uid'].astype(str)
    df = df.rename(columns={
        'qa_acc': f'qa_acc_rs{rs}',
        'pred_choice': f'pred_choice_rs{rs}',
        'pred_answer': f'pred_answer_rs{rs}',
        'retrieval_logits_path': f'retrieval_logits_path_rs{rs}',
    })
    keep = [
        'uid', 'video_id', 'question', 'answer', 'correct_choice', 'task',
        f'qa_acc_rs{rs}', f'pred_choice_rs{rs}', f'pred_answer_rs{rs}',
        f'retrieval_logits_path_rs{rs}',
    ]
    return df[[col for col in keep if col in df.columns]]


def normalize_probs(logits, temperature):
    logits = logits.float()
    std = logits.std(dim=-1, keepdim=True)
    mean = logits.mean(dim=-1, keepdim=True)
    z = (logits - mean) / std.clamp_min(1e-6)
    return torch.softmax(z / temperature, dim=-1)


def topk_span(indices):
    if indices.numel() == 0:
        return 0
    return int(indices.max().item() - indices.min().item() + 1)


def topk_segments(indices):
    if indices.numel() == 0:
        return 0
    values = sorted(int(v) for v in indices.tolist())
    segments = 1
    for prev, cur in zip(values, values[1:]):
        if cur != prev + 1:
            segments += 1
    return segments


def unit_features(logits, selected_indices, temperature, ks):
    num_blocks = int(logits.numel())
    if num_blocks == 0:
        return {}

    sorted_scores, sorted_indices = torch.sort(logits.float(), descending=True)
    probs = normalize_probs(logits[None, :], temperature=temperature)[0]
    sorted_probs = probs[sorted_indices]
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    entropy_norm = entropy / math.log(num_blocks) if num_blocks > 1 else 0.0
    effective_blocks = float(torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum()).item())

    out = {
        'num_blocks': num_blocks,
        'score_mean': float(logits.float().mean().item()),
        'score_std': float(logits.float().std().item()) if num_blocks > 1 else 0.0,
        'score_min': float(logits.float().min().item()),
        'score_max': float(logits.float().max().item()),
        'entropy': entropy,
        'entropy_norm': entropy_norm,
        'effective_blocks': effective_blocks,
        'top1_score': float(sorted_scores[0].item()),
        'top2_margin': float((sorted_scores[0] - sorted_scores[1]).item()) if num_blocks > 1 else float('nan'),
        'top5_margin': float((sorted_scores[0] - sorted_scores[min(4, num_blocks - 1)]).item()),
    }

    for k in ks:
        kk = min(int(k), num_blocks)
        idx = sorted_indices[:kk]
        out[f'top{k}_mass'] = float(sorted_probs[:kk].sum().item())
        out[f'top{k}_score_mean'] = float(sorted_scores[:kk].mean().item())
        out[f'top{k}_span_blocks'] = topk_span(idx)
        out[f'top{k}_num_segments'] = topk_segments(idx)
        out[f'top{k}_peak_density'] = float(kk / max(out[f'top{k}_span_blocks'], 1))

    if selected_indices is not None:
        sel = torch.as_tensor(selected_indices, dtype=torch.long)
        sel = sel[(sel >= 0) & (sel < num_blocks)]
        if sel.numel() > 0:
            out['selected_count'] = int(sel.numel())
            out['selected_mass'] = float(probs[sel].sum().item())
            out['selected_score_mean'] = float(logits[sel].float().mean().item())
            out['selected_span_blocks'] = topk_span(sel)
            out['selected_num_segments'] = topk_segments(sel)
        else:
            out['selected_count'] = 0
            out['selected_mass'] = float('nan')
            out['selected_score_mean'] = float('nan')
            out['selected_span_blocks'] = 0
            out['selected_num_segments'] = 0

    return out


def extract_features_from_pt(path, rs, temperature, ks):
    payload = torch.load(path, weights_only=False, map_location='cpu')
    rows = []
    for layer in payload['retrieval']['layers']:
        logits = layer.get('score_logits')
        if logits is None:
            continue
        logits = logits.float()
        selected = layer.get('retrieved_block_indices')
        if selected is None:
            selected = [None] * logits.shape[0]

        unit_rows = []
        for unit_idx in range(logits.shape[0]):
            selected_unit = selected[unit_idx] if unit_idx < len(selected) else None
            feats = unit_features(logits[unit_idx], selected_unit, temperature=temperature, ks=ks)
            if feats:
                unit_rows.append(feats)

        if not unit_rows:
            continue

        row = {
            'uid': str(payload.get('uid')),
            'video_id': payload.get('video_id'),
            'question': payload.get('question'),
            'task': payload.get('question_type'),
            'time_reference': payload.get('time_reference'),
            'rs': rs,
            'layer_idx': int(layer['layer_idx']),
            'layer_topk': int(layer.get('topk', rs)),
            'layer_chunk_size': int(layer.get('chunk_size', 1)),
            'block_size': int(layer.get('block_size', 0)),
            'num_global_block': int(layer.get('num_global_block', 0)),
            'num_units': int(logits.shape[0]),
        }
        for key in unit_rows[0].keys():
            values = [safe_float(unit[key]) for unit in unit_rows]
            row[f'{key}_mean'] = sum(values) / len(values)
        rows.append(row)
    return rows


def make_transition(row, low_rs, high_rs):
    low = safe_float(row.get(f'qa_acc_rs{low_rs}', float('nan'))) >= 50.0
    high = safe_float(row.get(f'qa_acc_rs{high_rs}', float('nan'))) >= 50.0
    if (not low) and high:
        return f'rs{low_rs}_wrong_rs{high_rs}_right'
    if low and high:
        return 'both_right'
    if low and (not high):
        return f'rs{low_rs}_right_rs{high_rs}_wrong'
    return 'both_wrong'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rs16_dir', default='/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench/16-1.0')
    parser.add_argument('--rs64_dir', default='/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench/64-1.0')
    parser.add_argument('--output_dir', default='/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench/logit_feature_analysis')
    parser.add_argument('--low_rs', type=int, default=16)
    parser.add_argument('--high_rs', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--ks', default='5,16,64')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in args.ks.split(',') if k.strip()]

    low_df = load_results(args.rs16_dir, args.low_rs)
    high_df = load_results(args.rs64_dir, args.high_rs)
    merged = low_df.merge(high_df, on=['uid'], how='inner', suffixes=(f'_rs{args.low_rs}', f'_rs{args.high_rs}'))
    merged['transition'] = merged.apply(lambda row: make_transition(row, args.low_rs, args.high_rs), axis=1)
    merged_path = output_dir / 'rs16_rs64_merged_results.csv'
    merged.to_csv(merged_path, index=False)

    feature_rows = []
    for rs, path_col in [(args.low_rs, f'retrieval_logits_path_rs{args.low_rs}'), (args.high_rs, f'retrieval_logits_path_rs{args.high_rs}')]:
        if path_col not in merged.columns:
            raise KeyError(f'{path_col} is missing. Run with --save_retrieval_logits true first.')
        for path_value in merged[path_col].dropna().astype(str):
            if not path_value:
                continue
            path = Path(path_value)
            if not path.exists():
                continue
            feature_rows.extend(extract_features_from_pt(path, rs=rs, temperature=args.temperature, ks=ks))

    features = pd.DataFrame(feature_rows)
    features_path = output_dir / 'layer_logit_features.csv'
    features.to_csv(features_path, index=False)

    transition_cols = ['uid', 'transition', f'qa_acc_rs{args.low_rs}', f'qa_acc_rs{args.high_rs}']
    transition_df = merged[[col for col in transition_cols if col in merged.columns]]
    features_with_transition = features.merge(transition_df, on='uid', how='left')
    features_transition_path = output_dir / 'layer_logit_features_with_transition.csv'
    features_with_transition.to_csv(features_transition_path, index=False)

    summary = (
        features_with_transition
        .groupby(['rs', 'transition', 'layer_idx'])
        .mean(numeric_only=True)
        .reset_index()
    )
    summary_path = output_dir / 'layer_logit_feature_summary_by_transition.csv'
    summary.to_csv(summary_path, index=False)

    counts = merged['transition'].value_counts().rename_axis('transition').reset_index(name='count')
    counts_path = output_dir / 'transition_counts.csv'
    counts.to_csv(counts_path, index=False)

    print(f'merged: {merged_path} ({len(merged)} questions)')
    print(f'features: {features_path} ({len(features)} rows)')
    print(f'features_with_transition: {features_transition_path}')
    print(f'summary: {summary_path}')
    print(f'transition_counts: {counts_path}')
    print(counts.to_string(index=False))


if __name__ == '__main__':
    main()
