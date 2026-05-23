import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch


def parse_alpha(path):
    match = re.search(r"alpha([0-9p.]+)-", path.name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def layer_actual_topk(layer):
    if layer.get("actual_topk") is not None:
        return float(layer["actual_topk"])
    selected_topk = layer.get("selected_topk_per_unit")
    if selected_topk:
        return float(sum(selected_topk) / len(selected_topk))
    selected = layer.get("retrieved_block_indices")
    if selected:
        return float(len(selected[0]))
    return None


def collect_logit_costs(results_csv):
    df = pd.read_csv(results_csv)
    if "retrieval_logits_path" not in df.columns:
        return [], []

    rows = []
    layer_rows = []
    for value in df["retrieval_logits_path"].dropna().astype(str):
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            continue
        payload = torch.load(path, weights_only=False, map_location="cpu")
        for layer in payload.get("retrieval", {}).get("layers", []):
            actual_topk = layer_actual_topk(layer)
            if actual_topk is None:
                continue
            layer_idx = int(layer.get("layer_idx", -1))
            alpha = layer.get("dynamic_alpha")
            max_topk = layer.get("dynamic_max_topk", layer.get("topk"))
            rows.append({
                "layer_idx": layer_idx,
                "actual_topk": actual_topk,
                "dynamic_alpha": alpha,
                "dynamic_max_topk": max_topk,
            })
            layer_rows.append({
                "layer_idx": layer_idx,
                "actual_topk": actual_topk,
            })
    return rows, layer_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench/dynamic_mass_min16_max64",
    )
    parser.add_argument("--sample_fps", default="1.0")
    parser.add_argument("--max_rs", type=float, default=64.0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    summary_rows = []
    all_layer_rows = []
    for run_dir in sorted(base_dir.glob(f"alpha*-{args.sample_fps}")):
        alpha = parse_alpha(run_dir)
        if alpha is None:
            continue
        result_path = run_dir / "result.json"
        results_csv = run_dir / "results.csv"
        if not result_path.exists() or not results_csv.exists():
            continue

        with result_path.open() as f:
            result = json.load(f)
        result_df = pd.read_csv(results_csv)
        cost_rows, layer_rows = collect_logit_costs(results_csv)
        avg_actual_topk = None
        cost_ratio = None
        if cost_rows:
            avg_actual_topk = sum(row["actual_topk"] for row in cost_rows) / len(cost_rows)
            cost_ratio = avg_actual_topk / args.max_rs

        summary_rows.append({
            "alpha": alpha,
            "acc": result.get("acc"),
            "num_questions": len(result_df),
            "avg_actual_topk": avg_actual_topk,
            "cost_ratio_vs_max": cost_ratio,
            "run_dir": str(run_dir),
        })
        for row in layer_rows:
            row["alpha"] = alpha
            all_layer_rows.append(row)

    if not summary_rows:
        raise FileNotFoundError(f"No completed alpha runs found under {base_dir}")

    summary = pd.DataFrame(summary_rows).sort_values("alpha")
    summary_path = base_dir / "dynamic_alpha_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"summary: {summary_path}")

    if all_layer_rows:
        layer_df = pd.DataFrame(all_layer_rows)
        layer_summary = (
            layer_df.groupby(["alpha", "layer_idx"], as_index=False)["actual_topk"]
            .mean()
            .sort_values(["alpha", "layer_idx"])
        )
        layer_path = base_dir / "dynamic_alpha_layer_cost.csv"
        layer_summary.to_csv(layer_path, index=False)
        print(f"layer_cost: {layer_path}")


if __name__ == "__main__":
    main()
