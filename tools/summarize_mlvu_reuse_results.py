#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def parse_run(run_dir):
    name = run_dir.name
    alpha_match = re.match(r"alpha([0-9p.]+)-", name)
    if alpha_match:
        alpha = float(alpha_match.group(1).replace("p", "."))
        return f"alpha{alpha:g}", "dynamic", alpha, None
    rs_match = re.match(r"rs([0-9]+)-", name)
    if rs_match:
        rs = int(rs_match.group(1))
        return f"rs{rs}", "fixed", None, rs
    return name, "unknown", None, None


def layer_actual_topk(layer):
    if layer.get("actual_topk") is not None:
        return float(layer["actual_topk"])
    selected = layer.get("selected_topk_per_unit") or []
    if selected:
        return float(sum(selected) / len(selected))
    indices = layer.get("retrieved_block_indices") or []
    if indices:
        return float(len(indices[0]))
    return None


def collect_costs(df, fallback_rs):
    if "retrieval_logits_path" not in df.columns or torch is None:
        return fallback_rs, None, None

    topks = []
    for value in df["retrieval_logits_path"].dropna().astype(str):
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for layer in payload.get("retrieval", {}).get("layers", []):
            topk = layer_actual_topk(layer)
            if topk is not None:
                topks.append(topk)

    if not topks:
        return fallback_rs, None, None
    s = pd.Series(topks, dtype="float64")
    return float(s.mean()), float(s.median()), float(s.quantile(0.90))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=[
            "/mnt/ssd1/mwnoh/MLVU/results/qwen2_5_vl_7b/mlvu/dynamic_mass_min16_max64_maxdur3600",
            "/mnt/ssd1/mwnoh/MLVU/results/qwen2_5_vl_7b/mlvu/fixed_rs16_rs64_maxdur3600",
        ],
    )
    parser.add_argument("--output", default="analysis_outputs/mlvu_reuse_summary.csv")
    args = parser.parse_args()

    rows = []
    task_rows = []
    for base in [Path(p) for p in args.base_dirs]:
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            results = run_dir / "results.csv"
            if not results.exists():
                continue

            label, run_type, alpha, rs = parse_run(run_dir)
            df = pd.read_csv(results)
            avg_blocks, median_blocks, p90_blocks = collect_costs(df, rs)
            rows.append({
                "label": label,
                "type": run_type,
                "alpha": alpha,
                "retrieve_size": rs,
                "n": len(df),
                "acc": float(df["qa_acc"].mean()),
                "avg_blocks": avg_blocks,
                "median_blocks": median_blocks,
                "p90_blocks": p90_blocks,
                "cost_vs_rs16": None if avg_blocks is None else avg_blocks / 16.0,
                "cost_vs_rs64": None if avg_blocks is None else avg_blocks / 64.0,
                "run_dir": str(run_dir),
            })
            for task, group in df.groupby("task"):
                task_rows.append({
                    "label": label,
                    "task": task,
                    "n": len(group),
                    "acc": float(group["qa_acc"].mean()),
                })

    if not rows:
        raise FileNotFoundError("No result.csv files found under --base-dirs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(["type", "retrieve_size", "alpha"], na_position="last")
    summary.to_csv(out_path, index=False)

    task_path = out_path.with_name(out_path.stem + "_task_accuracy.csv")
    pd.DataFrame(task_rows).sort_values(["task", "label"]).to_csv(task_path, index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"summary: {out_path.resolve()}")
    print(f"task_accuracy: {task_path.resolve()}")


if __name__ == "__main__":
    main()
