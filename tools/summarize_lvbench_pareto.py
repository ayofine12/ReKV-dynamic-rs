#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(",", " ").split()]


def acc_percent(df):
    if "qa_acc" in df.columns:
        values = pd.to_numeric(df["qa_acc"], errors="coerce")
        mean = float(values.mean())
        return mean * 100.0 if mean <= 1.0 else mean
    if {"pred_choice", "correct_choice"} <= set(df.columns):
        pred = df["pred_choice"].astype(str).str.strip().str.upper()
        gold = df["correct_choice"].astype(str).str.strip().str.upper()
        return float((pred == gold).mean() * 100.0)
    raise ValueError("results.csv must contain qa_acc or pred_choice/correct_choice")


def correct_values(df):
    if "qa_acc" in df.columns:
        values = pd.to_numeric(df["qa_acc"], errors="coerce").fillna(0.0)
        return values > (0.5 if values.max() <= 1.0 else 50.0)
    pred = df["pred_choice"].astype(str).str.strip().str.upper()
    gold = df["correct_choice"].astype(str).str.strip().str.upper()
    return pred == gold


def sample_keys(df):
    if "uid" in df.columns:
        return df["uid"].astype(str)
    if {"video_path", "question"} <= set(df.columns):
        return df["video_path"].astype(str) + "||" + df["question"].astype(str)
    return pd.Series(range(len(df)), index=df.index).astype(str)


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
    if fallback_rs is not None:
        value = float(fallback_rs)
        return value, value, value, 0
    if torch is None or "retrieval_logits_path" not in df.columns:
        return None, None, None, 0

    topks = []
    files_read = 0
    for value in df["retrieval_logits_path"].dropna().astype(str):
        if not value:
            continue
        path = Path(value)
        if not path.exists():
            continue
        files_read += 1
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for layer in payload.get("retrieval", {}).get("layers", []):
            topk = layer_actual_topk(layer)
            if topk is not None:
                topks.append(topk)

    if not topks:
        return None, None, None, files_read
    series = pd.Series(topks, dtype="float64")
    return (
        float(series.mean()),
        float(series.median()),
        float(series.quantile(0.90)),
        files_read,
    )


def alpha_from_dir(path):
    match = re.search(r"alpha([0-9p.]+)-", path.name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def add_fixed_runs(root, fixed_reuse_dir, sample_fps, fixed_rs):
    runs = []
    for rs in fixed_rs:
        candidates = [
            root / f"{rs}-{sample_fps}",
            fixed_reuse_dir / f"rs{rs}-{sample_fps}",
        ]
        for run_dir in candidates:
            results = run_dir / "results.csv"
            if results.exists():
                runs.append((f"rs{rs}", "fixed", rs, None, run_dir, results))
                break
    return runs


def add_dynamic_runs(dynamic_dir, sample_fps):
    runs = []
    for run_dir in sorted(dynamic_dir.glob(f"alpha*-{sample_fps}")):
        alpha = alpha_from_dir(run_dir)
        results = run_dir / "results.csv"
        if alpha is not None and results.exists():
            runs.append((f"alpha{alpha:g}", "dynamic_alpha", None, alpha, run_dir, results))
    return runs


def build_summary(runs):
    records = []
    data_by_label = {}
    for label, run_type, rs, alpha, run_dir, results in runs:
        df = pd.read_csv(results).copy()
        df["_key"] = sample_keys(df).values
        df["_correct"] = correct_values(df).astype(bool).values
        avg_blocks, median_blocks, p90_blocks, files_read = collect_costs(df, rs)
        records.append({
            "label": label,
            "type": run_type,
            "rs": rs,
            "alpha": alpha,
            "n": len(df),
            "acc_percent": acc_percent(df),
            "avg_blocks": avg_blocks,
            "median_blocks": median_blocks,
            "p90_blocks": p90_blocks,
            "cost_vs_rs64": None if avg_blocks is None else avg_blocks / 64.0,
            "logit_files_read": files_read,
            "run_dir": str(run_dir),
        })
        data_by_label[label] = df
    return pd.DataFrame(records), data_by_label


def common_summary(summary, data_by_label):
    common = None
    for df in data_by_label.values():
        keys = set(df["_key"])
        common = keys if common is None else common & keys
    common = common or set()

    rows = []
    for _, record in summary.iterrows():
        label = record["label"]
        df = data_by_label[label]
        sub = df[df["_key"].isin(common)]
        row = record.to_dict()
        row["common_n"] = len(sub)
        row["common_acc_percent"] = float(sub["_correct"].mean() * 100.0) if len(sub) else None
        rows.append(row)
    return pd.DataFrame(rows), len(common)


def add_pareto_flags(summary, acc_column):
    rows = []
    for _, current in summary.iterrows():
        dominated_by = []
        if pd.notna(current["avg_blocks"]) and pd.notna(current[acc_column]):
            for _, other in summary.iterrows():
                if current["label"] == other["label"]:
                    continue
                if pd.isna(other["avg_blocks"]) or pd.isna(other[acc_column]):
                    continue
                cheaper_or_equal = other["avg_blocks"] <= current["avg_blocks"] + 1e-9
                better_or_equal = other[acc_column] >= current[acc_column] - 1e-9
                strictly_better = (
                    other["avg_blocks"] < current["avg_blocks"] - 1e-9
                    or other[acc_column] > current[acc_column] + 1e-9
                )
                if cheaper_or_equal and better_or_equal and strictly_better:
                    dominated_by.append(other["label"])
        row = current.to_dict()
        row["is_pareto_all"] = len(dominated_by) == 0
        row["dominated_by"] = ",".join(dominated_by[:8])
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate_dynamic_vs_fixed(common):
    fixed = common[common["type"].eq("fixed") & common["avg_blocks"].notna()].sort_values("avg_blocks")
    dynamic = common[common["type"].eq("dynamic_alpha") & common["avg_blocks"].notna()].sort_values("avg_blocks")
    if len(fixed) < 2 or dynamic.empty:
        return pd.DataFrame()

    xs = fixed["avg_blocks"].astype(float).to_list()
    ys = fixed["common_acc_percent"].astype(float).to_list()
    rows = []
    for _, row in dynamic.iterrows():
        x = float(row["avg_blocks"])
        y = float(row["common_acc_percent"])
        yhat = None
        if min(xs) <= x <= max(xs):
            for idx in range(len(xs) - 1):
                if xs[idx] <= x <= xs[idx + 1]:
                    denom = xs[idx + 1] - xs[idx]
                    mix = 0.0 if denom == 0 else (x - xs[idx]) / denom
                    yhat = ys[idx] + mix * (ys[idx + 1] - ys[idx])
                    break
        rows.append({
            "label": row["label"],
            "avg_blocks": x,
            "common_acc_percent": y,
            "fixed_interp_acc_percent": yhat,
            "delta_vs_fixed_interp_pp": None if yhat is None else y - yhat,
            "common_n": row["common_n"],
        })
    return pd.DataFrame(rows)


def maybe_plot(summary, output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fixed = summary[summary["type"].eq("fixed")].sort_values("avg_blocks")
    dynamic = summary[summary["type"].eq("dynamic_alpha")].sort_values("avg_blocks")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
    ax.plot(
        fixed["avg_blocks"],
        fixed["common_acc_percent"],
        marker="o",
        linewidth=2,
        label="fixed rs",
    )
    if not dynamic.empty:
        ax.scatter(
            dynamic["avg_blocks"],
            dynamic["common_acc_percent"],
            s=55,
            marker="s",
            label="dynamic alpha",
        )
    for _, row in fixed.iterrows():
        ax.annotate(
            row["label"],
            (row["avg_blocks"], row["common_acc_percent"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    for _, row in dynamic.iterrows():
        ax.annotate(
            row["label"],
            (row["avg_blocks"], row["common_acc_percent"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Average retrieved blocks per layer")
    ax.set_ylabel("LVBench accuracy (%)")
    ax.set_title("LVBench Cost-Accuracy Pareto")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/mnt/ssd1/mwnoh/LVBench/results/qwen2_5_vl_7b/lvbench",
    )
    parser.add_argument("--sample_fps", default="1.0")
    parser.add_argument("--fixed_rs", default="16 24 32 40 48 56 64")
    parser.add_argument("--fixed_reuse_dir", default=None)
    parser.add_argument("--dynamic_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    fixed_reuse_dir = Path(args.fixed_reuse_dir) if args.fixed_reuse_dir else root / "fixed_rs24_rs32_rs40_rs48_rs56_reuse"
    dynamic_dir = Path(args.dynamic_dir) if args.dynamic_dir else root / "dynamic_mass_min16_max64"
    output_dir = Path(args.output_dir) if args.output_dir else root / "pareto_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    runs.extend(add_fixed_runs(root, fixed_reuse_dir, args.sample_fps, parse_int_list(args.fixed_rs)))
    runs.extend(add_dynamic_runs(dynamic_dir, args.sample_fps))
    if not runs:
        raise FileNotFoundError("No LVBench results.csv files found for the requested runs.")

    summary, data_by_label = build_summary(runs)
    summary = summary.sort_values(["type", "avg_blocks", "alpha"], na_position="last")
    common, common_n = common_summary(summary, data_by_label)
    common = common.sort_values(["type", "avg_blocks", "alpha"], na_position="last")
    pareto = add_pareto_flags(common, "common_acc_percent").sort_values(["avg_blocks", "label"])
    interp = interpolate_dynamic_vs_fixed(common)

    summary.to_csv(output_dir / "lvbench_pareto_summary_each_run.csv", index=False)
    common.to_csv(output_dir / "lvbench_pareto_summary_common_all.csv", index=False)
    pareto.to_csv(output_dir / "lvbench_pareto_all_methods.csv", index=False)
    interp.to_csv(output_dir / "lvbench_dynamic_vs_fixed_interp_common.csv", index=False)

    fixed = common[common["type"].eq("fixed")].copy().sort_values("avg_blocks")
    best = -1.0
    flags = []
    for _, row in fixed.iterrows():
        keep = row["common_acc_percent"] > best + 1e-12
        flags.append(keep)
        if keep:
            best = row["common_acc_percent"]
    fixed["is_pareto_fixed"] = flags
    fixed.to_csv(output_dir / "lvbench_fixed_pareto_curve.csv", index=False)

    plot_path = None
    if not args.no_plot:
        plot_path = maybe_plot(common, output_dir / "lvbench_pareto_curve.png")

    print(f"common_n: {common_n}")
    print(common[["label", "type", "common_n", "common_acc_percent", "avg_blocks", "cost_vs_rs64"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if not interp.empty:
        print("")
        print(interp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"output_dir: {output_dir}")
    if plot_path is not None:
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
