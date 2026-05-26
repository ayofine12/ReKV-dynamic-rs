#!/usr/bin/env python
import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path


MC_TASKS = [
    "1_plotQA",
    "2_needle",
    "3_ego",
    "4_count",
    "5_order",
    "6_anomaly_reco",
    "7_topic_reasoning",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/mnt/ssd1/mwnoh/MLVU/MLVU")
    parser.add_argument("--output", default="data/mlvu/full_mc.json")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=MC_TASKS,
        help="MLVU task json stems to include. Defaults to multiple-choice tasks only.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum video duration in seconds. Longer videos are skipped.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to the current working directory instead of absolute paths.",
    )
    parser.add_argument(
        "--balance-chunks",
        type=int,
        default=None,
        help="Reorder videos so BaseVQA's contiguous chunk split has balanced total duration.",
    )
    return parser.parse_args()


def make_video_path(input_root, task, video_name, relative_paths):
    path = input_root / "video" / task / video_name
    if relative_paths:
        return str(path)
    return str(path.resolve())


def balance_for_contiguous_chunks(items, num_chunks):
    if num_chunks is None or num_chunks <= 1:
        return items

    total = len(items)
    chunk_size = math.ceil(total / num_chunks)
    capacities = [
        max(0, min(chunk_size, total - idx * chunk_size))
        for idx in range(num_chunks)
    ]
    buckets = [[] for _ in range(num_chunks)]
    bucket_durations = [0.0 for _ in range(num_chunks)]

    for item in sorted(items, key=lambda x: float(x.get("duration") or 0), reverse=True):
        candidates = [idx for idx, bucket in enumerate(buckets) if len(bucket) < capacities[idx]]
        idx = min(candidates, key=lambda x: bucket_durations[x])
        buckets[idx].append(item)
        bucket_durations[idx] += float(item.get("duration") or 0)

    print(
        "balanced chunk durations: "
        + ", ".join(f"{idx}={duration:.1f}s" for idx, duration in enumerate(bucket_durations))
    )
    return [item for bucket in buckets for item in bucket]


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_path = Path(args.output)

    grouped = OrderedDict()
    skipped_missing = 0
    skipped_duration = 0
    skipped_open = 0

    for task in args.tasks:
        json_path = input_root / "json" / f"{task}.json"
        with json_path.open() as f:
            samples = json.load(f)

        for idx, sample in enumerate(samples):
            if "candidates" not in sample:
                skipped_open += 1
                continue
            duration = float(sample.get("duration") or 0)
            if args.max_duration is not None and duration > args.max_duration:
                skipped_duration += 1
                continue

            video_name = sample["video"]
            video_path = input_root / "video" / task / video_name
            if not video_path.exists():
                skipped_missing += 1
                continue

            video_id = Path(video_name).stem
            key = (task, video_id)
            if key not in grouped:
                grouped[key] = {
                    "video_id": video_id,
                    "video_path": make_video_path(input_root, task, video_name, args.relative_paths),
                    "duration": duration,
                    "conversations": [],
                }

            grouped[key]["conversations"].append(
                {
                    "uid": f"{task}:{video_id}:{idx}",
                    "question": sample["question"],
                    "choices": sample["candidates"],
                    "answer": sample.get("answer"),
                    "question_type": sample.get("question_type", task.split("_", 1)[-1]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_items = balance_for_contiguous_chunks(list(grouped.values()), args.balance_chunks)
    with output_path.open("w") as f:
        json.dump(output_items, f, indent=2)

    n_videos = len(grouped)
    n_questions = sum(len(item["conversations"]) for item in grouped.values())
    print(f"wrote {output_path}")
    print(f"videos={n_videos} questions={n_questions}")
    print(
        "skipped "
        f"missing={skipped_missing} duration={skipped_duration} open_ended={skipped_open}"
    )


if __name__ == "__main__":
    main()
