#!/usr/bin/env python3
"""Summarize SGLANG_MIMO_PREFILL_ANALYSIS_LOG records from server logs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MARKER = "MIMO_PREFILL_ANALYSIS "


def _median(records: list[dict], key: str) -> float:
    values = [record[key] for record in records if record.get(key) is not None]
    return statistics.median(values) if values else float("nan")


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(errors="replace") as file:
        for line in file:
            marker_pos = line.find(MARKER)
            if marker_pos < 0:
                continue
            payload = line[marker_pos + len(MARKER) :].strip()
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "mimo_prefill_iteration":
                records.append(record)
    records.sort(key=lambda record: record["forward_iter"])
    return records


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _complete_long_requests(records: list[dict]) -> list[dict]:
    completed = []
    current = []
    for record in records:
        current.append(record)
        if not record["contains_last_prefill_chunk"]:
            continue
        query_tokens = sum(item["query_tokens"] for item in current)
        if query_tokens >= 131072:
            gpu_forward_ms = sum(item["gpu_forward_ms"] for item in current)
            causal_kv_pairs = sum(item["causal_kv_pairs"] for item in current)
            completed.append(
                {
                    "chunks": len(current),
                    "query_tokens": query_tokens,
                    "gpu_forward_ms": gpu_forward_ms,
                    "causal_kv_pairs": causal_kv_pairs,
                    "causal_gpair_per_gpu_s": causal_kv_pairs / gpu_forward_ms / 1e6,
                    "schedule_ms": sum(item["schedule_ms"] for item in current),
                    "request_span_ms": (
                        current[-1]["result_end_s"] - current[0]["launch_start_s"]
                    )
                    * 1000.0,
                    "tail_query_tokens": current[-1]["query_tokens"],
                    "tail_gpu_forward_ms": current[-1]["gpu_forward_ms"],
                }
            )
        current = []
    return completed


def _print_summary(
    label: str,
    records: list[dict],
    request_records: list[dict],
    prefix_bin_tokens: int = 0,
) -> None:
    if not records:
        print(f"## {label}\n\nNo analysis records found.\n")
        return

    launch_start = min(record["launch_start_s"] for record in records)
    result_end = max(record["result_end_s"] for record in records)
    wall_seconds = max(result_end - launch_start, 0.0)
    total_query_tokens = sum(record["query_tokens"] for record in records)
    gpu_ms = sum(record["gpu_forward_ms"] for record in records)
    throughput = total_query_tokens / wall_seconds if wall_seconds > 0 else float("nan")
    occupancy = gpu_ms / (wall_seconds * 10.0) if wall_seconds > 0 else float("nan")

    for current, following in zip(records, records[1:]):
        launch_interval_ms = (
            following["launch_start_s"] - current["launch_start_s"]
        ) * 1000.0
        current["next_prefill_launch_interval_ms"] = launch_interval_ms
        current["launch_cadence_minus_gpu_ms"] = (
            launch_interval_ms - current["gpu_forward_ms"]
        )

    print(f"## {label}\n")
    print(
        f"{len(records)} iterations, {total_query_tokens:,} query tokens, "
        f"{throughput:,.2f} token/s over the logged span, "
        f"{occupancy:.2f}% summed TP0 GPU-forward occupancy.\n"
    )
    queue_empty = sum(record["queued_requests"] == 0 for record in records)
    print(
        f"Queue empty on {queue_empty}/{len(records)} iterations; median host phases: "
        f"recv={_fmt(_median(records, 'recv_ms'))} ms, "
        f"input={_fmt(_median(records, 'process_input_ms'))} ms, "
        f"schedule={_fmt(_median(records, 'schedule_ms'))} ms, "
        f"result={_fmt(_median(records, 'result_process_ms'))} ms.\n"
    )
    requests = _complete_long_requests(request_records)
    if requests:
        request_gpu_ms = _median(requests, "gpu_forward_ms")
        print(
            f"{len(requests)} long-prompt completion windows; medians: "
            f"chunks={_fmt(_median(requests, 'chunks'), 1)}, "
            f"GPU forward={_fmt(request_gpu_ms)} ms, "
            f"causal throughput={_fmt(_median(requests, 'causal_gpair_per_gpu_s'))} "
            f"Gpair/GPU-s, "
            f"request span={_fmt(_median(requests, 'request_span_ms'))} ms, "
            f"scheduler total={_fmt(_median(requests, 'schedule_ms'))} ms, "
            f"tail={_fmt(_median(requests, 'tail_query_tokens'), 0)} tokens / "
            f"{_fmt(_median(requests, 'tail_gpu_forward_ms'))} GPU ms.\n"
        )
    print(
        "| Query tokens | Samples | Prefix / bin | GPU forward ms | "
        "GPU ms / 1K query | Causal Gpair / GPU-s | Schedule ms | Submit ms | "
        "Result ms | Next launch ms | Cadence - GPU ms |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    by_query: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        prefix_bin = (
            record["prefix_len_max"] // prefix_bin_tokens * prefix_bin_tokens
            if prefix_bin_tokens > 0
            else -1
        )
        by_query[(record["query_tokens"], prefix_bin)].append(record)

    for (query_tokens, prefix_bin), group in sorted(by_query.items()):
        median_gpu_ms = _median(group, "gpu_forward_ms")
        median_pairs = _median(group, "causal_kv_pairs")
        gpu_ms_per_1k = median_gpu_ms * 1000.0 / query_tokens
        causal_gpair_per_gpu_s = (
            median_pairs / median_gpu_ms / 1e6 if median_gpu_ms > 0 else float("nan")
        )
        prefix_label = (
            f"{prefix_bin:,}-{prefix_bin + prefix_bin_tokens - 1:,}"
            if prefix_bin_tokens > 0
            else _fmt(_median(group, "prefix_tokens"), 0)
        )
        print(
            f"| {query_tokens:,} | {len(group)} | "
            f"{prefix_label} | "
            f"{_fmt(median_gpu_ms)} | {_fmt(gpu_ms_per_1k)} | "
            f"{_fmt(causal_gpair_per_gpu_s)} | "
            f"{_fmt(_median(group, 'schedule_ms'))} | "
            f"{_fmt(_median(group, 'launch_submit_ms'))} | "
            f"{_fmt(_median(group, 'result_process_ms'))} | "
            f"{_fmt(_median(group, 'next_prefill_launch_interval_ms'))} | "
            f"{_fmt(_median(group, 'launch_cadence_minus_gpu_ms'))} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="SGLang server log(s)")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Drop this many analysis iterations from each log (for warmup).",
    )
    parser.add_argument(
        "--min-query-tokens",
        type=int,
        default=0,
        help="Ignore smaller EXTEND records such as startup health checks.",
    )
    parser.add_argument(
        "--drop-first-per-query",
        action="store_true",
        help="Drop the first occurrence of every query shape to reduce JIT effects.",
    )
    parser.add_argument(
        "--prefix-bin-tokens",
        type=int,
        default=0,
        help="Also group rows by max-prefix bins of this size (for example 65536).",
    )
    args = parser.parse_args()

    for path in args.logs:
        request_records = _load(path)
        records = [
            record
            for record in request_records
            if record["query_tokens"] >= args.min_query_tokens
        ]
        if args.skip_first:
            records = records[args.skip_first :]
        if args.drop_first_per_query:
            seen_query_shapes = set()
            filtered_records = []
            for record in records:
                query_tokens = record["query_tokens"]
                if query_tokens in seen_query_shapes:
                    filtered_records.append(record)
                else:
                    seen_query_shapes.add(query_tokens)
            records = filtered_records
        _print_summary(str(path), records, request_records, args.prefix_bin_tokens)


if __name__ == "__main__":
    main()
