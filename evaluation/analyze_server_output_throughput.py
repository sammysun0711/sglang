#!/usr/bin/env python3
"""Measure sustained decode throughput from an SGLang TP0 server log.

The benchmark client metrics are intentionally not used.  Workloads are split
by successful TP0 cache-flush markers.  For each workload section, the actual
resident concurrency is the maximum TP0 ``#running-req`` value unless it is
explicitly supplied in ``--case``.

A full-residency plateau starts when TP0 enters the resident count or when the
``#full token`` counter decreases while residency remains full.  The first
transition record of every plateau is excluded.  Every later full-residency
record, including stalls, is retained with no outlier filtering.

Examples:

  # Inspect every cache-flush-delimited section; infer resident concurrency.
  python3 analyze_server_output_throughput.py /path/to/server.log

  # Attach case metadata to selected sections. Resident defaults to observed
  # maximum residency. The format is SECTION:CONTEXT:REQUESTED[:RESIDENT].
  python3 analyze_server_output_throughput.py /path/to/server.log \
    --case 0:64K:114 --case 1:64K:116 --case 2:64K:118

  # Produce a CSV and a separate Markdown audit containing retained samples.
  python3 analyze_server_output_throughput.py /path/to/server.log \
    --case 0:64K:114 --format csv --output results.csv \
    --details-output retained-samples.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO


CACHE_FLUSH_MARKER = "TP0] Cache flushed successfully!"
DECODE_RE = re.compile(
    r"^\[([^]]+) TP0\] Decode batch, "
    r"#running-req: (\d+), #full token: (\d+), "
    r"full token usage: ([0-9.]+)"
    r".*#prealloc-req: (\d+)"
    r".*gen throughput \(token/s\): ([0-9.]+)"
)


@dataclass(frozen=True)
class DecodeSample:
    line_number: int
    timestamp: datetime
    running_requests: int
    full_tokens: int
    full_token_usage: float
    preallocated_requests: int
    throughput: float


@dataclass(frozen=True)
class CaseSpec:
    section: int
    context: str
    requested: int
    resident: int | None = None


@dataclass(frozen=True)
class CaseResult:
    case: CaseSpec
    resident: int
    plateaus: tuple[tuple[DecodeSample, ...], ...]
    selected: tuple[DecodeSample, ...]
    server_throughput: float
    tps_per_request: float
    tpot_ms: float
    passed: bool


def parse_timestamp(value: str) -> datetime:
    """Parse the timestamp forms currently emitted by SGLang."""
    for timestamp_format in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(value, timestamp_format)
        except ValueError:
            pass
    raise ValueError(f"Unsupported SGLang timestamp: {value!r}")


def parse_sections(path: Path) -> list[list[DecodeSample]]:
    """Return TP0 decode records grouped by successful cache flush."""
    sections: list[list[DecodeSample]] = []
    current: list[DecodeSample] | None = None

    # SGLang/benchmark progress output can contain bare carriage returns. Use
    # only LF as a record boundary so reported line numbers agree with rg,
    # awk, editors, and the physical server-log lines.
    with path.open(encoding="utf-8", errors="replace", newline="\n") as handle:
        for line_number, line in enumerate(handle, start=1):
            if CACHE_FLUSH_MARKER in line:
                current = []
                sections.append(current)
                continue
            if current is None:
                continue

            match = DECODE_RE.search(line)
            if match is None:
                continue
            current.append(
                DecodeSample(
                    line_number=line_number,
                    timestamp=parse_timestamp(match.group(1)),
                    running_requests=int(match.group(2)),
                    full_tokens=int(match.group(3)),
                    full_token_usage=float(match.group(4)),
                    preallocated_requests=int(match.group(5)),
                    throughput=float(match.group(6)),
                )
            )

    return sections


def parse_case(value: str) -> CaseSpec:
    """Parse SECTION:CONTEXT:REQUESTED[:RESIDENT]."""
    fields = value.split(":")
    if len(fields) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "case must be SECTION:CONTEXT:REQUESTED[:RESIDENT], "
            f"got {value!r}"
        )
    try:
        section = int(fields[0])
        requested = int(fields[2])
        resident = int(fields[3]) if len(fields) == 4 else None
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"section, requested, and resident must be integers: {value!r}"
        ) from error
    if section < 0 or requested <= 0 or (resident is not None and resident <= 0):
        raise argparse.ArgumentTypeError(
            f"section must be non-negative and concurrency must be positive: {value!r}"
        )
    if not fields[1]:
        raise argparse.ArgumentTypeError(f"context must not be empty: {value!r}")
    return CaseSpec(section, fields[1], requested, resident)


def infer_cases(sections: list[list[DecodeSample]]) -> list[CaseSpec]:
    """Build generic case labels when the caller supplies no metadata."""
    cases: list[CaseSpec] = []
    for section, rows in enumerate(sections):
        if not rows:
            continue
        resident = max(row.running_requests for row in rows)
        cases.append(CaseSpec(section, f"section-{section}", resident, resident))
    return cases


def select_full_residency_samples(
    rows: list[DecodeSample], resident: int
) -> tuple[tuple[DecodeSample, ...], tuple[DecodeSample, ...]]:
    """Apply the transition-exclusion plateau method used in the reports."""
    plateaus: list[list[DecodeSample]] = []
    plateau: list[DecodeSample] = []
    previous: DecodeSample | None = None

    for row in rows:
        if row.running_requests == resident:
            starts_plateau = (
                previous is None
                or previous.running_requests != resident
                or (plateau and row.full_tokens < plateau[-1].full_tokens)
            )
            if starts_plateau:
                if plateau:
                    plateaus.append(plateau)
                plateau = [row]
            else:
                plateau.append(row)
        elif plateau:
            plateaus.append(plateau)
            plateau = []
        previous = row

    if plateau:
        plateaus.append(plateau)

    frozen_plateaus = tuple(tuple(group) for group in plateaus)
    selected = tuple(row for group in frozen_plateaus for row in group[1:])
    return frozen_plateaus, selected


def analyze_case(
    case: CaseSpec,
    sections: list[list[DecodeSample]],
    minimum_tps: float,
    maximum_tpot_ms: float,
) -> CaseResult:
    if case.section >= len(sections):
        raise ValueError(
            f"Section {case.section} does not exist; log contains "
            f"{len(sections)} cache-flush sections"
        )
    rows = sections[case.section]
    if not rows:
        raise ValueError(f"Section {case.section} contains no TP0 decode records")

    observed_resident = max(row.running_requests for row in rows)
    resident = case.resident if case.resident is not None else observed_resident
    if resident > observed_resident:
        raise ValueError(
            f"Section {case.section}: requested resident {resident}, but maximum "
            f"observed #running-req is {observed_resident}"
        )

    plateaus, selected = select_full_residency_samples(rows, resident)
    if not selected:
        raise ValueError(
            f"Section {case.section}: no retained samples at resident {resident}; "
            "each plateau must contain a transition record plus at least one "
            "measurement record"
        )

    values = [row.throughput for row in selected]
    server_throughput = sum(values) / len(values)
    tps_per_request = server_throughput / resident
    tpot_ms = 1000.0 * resident / server_throughput
    passed = tps_per_request >= minimum_tps and tpot_ms <= maximum_tpot_ms
    return CaseResult(
        case=case,
        resident=resident,
        plateaus=plateaus,
        selected=selected,
        server_throughput=server_throughput,
        tps_per_request=tps_per_request,
        tpot_ms=tpot_ms,
        passed=passed,
    )


def write_markdown(results: Iterable[CaseResult], output: TextIO) -> None:
    print(
        "| Section | Context | Requested | Resident | Plateaus | Samples | "
        "Server tok/s | TPS/request | TPOT ms | Result | Min | Max |",
        file=output,
    )
    print(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        file=output,
    )
    for result in results:
        values = [row.throughput for row in result.selected]
        print(
            f"| {result.case.section} | {result.case.context} | "
            f"{result.case.requested} | {result.resident} | "
            f"{len(result.plateaus)} | {len(result.selected)} | "
            f"{result.server_throughput:.2f} | "
            f"{result.tps_per_request:.3f} | {result.tpot_ms:.3f} | "
            f"{'PASS' if result.passed else 'FAIL'} | "
            f"{min(values):.2f} | {max(values):.2f} |",
            file=output,
        )


def write_csv(results: Iterable[CaseResult], output: TextIO) -> None:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "section",
            "context",
            "requested",
            "resident",
            "plateaus",
            "samples",
            "server_tok_s",
            "tps_per_request",
            "tpot_ms",
            "result",
            "min_tok_s",
            "max_tok_s",
        )
    )
    for result in results:
        values = [row.throughput for row in result.selected]
        writer.writerow(
            (
                result.case.section,
                result.case.context,
                result.case.requested,
                result.resident,
                len(result.plateaus),
                len(result.selected),
                f"{result.server_throughput:.2f}",
                f"{result.tps_per_request:.3f}",
                f"{result.tpot_ms:.3f}",
                "PASS" if result.passed else "FAIL",
                f"{min(values):.2f}",
                f"{max(values):.2f}",
            )
        )


def write_details(results: Iterable[CaseResult], output: TextIO) -> None:
    """Write transition and retained rows for line-by-line auditing."""
    for result in results:
        print(
            f"## Section {result.case.section}: {result.case.context}, "
            f"requested {result.case.requested}, resident {result.resident}",
            file=output,
        )
        print(file=output)
        print(
            "| Plateau | Status | Log line | Timestamp | Running | Full tokens | "
            "Preallocated | Server tok/s |",
            file=output,
        )
        print(
            "|---:|---|---:|---|---:|---:|---:|---:|",
            file=output,
        )
        for plateau_number, plateau in enumerate(result.plateaus, start=1):
            for sample_number, sample in enumerate(plateau):
                status = "excluded transition" if sample_number == 0 else "retained"
                print(
                    f"| {plateau_number} | {status} | {sample.line_number} | "
                    f"{sample.timestamp} | {sample.running_requests} | "
                    f"{sample.full_tokens} | {sample.preallocated_requests} | "
                    f"{sample.throughput:.2f} |",
                    file=output,
                )
        print(file=output)
        selected_sum = sum(row.throughput for row in result.selected)
        print(f"Selected sample count: `{len(result.selected)}`", file=output)
        print(f"Selected throughput sum: `{selected_sum:.2f}`", file=output)
        print(
            f"Arithmetic mean: `{selected_sum:.2f} / {len(result.selected)} = "
            f"{result.server_throughput:.2f} tok/s`",
            file=output,
        )
        print(
            f"TPS/request: `{result.server_throughput:.2f} / {result.resident} = "
            f"{result.tps_per_request:.3f}`",
            file=output,
        )
        print(
            f"TPOT: `1000 * {result.resident} / "
            f"{result.server_throughput:.2f} = {result.tpot_ms:.3f} ms`",
            file=output,
        )
        print(file=output)


def open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline=""), True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze TP0 SGLang decode throughput by full-residency plateau."
    )
    parser.add_argument("server_log", type=Path, help="SGLang server log to analyze")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        type=parse_case,
        metavar="SECTION:CONTEXT:REQUESTED[:RESIDENT]",
        help=(
            "case metadata in server-log section order; repeat for multiple cases. "
            "When RESIDENT is omitted, use maximum observed #running-req. If no "
            "cases are supplied, analyze every non-empty section."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="summary output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="summary output path (default: standard output)",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="optional Markdown audit with transition and retained log records",
    )
    parser.add_argument(
        "--minimum-tps",
        type=float,
        default=40.0,
        help="minimum output tokens/s/request (default: 40)",
    )
    parser.add_argument(
        "--maximum-tpot-ms",
        type=float,
        default=25.0,
        help="maximum derived TPOT in milliseconds (default: 25)",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.server_log.is_file():
        print(f"error: server log does not exist: {args.server_log}", file=sys.stderr)
        return 2
    if args.minimum_tps <= 0 or args.maximum_tpot_ms <= 0:
        print("error: throughput and TPOT thresholds must be positive", file=sys.stderr)
        return 2

    try:
        sections = parse_sections(args.server_log)
        if not sections:
            raise ValueError("No successful TP0 cache-flush markers found")
        cases = args.case or infer_cases(sections)
        if not cases:
            raise ValueError("No non-empty TP0 decode sections found")
        results = [
            analyze_case(
                case,
                sections,
                minimum_tps=args.minimum_tps,
                maximum_tpot_ms=args.maximum_tpot_ms,
            )
            for case in cases
        ]
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    summary_output, close_summary = open_output(args.output)
    try:
        if args.format == "markdown":
            write_markdown(results, summary_output)
        else:
            write_csv(results, summary_output)
    finally:
        if close_summary:
            summary_output.close()

    if args.details_output is not None:
        details_output, close_details = open_output(args.details_output)
        try:
            write_details(results, details_output)
        finally:
            if close_details:
                details_output.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
