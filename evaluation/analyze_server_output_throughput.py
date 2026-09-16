#!/usr/bin/env python3
"""
Decode Benchmark Log Analyzer

Standardized decode throughput analysis for SGLang logs.
Extracts steady-state, fully-loaded throughput data with rigorous multi-stage
filtering to produce reproducible, comparable performance metrics.

Pipeline:
  1. Parse all "Decode batch" lines + detect cache flush section boundaries
  2. TP rank filter (single scheduler)
  3. Split into sections (by cache flush markers)
  4. Per-section full-load filter (running_req >= ceil(target_bs * ratio))
  5. Per-section round detection (time-gap method) + warmup/drain trim
  6. Pool all retained samples → IQR outlier removal
  7. Statistics: Mean / Median / P-tiles / TPS-per-req / TPOT
  8. Optional: per-round breakdown, linear-regression drift check

Usage:
    python decode_bench_analyzer.py <log_file> --target-bs 96
    python decode_bench_analyzer.py <log_file> --target-bs 194 --per-round --drift-check
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DecodeSample:
    line_no: int
    timestamp: float
    tp_rank: int
    dp_rank: int
    running_req: int
    throughput: float
    accept_len: float
    queue_req: int
    section_id: int = -1
    round_id: str = ""


@dataclass
class SectionDetail:
    section_id: int
    resident: int
    total_samples: int
    full_load_samples: int
    num_rounds: int
    after_trim: int
    after_iqr: int = 0
    iqr_lower: float = 0.0
    iqr_upper: float = 0.0


@dataclass
class SectionStats:
    section_id: int
    resident: int
    stats: dict
    iqr_lower: float
    iqr_upper: float
    iqr_removed: int
    removed_outliers: list


@dataclass
class RoundStats:
    label: str
    num_samples: int
    mean: float
    median: float
    std: float
    cov: float
    min_val: float
    max_val: float


@dataclass
class FilteringReport:
    total_decode_lines: int = 0
    after_tp_filter: int = 0
    tp_removed: int = 0
    tp_rank_used: str = ""
    num_sections: int = 0
    section_details: List[SectionDetail] = field(default_factory=list)
    after_full_load: int = 0
    full_load_removed: int = 0
    total_rounds: int = 0
    after_trim: int = 0
    trim_removed: int = 0
    short_rounds_dropped: int = 0
    after_iqr: int = 0
    iqr_removed: int = 0
    removed_outliers: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_DECODE_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?:DP(\d+)\s+)?TP(\d+)(?:\s+EP\d+)?\]\s+"
    r"Decode batch,\s+"
    r"#running-req:\s*(\d+),.*?"
    r"accept len:\s*([\d.]+),.*?"
    r"gen throughput \(token/s\):\s*([\d.]+),\s*"
    r"#queue-req:\s*(\d+)"
)

_LEADING_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+"
)

_CACHE_FLUSH_RE = re.compile(r"Cache flushed successfully")


def _parse_timestamp(bracket_ts: str, leading_ts: Optional[str]) -> float:
    if leading_ts:
        dt = datetime.strptime(leading_ts, "%Y-%m-%d %H:%M:%S.%f")
    else:
        dt = datetime.strptime(bracket_ts, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def parse_log(filepath: str) -> Tuple[List[DecodeSample], List[int]]:
    """Parse log file. Returns (samples, flush_line_nos)."""
    samples = []
    flush_line_nos = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if _CACHE_FLUSH_RE.search(line):
                flush_line_nos.append(line_no)
                continue
            m = _DECODE_RE.search(line)
            if not m:
                continue
            leading = _LEADING_TS_RE.match(line)
            leading_ts = leading.group(1) if leading else None
            ts = _parse_timestamp(m.group(1), leading_ts)
            dp_rank = int(m.group(2)) if m.group(2) else 0
            samples.append(DecodeSample(
                line_no=line_no,
                timestamp=ts,
                tp_rank=int(m.group(3)),
                dp_rank=dp_rank,
                running_req=int(m.group(4)),
                throughput=float(m.group(6)),
                accept_len=float(m.group(5)),
                queue_req=int(m.group(7)),
            ))
    return samples, flush_line_nos


# ---------------------------------------------------------------------------
# Filtering pipeline
# ---------------------------------------------------------------------------


def filter_tp_rank(
    samples: List[DecodeSample],
    tp_rank: Optional[int],
    report: FilteringReport,
) -> List[DecodeSample]:
    ranks = sorted(set(s.tp_rank for s in samples))
    if tp_rank is not None:
        chosen = tp_rank
        if chosen not in ranks:
            print(
                f"WARNING: --tp-rank {chosen} not found. Available: {ranks}",
                file=sys.stderr,
            )
    elif len(ranks) == 1:
        chosen = ranks[0]
    else:
        chosen = ranks[0]
        print(
            f"WARNING: Multiple TP ranks: {ranks}. Using TP{chosen}. "
            f"Override with --tp-rank.",
            file=sys.stderr,
        )
    report.tp_rank_used = f"TP{chosen}"
    filtered = [s for s in samples if s.tp_rank == chosen]
    report.tp_removed = len(samples) - len(filtered)
    report.after_tp_filter = len(filtered)
    return filtered


def split_sections(
    samples: List[DecodeSample],
    flush_line_nos: List[int],
) -> List[List[DecodeSample]]:
    """Split samples into sections by cache flush markers."""
    if not flush_line_nos:
        if samples:
            for s in samples:
                s.section_id = 0
        return [samples] if samples else []

    sections = []
    flush_sorted = sorted(flush_line_nos)

    for s in samples:
        sid = 0
        for fl in flush_sorted:
            if s.line_no > fl:
                sid += 1
        s.section_id = sid

    for sid in range(max(s.section_id for s in samples) + 1):
        sec = [s for s in samples if s.section_id == sid]
        if sec:
            sections.append(sec)

    return sections


def filter_full_load(
    samples: List[DecodeSample],
    target_bs: int,
    ratio: float,
) -> Tuple[List[DecodeSample], int]:
    threshold = math.ceil(target_bs * ratio)
    return [s for s in samples if s.running_req >= threshold], threshold


def split_rounds(
    samples: List[DecodeSample],
    gap_multiplier: float = 5.0,
) -> List[List[DecodeSample]]:
    """Split by time gaps. gap > gap_multiplier × median_interval → new round."""
    if len(samples) < 2:
        return [samples] if samples else []

    intervals = [
        samples[i].timestamp - samples[i - 1].timestamp
        for i in range(1, len(samples))
    ]
    median_iv = float(np.median(intervals))
    threshold = gap_multiplier * median_iv if median_iv > 0 else 999999

    rounds = []
    current = [samples[0]]
    for i in range(1, len(samples)):
        gap = samples[i].timestamp - samples[i - 1].timestamp
        if gap > threshold:
            rounds.append(current)
            current = [samples[i]]
        else:
            current.append(samples[i])
    rounds.append(current)
    return rounds


def trim_rounds(
    rounds: List[List[DecodeSample]],
    warmup_n: int,
    drain_n: int,
    min_round_samples: int = 10,
) -> Tuple[List[DecodeSample], List[List[DecodeSample]], int, int]:
    """Trim warmup/drain from each round. Returns (pooled, rounds, removed, short_dropped)."""
    trimmed_all = []
    trimmed_rounds = []
    total_removed = 0
    short_dropped = 0
    min_samples = max(warmup_n + drain_n + 1, min_round_samples)

    for rnd in rounds:
        if len(rnd) < min_samples:
            total_removed += len(rnd)
            short_dropped += 1
            continue
        end = len(rnd) - drain_n if drain_n > 0 else len(rnd)
        trimmed = rnd[warmup_n:end]
        total_removed += len(rnd) - len(trimmed)
        trimmed_all.extend(trimmed)
        trimmed_rounds.append(trimmed)

    return trimmed_all, trimmed_rounds, total_removed, short_dropped


def _apply_iqr(
    samples: List[DecodeSample],
    k: float,
) -> Tuple[List[DecodeSample], float, float, List[dict]]:
    """Apply IQR filter. Returns (filtered, lower, upper, outlier_dicts)."""
    if not samples:
        return samples, 0.0, 0.0, []

    throughputs = np.array([s.throughput for s in samples])
    q1 = float(np.percentile(throughputs, 25))
    q3 = float(np.percentile(throughputs, 75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    filtered = []
    outliers = []
    for s in samples:
        if lower <= s.throughput <= upper:
            filtered.append(s)
        else:
            reason = "below" if s.throughput < lower else "above"
            bound = lower if s.throughput < lower else upper
            outliers.append({
                "line_no": s.line_no,
                "throughput": round(s.throughput, 2),
                "running_req": s.running_req,
                "section": s.section_id,
                "reason": f"{reason} bound ({bound:.2f})",
            })
    return filtered, lower, upper, outliers


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(samples: List[DecodeSample], target_bs: int) -> dict:
    if not samples:
        return {}
    t = np.array([s.throughput for s in samples])
    mean = float(np.mean(t))
    std = float(np.std(t, ddof=1)) if len(t) > 1 else 0.0
    return {
        "mean": mean,
        "median": float(np.median(t)),
        "std": std,
        "cov": std / mean if mean > 0 else 0.0,
        "min": float(np.min(t)),
        "max": float(np.max(t)),
        "p50": float(np.percentile(t, 50)),
        "p90": float(np.percentile(t, 90)),
        "p95": float(np.percentile(t, 95)),
        "p99": float(np.percentile(t, 99)),
        "valid_samples": len(t),
        "tps_per_request": mean / target_bs if target_bs > 0 else 0.0,
        "tpot_ms": 1000.0 * target_bs / mean if mean > 0 else 0.0,
    }


def compute_round_stats(
    trimmed_rounds: List[List[DecodeSample]],
    labels: List[str],
    iqr_k: float,
) -> List[RoundStats]:
    results = []
    for label, rnd in zip(labels, trimmed_rounds):
        if not rnd:
            continue
        t_arr = np.array([s.throughput for s in rnd])
        q1, q3 = float(np.percentile(t_arr, 25)), float(np.percentile(t_arr, 75))
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        t_clean = t_arr[(t_arr >= lo) & (t_arr <= hi)]
        if len(t_clean) == 0:
            t_clean = t_arr
        mean = float(np.mean(t_clean))
        std = float(np.std(t_clean, ddof=1)) if len(t_clean) > 1 else 0.0
        results.append(RoundStats(
            label=label,
            num_samples=len(t_clean),
            mean=mean,
            median=float(np.median(t_clean)),
            std=std,
            cov=std / mean if mean > 0 else 0.0,
            min_val=float(np.min(t_clean)),
            max_val=float(np.max(t_clean)),
        ))
    return results


def compute_drift(samples: List[DecodeSample]) -> Optional[dict]:
    if len(samples) < 3:
        return None
    t0 = samples[0].timestamp
    x = np.array([s.timestamp - t0 for s in samples])
    y = np.array([s.throughput for s in samples])
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])

    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(x)
    p_value = None
    t_stat = 0.0
    if n > 2 and ss_tot > 0:
        x_var = float(np.sum((x - np.mean(x)) ** 2))
        if x_var > 0:
            se_slope = math.sqrt(ss_res / (n - 2) / x_var)
            if se_slope > 0:
                t_stat = slope / se_slope
                p_value = 2 * (1 - _normal_cdf(abs(t_stat))) if n > 30 else None

    duration_s = float(x[-1] - x[0])
    mean_y = float(np.mean(y))
    significant = p_value is not None and p_value < 0.05

    return {
        "slope": slope,
        "slope_pct_per_min": (slope * 60 / mean_y * 100) if mean_y > 0 else 0.0,
        "r_squared": r_squared,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": significant,
        "duration_s": duration_s,
    }


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def detect_sample_interval(samples: List[DecodeSample]) -> Optional[float]:
    if len(samples) < 2:
        return None
    intervals = [
        samples[i + 1].timestamp - samples[i].timestamp
        for i in range(min(len(samples) - 1, 50))
    ]
    return float(np.median(intervals)) * 1000


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> Tuple[
    FilteringReport, dict, Optional[List[RoundStats]], Optional[dict],
    Optional[float], int, int, Optional[List[SectionStats]],
]:
    samples, flush_line_nos = parse_log(args.log_file)
    if not samples:
        print("ERROR: No 'Decode batch' lines found.", file=sys.stderr)
        sys.exit(1)

    report = FilteringReport()
    report.total_decode_lines = len(samples)
    total_raw = len(samples)

    # Step 1: TP rank filter
    samples = filter_tp_rank(samples, args.tp_rank, report)

    # Step 2: Split into sections
    sections = split_sections(samples, flush_line_nos)
    report.num_sections = len(sections)

    # Step 3-5: Per-section full-load → round detection → trim
    threshold = math.ceil(args.target_bs * args.full_load_ratio)
    all_trimmed = []
    all_trimmed_rounds = []
    all_round_labels = []
    per_section_trimmed = {}
    total_full_load = 0
    total_trim_removed = 0
    total_short_dropped = 0

    for sec in sections:
        sid = sec[0].section_id
        full_load, _ = filter_full_load(sec, args.target_bs, args.full_load_ratio)
        total_full_load += len(full_load)
        resident = max(s.running_req for s in sec) if sec else 0

        if not full_load:
            report.section_details.append(SectionDetail(
                section_id=sid, resident=resident,
                total_samples=len(sec), full_load_samples=0,
                num_rounds=0, after_trim=0,
            ))
            continue

        rounds = split_rounds(full_load, gap_multiplier=args.gap_multiplier)
        trimmed_pool, trimmed_rounds, removed, short_dropped = trim_rounds(
            rounds, args.warmup_n, args.drain_n, args.min_round_samples,
        )
        total_trim_removed += removed
        total_short_dropped += short_dropped

        for ri, rnd in enumerate(trimmed_rounds):
            label = f"S{sid}-R{ri}" if report.num_sections > 1 else f"R{ri}"
            for s in rnd:
                s.round_id = label
            all_round_labels.append(label)
        all_trimmed.extend(trimmed_pool)
        all_trimmed_rounds.extend(trimmed_rounds)
        per_section_trimmed[sid] = trimmed_pool

        report.section_details.append(SectionDetail(
            section_id=sid, resident=resident,
            total_samples=len(sec), full_load_samples=len(full_load),
            num_rounds=len(rounds), after_trim=len(trimmed_pool),
        ))

    report.after_full_load = total_full_load
    report.full_load_removed = report.after_tp_filter - total_full_load
    report.total_rounds = sum(sd.num_rounds for sd in report.section_details)
    report.after_trim = len(all_trimmed)
    report.trim_removed = total_trim_removed
    report.short_rounds_dropped = total_short_dropped

    if not all_trimmed:
        print("ERROR: No samples after full-load filter + trim.", file=sys.stderr)
        sys.exit(1)

    # Step 6: IQR — always per-section (different sections may have different
    # bs/inp_len, so their throughput distributions differ fundamentally)
    final_samples = []
    all_outliers = []
    section_stats_list = []
    for sd in report.section_details:
        sid = sd.section_id
        sec_samples = per_section_trimmed.get(sid, [])
        if not sec_samples:
            continue
        sec_filtered, sec_lo, sec_hi, sec_outliers = _apply_iqr(
            sec_samples, args.iqr_k,
        )
        sd.after_iqr = len(sec_filtered)
        sd.iqr_lower = sec_lo
        sd.iqr_upper = sec_hi
        final_samples.extend(sec_filtered)
        all_outliers.extend(sec_outliers)
        if sec_filtered:
            sec_stats = compute_stats(sec_filtered, args.target_bs)
            section_stats_list.append(SectionStats(
                section_id=sid,
                resident=sd.resident,
                stats=sec_stats,
                iqr_lower=sec_lo,
                iqr_upper=sec_hi,
                iqr_removed=len(sec_samples) - len(sec_filtered),
                removed_outliers=sec_outliers,
            ))

    report.iqr_removed = len(all_trimmed) - len(final_samples)
    report.after_iqr = len(final_samples)
    report.removed_outliers = all_outliers

    if not final_samples:
        print("ERROR: No samples after IQR removal.", file=sys.stderr)
        sys.exit(1)

    # Step 7: Stats
    stats = compute_stats(final_samples, args.target_bs)
    interval_ms = detect_sample_interval(final_samples)

    # Step 8: Optional
    round_stats = None
    if args.per_round and all_trimmed_rounds:
        round_stats = compute_round_stats(
            all_trimmed_rounds, all_round_labels, args.iqr_k,
        )

    drift = None
    if args.drift_check:
        drift = compute_drift(final_samples)

    return report, stats, round_stats, drift, interval_ms, total_raw, threshold, section_stats_list


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_text_report(
    filepath: str,
    args: argparse.Namespace,
    report: FilteringReport,
    stats: dict,
    threshold: int,
    round_stats: Optional[List[RoundStats]],
    drift: Optional[dict],
    interval_ms: Optional[float],
    total_raw: int,
    section_stats_list: Optional[List[SectionStats]] = None,
) -> str:
    lines = []
    W = 80

    lines.append("=" * W)
    lines.append("Decode Benchmark Analysis Report".center(W))
    lines.append("=" * W)
    lines.append("")
    lines.append(f"  Log File:             {filepath}")
    lines.append(f"  Analysis Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Target Batch Size:    {args.target_bs}")
    lines.append(
        f"  Full-Load Threshold:  >= {threshold} "
        f"(ceil({args.target_bs} * {args.full_load_ratio}))"
    )
    lines.append(f"  IQR Coefficient (k):  {args.iqr_k}")
    lines.append(f"  Warmup / Drain Trim:  {args.warmup_n} / {args.drain_n}")
    lines.append(f"  Gap Multiplier:       {args.gap_multiplier}")
    lines.append("")

    # Section summary
    if report.num_sections > 1 or (report.section_details and report.section_details[0].num_rounds > 0):
        lines.append("-" * W)
        lines.append("  Section Summary" + (
            f"  ({report.num_sections} sections detected by cache flush markers)"
            if report.num_sections > 1 else "  (1 section, no cache flush markers)"
        ))
        lines.append("-" * W)
        hdr = "  {:>4s}  {:>8s}  {:>6s}  {:>9s}  {:>6s}  {:>10s}"
        lines.append(hdr.format("Sec", "Resident", "Total", "Full-Load", "Rounds", "After Trim"))
        lines.append("  " + "-" * (W - 4))
        for sd in report.section_details:
            lines.append(hdr.format(
                f"#{sd.section_id}",
                str(sd.resident),
                str(sd.total_samples),
                str(sd.full_load_samples),
                str(sd.num_rounds),
                str(sd.after_trim),
            ))
        lines.append("")

    # Filtering pipeline
    lines.append("-" * W)
    lines.append("  Filtering Pipeline")
    lines.append("-" * W)
    fmt = "  {:<35s} {:>6s}  {:>7s}  {}"
    lines.append(fmt.format("Step", "Kept", "Removed", ""))
    lines.append("  " + "-" * (W - 4))
    lines.append(fmt.format(
        "Raw Decode lines", str(report.total_decode_lines), "-", "",
    ))
    lines.append(fmt.format(
        f"TP rank filter ({report.tp_rank_used})",
        str(report.after_tp_filter), str(report.tp_removed), "",
    ))
    lines.append(fmt.format(
        f"Full-load filter (>={threshold})",
        str(report.after_full_load), str(report.full_load_removed), "",
    ))
    rnd_note = f"{report.total_rounds} round(s)"
    if report.short_rounds_dropped > 0:
        rnd_note += f", {report.short_rounds_dropped} short dropped"
    lines.append(fmt.format(
        f"Warmup/Drain trim ({args.warmup_n}/{args.drain_n})",
        str(report.after_trim), str(report.trim_removed), rnd_note,
    ))
    iqr_sections_with_data = [sd for sd in report.section_details if sd.after_iqr > 0 or sd.after_trim > 0]
    if len(iqr_sections_with_data) == 1:
        sd0 = iqr_sections_with_data[0]
        iqr_note = f"[{sd0.iqr_lower:.2f}, {sd0.iqr_upper:.2f}]"
    else:
        iqr_note = "per-section IQR"
    lines.append(fmt.format(
        "IQR outlier removal",
        str(report.after_iqr), str(report.iqr_removed), iqr_note,
    ))
    lines.append("  " + "-" * (W - 4))
    if len(iqr_sections_with_data) > 1:
        for sd in iqr_sections_with_data:
            removed = sd.after_trim - sd.after_iqr
            lines.append(
                f"    S{sd.section_id} [bs={sd.resident}]: "
                f"IQR [{sd.iqr_lower:.2f}, {sd.iqr_upper:.2f}], "
                f"kept {sd.after_iqr}, removed {removed}"
            )

    if report.removed_outliers:
        lines.append("")
        lines.append("  Removed outliers:")
        for o in report.removed_outliers:
            sec_tag = f"S{o['section']} " if report.num_sections > 1 else ""
            lines.append(
                f"    [{sec_tag}line {o['line_no']}] "
                f"throughput={o['throughput']:.2f} "
                f"(running={o['running_req']}, {o['reason']})"
            )
    lines.append("")

    # Throughput statistics
    if stats:
        lines.append("-" * W)
        lines.append("  Throughput Statistics")
        lines.append("-" * W)
        lines.append(f"  Server Throughput (token/s)")
        lines.append(f"    Mean:          {stats['mean']:>10.2f}")
        lines.append(f"    Median:        {stats['median']:>10.2f}  <-- recommended")
        lines.append(f"    Std:           {stats['std']:>10.2f}")
        lines.append(f"    CoV:           {stats['cov']*100:>9.2f}%")
        lines.append(f"    Min:           {stats['min']:>10.2f}")
        lines.append(f"    Max:           {stats['max']:>10.2f}")
        lines.append(f"    P90:           {stats['p90']:>10.2f}")
        lines.append(f"    P95:           {stats['p95']:>10.2f}")
        lines.append(f"    P99:           {stats['p99']:>10.2f}")
        lines.append("")
        lines.append(f"  Derived Metrics (target_bs={args.target_bs})")
        lines.append(f"    TPS/request:   {stats['tps_per_request']:>10.3f}")
        lines.append(f"    TPOT:          {stats['tpot_ms']:>10.3f} ms")

        if hasattr(args, 'min_tps') and args.min_tps is not None:
            tps_ok = stats['tps_per_request'] >= args.min_tps
            lines.append(f"    TPS threshold: {'>= ' + str(args.min_tps):<10s}  "
                         f"{'PASS' if tps_ok else '** FAIL **'}")
        if hasattr(args, 'max_tpot') and args.max_tpot is not None:
            tpot_ok = stats['tpot_ms'] <= args.max_tpot
            lines.append(f"    TPOT threshold: {'<= ' + str(args.max_tpot):<10s} "
                         f"{'PASS' if tpot_ok else '** FAIL **'}")

        lines.append("")
        lines.append(
            f"  Valid Samples: {stats['valid_samples']} / {report.total_decode_lines} "
            f"({stats['valid_samples']/report.total_decode_lines*100:.1f}%)"
        )

        if interval_ms is not None and interval_ms < 100:
            lines.append("")
            lines.append(
                f"  CAVEAT: interval ~{interval_ms:.0f}ms (decode_log_interval ≈ 1). "
                f"Samples are highly autocorrelated."
            )
    else:
        lines.append("  No valid samples after filtering!")
    lines.append("")

    # Per-section statistics (always show when multiple sections; --per-section for single)
    if section_stats_list and (len(section_stats_list) > 1 or args.per_section):
        lines.append("-" * W)
        lines.append("  Per-Section Statistics (independent IQR per section)")
        lines.append("-" * W)
        for ss in section_stats_list:
            s = ss.stats
            lines.append(
                f"  Section #{ss.section_id}  [bs={ss.resident}]  "
                f"(IQR [{ss.iqr_lower:.2f}, {ss.iqr_upper:.2f}], "
                f"removed {ss.iqr_removed} outliers, kept {s['valid_samples']})"
            )
            lines.append(f"    Server Throughput (token/s)")
            lines.append(f"      Mean:          {s['mean']:>10.2f}")
            lines.append(f"      Median:        {s['median']:>10.2f}  <-- recommended")
            lines.append(f"      Std:           {s['std']:>10.2f}")
            lines.append(f"      CoV:           {s['cov']*100:>9.2f}%")
            lines.append(f"      Min:           {s['min']:>10.2f}")
            lines.append(f"      Max:           {s['max']:>10.2f}")
            lines.append(f"      P90:           {s['p90']:>10.2f}")
            lines.append(f"      P95:           {s['p95']:>10.2f}")
            lines.append(f"      P99:           {s['p99']:>10.2f}")
            lines.append(f"    Derived Metrics (target_bs={args.target_bs})")
            lines.append(f"      TPS/request:   {s['tps_per_request']:>10.3f}")
            lines.append(f"      TPOT:          {s['tpot_ms']:>10.3f} ms")
            if hasattr(args, 'min_tps') and args.min_tps is not None:
                tps_ok = s['tps_per_request'] >= args.min_tps
                lines.append(f"      TPS threshold: {'>= ' + str(args.min_tps):<10s}  "
                             f"{'PASS' if tps_ok else '** FAIL **'}")
            if hasattr(args, 'max_tpot') and args.max_tpot is not None:
                tpot_ok = s['tpot_ms'] <= args.max_tpot
                lines.append(f"      TPOT threshold: {'<= ' + str(args.max_tpot):<10s} "
                             f"{'PASS' if tpot_ok else '** FAIL **'}")
            lines.append("")
        if len(section_stats_list) > 1:
            medians = [ss.stats['median'] for ss in section_stats_list]
            med_of_med = float(np.median(medians))
            inter_std = float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0
            inter_mean = float(np.mean(medians))
            inter_cov = inter_std / inter_mean * 100 if inter_mean > 0 else 0.0
            lines.append(f"  Cross-section: median-of-medians={med_of_med:.2f}, "
                         f"inter-section CoV={inter_cov:.2f}%")
            lines.append("")

    # Per-round statistics
    if round_stats:
        lines.append("-" * W)
        lines.append("  Per-Round Statistics")
        lines.append("-" * W)
        hdr = "  {:>8s}  {:>4s}  {:>10s}  {:>10s}  {:>8s}  {:>6s}  {:>10s}  {:>10s}"
        lines.append(hdr.format(
            "Round", "N", "Mean", "Median", "Std", "CoV%", "Min", "Max",
        ))
        lines.append("  " + "-" * (W - 4))
        for rs in round_stats:
            lines.append(hdr.format(
                rs.label,
                str(rs.num_samples),
                f"{rs.mean:.2f}",
                f"{rs.median:.2f}",
                f"{rs.std:.2f}",
                f"{rs.cov*100:.1f}",
                f"{rs.min_val:.2f}",
                f"{rs.max_val:.2f}",
            ))
        if len(round_stats) > 1:
            medians = [rs.median for rs in round_stats]
            med_of_med = float(np.median(medians))
            inter_std = float(np.std(medians, ddof=1))
            inter_cov = inter_std / float(np.mean(medians)) if np.mean(medians) > 0 else 0.0
            lines.append("  " + "-" * (W - 4))
            lines.append(
                f"  Inter-round: median-of-medians={med_of_med:.2f}, "
                f"CoV={inter_cov*100:.2f}%"
            )
        lines.append("")

    # Drift analysis
    if drift is not None:
        lines.append("-" * W)
        lines.append("  Drift Analysis (linear regression: throughput vs. time)")
        lines.append("-" * W)
        lines.append(
            f"  Slope: {drift['slope']:.4f} tok/s per second "
            f"({drift['slope_pct_per_min']:+.4f}%/min)"
        )
        lines.append(f"  R² = {drift['r_squared']:.4f}, duration = {drift['duration_s']:.1f}s")
        if drift["p_value"] is not None:
            lines.append(f"  p-value = {drift['p_value']:.6f}")
        else:
            lines.append("  p-value = N/A (insufficient samples)")
        lines.append("")
        if drift["significant"] and drift["slope"] < 0:
            lines.append(
                "  *** WARNING: Significant performance DEGRADATION detected ***"
            )
        elif drift["significant"] and drift["slope"] > 0:
            lines.append(
                "  NOTE: Significant upward trend (possible warmup effect)."
            )
        else:
            lines.append("  No significant drift detected.")
        lines.append("")

    lines.append("=" * W)
    return "\n".join(lines)


def format_json_report(
    filepath: str,
    args: argparse.Namespace,
    report: FilteringReport,
    stats: dict,
    threshold: int,
    round_stats: Optional[List[RoundStats]],
    drift: Optional[dict],
    interval_ms: Optional[float],
    total_raw: int,
    section_stats_list: Optional[List[SectionStats]] = None,
) -> str:
    def _round_floats(d):
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}

    result = {
        "config": {
            "log_file": filepath,
            "analysis_time": datetime.now().isoformat(),
            "target_bs": args.target_bs,
            "full_load_ratio": args.full_load_ratio,
            "full_load_threshold": threshold,
            "iqr_k": args.iqr_k,
            "warmup_n": args.warmup_n,
            "drain_n": args.drain_n,
            "gap_multiplier": args.gap_multiplier,
        },
        "filtering": {
            "total_decode_lines": report.total_decode_lines,
            "after_tp_filter": report.after_tp_filter,
            "tp_rank_used": report.tp_rank_used,
            "num_sections": report.num_sections,
            "sections": [asdict(sd) for sd in report.section_details],
            "after_full_load": report.after_full_load,
            "total_rounds": report.total_rounds,
            "short_rounds_dropped": report.short_rounds_dropped,
            "after_trim": report.after_trim,
            "after_iqr": report.after_iqr,
            "iqr_per_section": [
                {
                    "section_id": sd.section_id,
                    "lower": round(sd.iqr_lower, 2),
                    "upper": round(sd.iqr_upper, 2),
                    "removed": sd.after_trim - sd.after_iqr,
                }
                for sd in report.section_details
                if sd.after_trim > 0
            ],
            "removed_outliers": report.removed_outliers,
        },
        "statistics": _round_floats(stats),
        "sample_interval_ms": round(interval_ms, 1) if interval_ms else None,
        "autocorrelation_warning": interval_ms is not None and interval_ms < 100,
    }

    if hasattr(args, 'min_tps') and args.min_tps is not None:
        result["pass_fail"] = {}
        result["pass_fail"]["tps_per_request"] = {
            "threshold": args.min_tps,
            "value": round(stats.get("tps_per_request", 0), 4),
            "pass": stats.get("tps_per_request", 0) >= args.min_tps,
        }
    if hasattr(args, 'max_tpot') and args.max_tpot is not None:
        if "pass_fail" not in result:
            result["pass_fail"] = {}
        result["pass_fail"]["tpot_ms"] = {
            "threshold": args.max_tpot,
            "value": round(stats.get("tpot_ms", 0), 4),
            "pass": stats.get("tpot_ms", 0) <= args.max_tpot,
        }

    if section_stats_list:
        result["per_section"] = [
            {
                "section_id": ss.section_id,
                "resident": ss.resident,
                "statistics": _round_floats(ss.stats),
                "iqr_bounds": {"lower": round(ss.iqr_lower, 2), "upper": round(ss.iqr_upper, 2)},
                "iqr_removed": ss.iqr_removed,
            }
            for ss in section_stats_list
        ]
    if round_stats is not None:
        result["per_round"] = [asdict(rs) for rs in round_stats]
    if drift is not None:
        result["drift"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in drift.items()
            if v is not None
        }
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Decode Benchmark Log Analyzer — "
        "standardized decode throughput analysis for SGLang logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log_file", help="Path to the decode log file")
    parser.add_argument(
        "--target-bs", type=int, required=True,
        help="Target batch size (required).",
    )
    parser.add_argument(
        "--full-load-ratio", type=float, default=1.00,
        help="Full-load threshold ratio (default: 1.00, i.e. running_req must equal target_bs).",
    )
    parser.add_argument(
        "--iqr-k", type=float, default=1.0,
        help="IQR coefficient for outlier removal (default: 1.0).",
    )
    parser.add_argument(
        "--warmup-n", type=int, default=1,
        help="Samples to trim from start of each round (default: 1).",
    )
    parser.add_argument(
        "--drain-n", type=int, default=1,
        help="Samples to trim from end of each round (default: 1).",
    )
    parser.add_argument(
        "--gap-multiplier", type=float, default=5.0,
        help="Time gap multiplier for round splitting (default: 5.0).",
    )
    parser.add_argument(
        "--min-round-samples", type=int, default=10,
        help="Minimum samples per round after trim; shorter rounds are dropped (default: 10).",
    )
    parser.add_argument(
        "--per-section", action="store_true",
        help="Report each cache-flush section independently (separate IQR + stats).",
    )
    parser.add_argument(
        "--per-round", action="store_true",
        help="Enable per-round statistics.",
    )
    parser.add_argument(
        "--drift-check", action="store_true",
        help="Enable linear regression drift detection.",
    )
    parser.add_argument(
        "--tp-rank", type=int, default=None,
        help="TP rank to filter (default: auto).",
    )
    parser.add_argument(
        "--min-tps", type=float, default=None,
        help="TPS/request pass threshold (optional).",
    )
    parser.add_argument(
        "--max-tpot", type=float, default=None,
        help="TPOT (ms) pass threshold (optional).",
    )
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="Output format (default: text).",
    )

    args = parser.parse_args()

    report, stats, round_stats, drift, interval_ms, total_raw, threshold, \
        section_stats_list = run_pipeline(args)

    if args.output == "json":
        print(format_json_report(
            args.log_file, args, report, stats, threshold,
            round_stats, drift, interval_ms, total_raw, section_stats_list,
        ))
    else:
        print(format_text_report(
            args.log_file, args, report, stats, threshold,
            round_stats, drift, interval_ms, total_raw, section_stats_list,
        ))


if __name__ == "__main__":
    main()
