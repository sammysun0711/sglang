"""Check replica isolation in the evaluation log analyzer."""

import json
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ANALYZER = (
    Path(__file__).resolve().parents[3]
    / "evaluation"
    / "analyze_server_output_throughput.py"
)


def run_analyzer(tmp_path, ranks, *options):
    log = tmp_path / "server.log"
    log.write_text(
        "".join(
            f"[2026-10-25 00:00:{second:02d} {rank}TP0] Decode batch, "
            f"#running-req: 64, accept len: 4.00, "
            f"gen throughput (token/s): {rate}, #queue-req: 64\n"
            for second in range(20)
            for rank, rate in ranks
        )
    )
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(log),
            "--target-bs",
            "64",
            "--exact-bs",
            "--output",
            "json",
            *options,
        ],
        capture_output=True,
        text=True,
    )


def test_multiple_replicas_require_selection(tmp_path):
    result = run_analyzer(tmp_path, [("DP0 ", 3200), ("DP1 ", 6400)])
    assert result.returncode != 0
    assert "Select --dp-rank" in result.stderr


def test_selected_replica_has_its_own_rate_and_request_latency(tmp_path):
    result = run_analyzer(tmp_path, [("DP0 ", 3200), ("DP1 ", 6400)], "--dp-rank", "1")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["filtering"]["dp_rank_used"] == "DP1"
    assert report["filtering"]["after_dp_filter"] == 20
    assert report["statistics"]["median"] == 6400
    assert report["statistics"]["tps_per_request"] == 100
    assert report["statistics"]["tpot_ms"] == 10


def test_legacy_single_replica_log_needs_no_dp_option(tmp_path):
    result = run_analyzer(tmp_path, [("", 3200)])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["statistics"]["median"] == 3200


def test_missing_replica_is_an_error(tmp_path):
    result = run_analyzer(tmp_path, [("DP0 ", 3200)], "--dp-rank", "1")
    assert result.returncode != 0
    assert "DP1 not found" in result.stderr
