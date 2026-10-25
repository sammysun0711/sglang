#!/usr/bin/env python3
"""Run EvalScope GSM8K and gate SGLang acceptance on the same requests."""

# One EvalScope GSM8K run supplies both answer accuracy and SGLang acceptance.
# Defaults: full main test split, 4-shot, each worker's acceptance length >= 4.0.
# Acceptance length follows upstream SGLang and includes the bonus token.
# Uses upstream /server_info averages; no SGLang runtime patches are required.
# Run on a dedicated real-acceptance server. The upstream API does not expose
# simulation settings or raw counts for an exact global weighted mean.
# Reuses EvalScope 1.11.0 or auto-installs it in a dedicated client venv.
# EVALSCOPE_BIN must match that version. AUTO_INSTALL_EVALSCOPE=0 disables setup;
# EVALSCOPE_VENV overrides the managed venv. Requires a real-prefill server.
# Set CHECK_ACCEPTANCE=0 for answer accuracy only. MODEL_PATH defaults to the
# served model. Existing BASE_URL, MODEL_ID, LIMIT, EVAL_BATCH_SIZE, MAX_TOKENS,
# DATASET_*, SEED, RUN_ID, OUTPUT_DIR, and FLUSH_CACHE settings remain supported.
# MIN_GSM8K_ACCURACY is an optional fraction in [0, 1]; unset reports the score.
# DRY_RUN=1 or --dry-run prints the command without server requests or logs.

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

EVALSCOPE_VERSION = "1.11.0"


class ValidationError(ValueError):
    pass


def positive_integer(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a positive integer") from None
    if str(number) != str(value) or number <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return number


def bounded_float(value, name, lower, upper):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be between {lower} and {upper}") from None
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValidationError(f"{name} must be between {lower} and {upper}")
    return number


def boolean(value, name):
    if value not in ("0", "1"):
        raise ValidationError(f"{name} must be 0 or 1")
    return value == "1"


@dataclass
class Config:
    base_url: str
    output_dir: Path
    dataset_dir: Path
    dataset_hub: str
    dataset_args: dict
    model: str | None
    model_id: str | None
    limit: str
    expected_samples: int
    batch_size: int
    max_tokens: int
    seed: int
    flush_cache: bool
    check_acceptance: bool
    min_accept_length: float
    min_accuracy: float | None
    evalscope_bin: str
    evalscope_bin_explicit: bool
    auto_install_evalscope: bool
    evalscope_venv: Path

    @classmethod
    def from_env(cls, env):
        script_dir = Path(__file__).resolve().parent
        # Repository copies share the workspace venv; standalone copies keep it nearby.
        workspace_dir = (
            script_dir.parents[1]
            if script_dir.name == "evaluation" and len(script_dir.parents) > 1
            else script_dir
        )
        run_id = env.get("RUN_ID") or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        limit = env.get("LIMIT", "all")
        full_samples = positive_integer(
            env.get("EXPECTED_SAMPLES", "1319"), "EXPECTED_SAMPLES"
        )
        expected = (
            full_samples
            if limit == "all"
            else min(positive_integer(limit, "LIMIT"), full_samples)
        )
        hub = env.get("DATASET_HUB") or "huggingface"
        default_args = (
            {
                "gsm8k": {
                    "dataset_id": "openai/gsm8k",
                    "subset_list": ["main"],
                    "few_shot_num": 4,
                }
            }
            if hub == "huggingface"
            else {}
        )
        try:
            dataset_args = (
                json.loads(env["DATASET_ARGS"])
                if env.get("DATASET_ARGS")
                else default_args
            )
        except json.JSONDecodeError as error:
            raise ValidationError(f"Invalid DATASET_ARGS JSON: {error}") from error
        if not isinstance(dataset_args, dict):
            raise ValidationError("DATASET_ARGS must be a JSON object")
        gsm8k_args = dataset_args.get("gsm8k", {})
        if not isinstance(gsm8k_args, dict) or gsm8k_args.get(
            "subset_list", ["main"]
        ) != ["main"]:
            raise ValidationError("This runner evaluates GSM8K's main subset")
        min_accuracy = env.get("MIN_GSM8K_ACCURACY")
        seed = env.get("SEED", "12345")
        if not seed.isdecimal():
            raise ValidationError("SEED must be a non-negative integer")
        flush_cache = boolean(env.get("FLUSH_CACHE", "1"), "FLUSH_CACHE")
        check_acceptance = boolean(env.get("CHECK_ACCEPTANCE", "1"), "CHECK_ACCEPTANCE")
        if check_acceptance and not flush_cache:
            raise ValidationError("FLUSH_CACHE=1 is required for the acceptance check")
        return cls(
            base_url=env.get("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/"),
            output_dir=Path(
                env.get("OUTPUT_DIR") or script_dir / "logs" / f"gsm8k_{run_id}"
            ),
            dataset_dir=Path(env.get("DATASET_DIR") or script_dir / "evalscope_cache"),
            dataset_hub=hub,
            dataset_args=dataset_args,
            model=env.get("MODEL_PATH") or None,
            model_id=env.get("MODEL_ID") or None,
            limit=limit,
            expected_samples=expected,
            batch_size=positive_integer(
                env.get("EVAL_BATCH_SIZE", "32"), "EVAL_BATCH_SIZE"
            ),
            max_tokens=positive_integer(env.get("MAX_TOKENS", "8192"), "MAX_TOKENS"),
            seed=int(seed),
            flush_cache=flush_cache,
            check_acceptance=check_acceptance,
            min_accept_length=bounded_float(
                env.get("MIN_ACCEPT_LENGTH", "4.0"),
                "MIN_ACCEPT_LENGTH",
                1,
                float("inf"),
            ),
            min_accuracy=(
                bounded_float(min_accuracy, "MIN_GSM8K_ACCURACY", 0, 1)
                if min_accuracy
                else None
            ),
            evalscope_bin=env.get("EVALSCOPE_BIN") or "evalscope",
            evalscope_bin_explicit=bool(env.get("EVALSCOPE_BIN")),
            auto_install_evalscope=boolean(
                env.get("AUTO_INSTALL_EVALSCOPE", "1"), "AUTO_INSTALL_EVALSCOPE"
            ),
            evalscope_venv=Path(
                env.get("EVALSCOPE_VENV")
                or workspace_dir / ".venvs" / f"evalscope-{EVALSCOPE_VERSION}-mimo-v26"
            ),
        )

    @property
    def server_root(self):
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def evalscope_version(binary):
    executable = shutil.which(str(binary))
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    match = re.search(r"^evalscope\s+(\S+)\s*$", result.stdout, re.MULTILINE)
    return match.group(1) if match else None


def ensure_evalscope(config):
    """Reuse the pinned CLI or install it in a dedicated client environment."""
    candidate = shutil.which(config.evalscope_bin)
    version = evalscope_version(candidate) if candidate else None
    if version == EVALSCOPE_VERSION:
        return candidate
    if config.evalscope_bin_explicit:
        raise ValidationError(
            f"EVALSCOPE_BIN must provide EvalScope {EVALSCOPE_VERSION}; "
            f"observed {version or 'unavailable'}. Unset EVALSCOPE_BIN for automatic setup."
        )

    venv = config.evalscope_venv.expanduser().resolve()
    binary = venv / "bin" / "evalscope"
    if evalscope_version(binary) == EVALSCOPE_VERSION:
        return str(binary)
    if not config.auto_install_evalscope:
        raise ValidationError(
            f"EvalScope {EVALSCOPE_VERSION} is unavailable and AUTO_INSTALL_EVALSCOPE=0. "
            "Install the pinned version or set EVALSCOPE_BIN."
        )
    if venv in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise ValidationError(
            "EVALSCOPE_VENV must be separate from the current Python environment"
        )

    venv.parent.mkdir(parents=True, exist_ok=True)
    install_env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        install_env.pop(name, None)
    # Serialize bootstrap when multiple clients share the managed environment.
    with (venv.parent / f".{venv.name}.install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if evalscope_version(binary) == EVALSCOPE_VERSION:
            return str(binary)
        if not (venv / "pyvenv.cfg").is_file():
            if venv.exists() and any(venv.iterdir()):
                raise ValidationError(
                    "EVALSCOPE_VENV is not an empty directory or a virtual environment"
                )
            print(f"Creating EvalScope client environment: {venv}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv)], check=True, env=install_env
            )
        print(f"Installing evalscope=={EVALSCOPE_VERSION} in {venv}", flush=True)
        subprocess.run(
            [
                str(venv / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--prefix",
                str(venv),
                "--no-user",
                f"evalscope=={EVALSCOPE_VERSION}",
            ],
            check=True,
            env=install_env,
        )
        if evalscope_version(binary) != EVALSCOPE_VERSION:
            raise ValidationError(
                "EvalScope installation did not produce the required version"
            )
    return str(binary)


def request(server_root, endpoint, *, post=False):
    headers = {}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
    req = urllib.request.Request(
        server_root + endpoint, headers=headers, method="POST" if post else "GET"
    )
    with urllib.request.urlopen(req, timeout=70 if post else 15) as response:
        return response.read().decode()


def worker_means(info):
    """Read the ordinary upstream per-worker averages; response order is not DP rank."""
    states = info.get("internal_states", [])
    if (
        not isinstance(states, list)
        or not states
        or len(states) != info.get("dp_size", len(states))
        or any(not isinstance(state, dict) for state in states)
    ):
        raise ValidationError("Incomplete worker statistics in SGLang /server_info")
    return [state.get("avg_spec_accept_length") for state in states]


def require_reset_statistics(info):
    if any(value not in (None, 0) for value in worker_means(info)):
        raise ValidationError(
            "Acceptance statistics were not reset; use an idle dedicated server and FLUSH_CACHE=1"
        )


def preflight(info, config):
    if info.get("disaggregation_mode", "null") != "null":
        raise ValidationError(
            "Combined accuracy checks require a server with real prefill"
        )
    if info.get("status", "ready") != "ready":
        raise ValidationError("SGLang server is not ready")
    if config.check_acceptance:
        if not info.get("speculative_algorithm"):
            raise ValidationError(
                "Enable speculative decoding or set CHECK_ACCEPTANCE=0"
            )
        draft_tokens = info.get("speculative_num_draft_tokens")
        if not isinstance(draft_tokens, int) or config.min_accept_length > draft_tokens:
            raise ValidationError(
                "MIN_ACCEPT_LENGTH exceeds the server's draft block size"
            )
        worker_means(info)


def acceptance_summary(before, after, threshold):
    for key in (
        "model_path",
        "weight_version",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "dp_size",
        "tp_size",
    ):
        if before.get(key) != after.get(key):
            raise ValidationError(
                f"Server configuration changed during evaluation: {key}"
            )
    before_pids, after_pids = before.get("scheduler_pids"), after.get("scheduler_pids")
    if before_pids is not None or after_pids is not None:
        if (
            before_pids is None
            or after_pids is None
            or sorted(before_pids) != sorted(after_pids)
        ):
            raise ValidationError("Scheduler processes changed during evaluation")
    require_reset_statistics(before)
    means = worker_means(after)
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in means
    ):
        raise ValidationError(
            "Missing or invalid acceptance length for a worker; run enough requests to exercise every replica"
        )
    return {
        "passed": all(value >= threshold for value in means),
        "min_accept_length": threshold,
        "mean_worker_accept_length": math.fsum(means) / len(means),
        "min_worker_accept_length": min(means),
        "max_worker_accept_length": max(means),
        "workers": [
            {
                "response_index": index,
                "accept_length": value,
                "passed": value >= threshold,
            }
            for index, value in enumerate(means)
        ],
        "gate": "Every reported worker mean must meet MIN_ACCEPT_LENGTH",
        "aggregation": "Arithmetic mean of per-worker avg_spec_accept_length values, not an exact global weighted mean",
        "source": "Upstream SGLang /server_info internal_states[].avg_spec_accept_length (bonus-inclusive)",
        "scope": "After successful pre-run flush; requires real acceptance, no concurrent inference, and no statistics resets during evaluation",
    }


def gsm8k_result(work_dir, expected_samples, min_accuracy):
    reports = list((work_dir / "reports").rglob("gsm8k.json"))
    reviews = list((work_dir / "reviews").rglob("gsm8k_main.jsonl"))
    if len(reports) != 1 or len(reviews) != 1:
        raise ValidationError(
            "Expected one GSM8K report and one main-split review file"
        )
    report = json.loads(reports[0].read_text())
    execution = report.get("execution_summary", {})
    if (
        report.get("num") != expected_samples
        or execution.get("requested") != expected_samples
        or execution.get("succeeded") != expected_samples
        or execution.get("errored") != 0
        or execution.get("incomplete", True)
    ):
        raise ValidationError(
            f"GSM8K evaluation was incomplete or had errors: {execution}"
        )
    indices, correct = set(), 0
    with reviews[0].open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["index"] in indices:
                raise ValidationError("Duplicate GSM8K review index")
            indices.add(row["index"])
            score = row["sample_score"]["score"]
            value = score["value"]["accuracy"]
            if score.get("status") != "success" or value not in (0.0, 1.0):
                raise ValidationError("Invalid GSM8K review score")
            correct += int(value)
    if len(indices) != expected_samples:
        raise ValidationError("GSM8K review count differs from the requested scope")
    accuracy = correct / expected_samples
    metrics = [
        m
        for m in report.get("metrics", [])
        if m.get("identity", {}).get("name") == "accuracy"
    ]
    if len(metrics) != 1 or abs(metrics[0]["score"] - accuracy) > 0.000051:
        raise ValidationError(
            "GSM8K report score disagrees with the per-example reviews"
        )
    return {
        "accuracy": accuracy,
        "accuracy_percent": 100 * accuracy,
        "correct": correct,
        "total": expected_samples,
        "api_errors": 0,
        "min_accuracy": min_accuracy,
        "threshold_passed": (
            accuracy >= min_accuracy if min_accuracy is not None else None
        ),
        "report": str(reports[0]),
    }


def evalscope_command(config):
    model = config.model or "<model from SGLang server_info>"
    model_id = config.model_id or Path(model.rstrip("/")).name
    command = [
        config.evalscope_bin,
        "eval",
        "--model",
        model,
        "--model-id",
        model_id,
        "--eval-type",
        "openai_api",
        "--api-url",
        config.base_url,
        "--api-key",
        os.environ.get("API_KEY", "EMPTY"),
        "--datasets",
        "gsm8k",
        "--dataset-dir",
        str(config.dataset_dir),
        "--dataset-hub",
        config.dataset_hub,
        "--dataset-args",
        json.dumps(config.dataset_args),
        "--eval-batch-size",
        str(config.batch_size),
        "--seed",
        str(config.seed),
        "--generation-config",
        f"temperature=0.0,top_p=1.0,max_tokens={config.max_tokens},seed={config.seed},stream=False,timeout=1800,retries=1",
        "--work-dir",
        str(config.output_dir / "evalscope"),
        "--no-timestamp",
    ]
    if config.limit != "all":
        command.extend(["--limit", config.limit])
    return command


def run_evalscope(command, log_path):
    with log_path.open("w") as log:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()


def run(config, dry_run=False):
    if dry_run:
        command = evalscope_command(config)
        command[command.index("--api-key") + 1] = "<redacted>"
        print(shlex.join(command))
        print(
            f"Acceptance gate: {config.min_accept_length if config.check_acceptance else 'disabled'}; accuracy threshold: {config.min_accuracy}"
        )
        print(
            f"Required EvalScope: {EVALSCOPE_VERSION}; auto-install: {config.auto_install_evalscope}"
        )
        print("Dry run: no server requests or output files created.")
        return 0
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValidationError(
            "OUTPUT_DIR is not empty; choose a fresh directory for this evaluation"
        )
    initial = json.loads(request(config.server_root, "/server_info"))
    preflight(initial, config)
    config.evalscope_bin = ensure_evalscope(config)
    config.model = (
        config.model or initial.get("served_model_name") or initial.get("model_path")
    )
    if not config.model:
        raise ValidationError("Set MODEL_PATH; the server did not report a model name")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.output_dir / "run_config.json", asdict(config))
    save_json(config.output_dir / "server_info.json", initial)
    (config.output_dir / "eval_metadata.txt").write_text(
        f"evalscope_version={EVALSCOPE_VERSION}\nevalscope_bin={config.evalscope_bin}\n"
        f"result_scope={'full' if config.limit == 'all' else 'debug-only'}\n"
        f"model={config.model}\nexpected_samples={config.expected_samples}\n"
        f"min_accept_length={config.min_accept_length}\nmin_gsm8k_accuracy={config.min_accuracy}\n"
    )
    if config.flush_cache:
        (config.output_dir / "flush_cache.txt").write_text(
            request(config.server_root, "/flush_cache?timeout=60", post=True)
        )
    before = json.loads(request(config.server_root, "/server_info"))
    preflight(before, config)
    if config.check_acceptance:
        require_reset_statistics(before)
    save_json(config.output_dir / "server_info_before_eval.json", before)
    if config.limit != "all":
        print(f"Debug-only run: LIMIT={config.limit}; not a full GSM8K result.")
    started = datetime.now(timezone.utc).isoformat()
    exit_code = run_evalscope(
        evalscope_command(config), config.output_dir / "evalscope_console.log"
    )
    summary = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "result_scope": "full" if config.limit == "all" else "debug-only",
        "evaluation_exit_code": exit_code,
        "gsm8k": None,
        "acceptance": None,
        "checks": {"evalscope_exit": exit_code == 0},
        "errors": [],
    }
    try:
        after = json.loads(request(config.server_root, "/server_info"))
        preflight(after, config)
        save_json(config.output_dir / "server_info_after.json", after)
        if config.check_acceptance:
            summary["acceptance"] = acceptance_summary(
                before, after, config.min_accept_length
            )
            summary["checks"]["acceptance"] = summary["acceptance"]["passed"]
            save_json(
                config.output_dir / "speculative_acceptance.json", summary["acceptance"]
            )
    except (ValidationError, OSError, ValueError, KeyError) as error:
        summary["checks"]["server_statistics"] = False
        summary["errors"].append(str(error))
    try:
        summary["gsm8k"] = gsm8k_result(
            config.output_dir / "evalscope",
            config.expected_samples,
            config.min_accuracy,
        )
        summary["checks"]["gsm8k_complete"] = True
        if config.min_accuracy is not None:
            summary["checks"]["gsm8k_accuracy"] = summary["gsm8k"]["threshold_passed"]
    except (ValidationError, OSError, ValueError, KeyError) as error:
        summary["checks"]["gsm8k_complete"] = False
        summary["errors"].append(str(error))
    summary["passed"] = all(summary["checks"].values())
    save_json(config.output_dir / "gate_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Combined results: {config.output_dir / 'gate_summary.json'}")
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without contacting the server",
    )
    args = parser.parse_args()
    try:
        return run(
            Config.from_env(os.environ),
            args.dry_run or boolean(os.environ.get("DRY_RUN", "0"), "DRY_RUN"),
        )
    except (
        ValidationError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
