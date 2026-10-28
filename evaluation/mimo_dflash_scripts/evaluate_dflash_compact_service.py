"""Retain every synthetic service result before applying the TPOT constraint."""

import json
import subprocess
from pathlib import Path


def evaluate_service(url, output, args, env, source_root):
    output = Path(output)
    report = {
        "completed_requests": 0,
        "cases": [],
        "latency_rule": "median TPOT <= 25 ms",
    }
    for concurrency in args.service_concurrencies:
        observer_path = output / "service-observation/resident-graphs.jsonl"
        before_lines = (
            len(observer_path.read_text().splitlines()) if observer_path.exists() else 0
        )
        destination = output / f"service-c{concurrency}.jsonl"
        command = [
            "python",
            str(Path(__file__).resolve().parents[2] / "python/sglang/bench_serving.py"),
            "--backend",
            "sglang",
            "--model",
            env["MODEL"],
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dataset-name",
            "random",
            "--dataset-path",
            env["DATASET_PATH"],
            "--random-input-len",
            str(args.service_input_len),
            "--random-output-len",
            str(args.service_output_len),
            "--random-range-ratio",
            "1.0",
            "--temperature",
            "0",
            "--num-prompts",
            str(args.service_waves * concurrency),
            "--warmup-requests",
            str(concurrency),
            "--max-concurrency",
            str(concurrency),
            "--tokenize-prompt",
            "--fake-prefill",
            "--flush-cache",
            "--seed",
            "12345",
            "--output-details",
            "--output-file",
            str(destination),
        ]
        (output / f"service-c{concurrency}-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        with (output / f"service-c{concurrency}.log").open("w") as log:
            subprocess.run(
                command,
                env=env,
                cwd=source_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        value = json.loads(destination.read_text().splitlines()[-1])
        expected = args.service_waves * concurrency
        complete = value["completed"] == expected and all(
            n == args.service_output_len for n in value["output_lens"]
        )
        no_errors = not any(value["errors"])
        no_retractions = not any(value["num_retractions"])
        acceptance = value["accept_length"]
        acceptance_passed = abs(acceptance - 4.0) < 1e-6
        case = {
            "concurrency": concurrency,
            "input_len": args.service_input_len,
            "output_len": args.service_output_len,
            "completed": value["completed"],
            "complete": complete,
            "no_http_errors": no_errors,
            "retractions": sum(value["num_retractions"]),
            "simulated_acceptance": acceptance,
            "acceptance_passed": acceptance_passed,
            "output_throughput": value["output_throughput"],
            "median_tpot_ms": value["median_tpot_ms"],
            "p99_tpot_ms": value["p99_tpot_ms"],
            "median_ttft_ms": value["median_ttft_ms"],
            "latency_passed": value["median_tpot_ms"] <= 25,
        }
        samples = (
            [
                json.loads(line)
                for line in observer_path.read_text().splitlines()[before_lines:]
            ]
            if observer_path.exists()
            else []
        )
        case["observed_max_residency"] = max(
            (v["resident_batch"] for v in samples), default=0
        )
        case["exact_resident_samples"] = sum(
            v["resident_batch"] == concurrency for v in samples
        )
        case["graph_batch_pairs"] = sorted(
            {
                (v["resident_batch"], v["target_graph_batch"], v["draft_graph_batch"])
                for v in samples
            }
        )
        case["residency_qualified"] = (
            case["observed_max_residency"] >= concurrency
            and case["exact_resident_samples"] >= 3
        )
        (output / f"service-c{concurrency}-graph-samples.json").write_text(
            json.dumps(samples, indent=2) + "\n"
        )
        report["cases"].append(case)
        report["completed_requests"] += value["completed"]
        (output / "service-summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"C{concurrency}: {value['completed']}/{expected}, TPS={value['output_throughput']:.2f}, median TPOT={value['median_tpot_ms']:.3f} ms",
            flush=True,
        )
        if not complete or not no_errors or not no_retractions or not acceptance_passed:
            raise RuntimeError(f"synthetic service requests failed: {case}")
    report["passing_cases"] = [c for c in report["cases"] if c["latency_passed"]]
    report["best_passing_case"] = max(
        report["passing_cases"], key=lambda c: c["output_throughput"], default=None
    )
    report["best_residency_qualified_case"] = max(
        (c for c in report["passing_cases"] if c["residency_qualified"]),
        key=lambda c: c["output_throughput"],
        default=None,
    )
    report["scope"] = (
        "synthetic acceptance 4; HTTP elapsed TPOT; resident graph samples retained separately; fresh-run variability and real workload capacity require separate qualification"
    )
    (output / "service-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
