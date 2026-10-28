"""Actual chunked-prefill memory workload, distinct from synthetic service."""

import concurrent.futures
import json
import time
from pathlib import Path

import requests


def evaluate_real_capacity(url, output, max_running):
    output = Path(output)
    destination = output / "real-capacity-responses"
    destination.mkdir()
    report = {"completed_requests": 0, "cases": []}
    cases = [(1, 65536), (1, 262144), (1, 1047488), (min(16, max_running), 65536)]

    def generate(index, concurrency, length):
        ids = ([791, 6864, 315, 9822, 374] * ((length + 4) // 5))[:length]
        rid = f"real-capacity-c{concurrency}-p{length}-{index}"
        with requests.Session() as client:
            client.trust_env = False
            started = time.monotonic()
            response = client.post(
                url + "/generate",
                json={
                    "rid": rid,
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                        "ignore_eos": True,
                    },
                },
                timeout=3600,
            )
            response.raise_for_status()
            value = response.json()
        (destination / f"{rid}.json").write_text(json.dumps(value, indent=2) + "\n")
        meta = value["meta_info"]
        assert meta["prompt_tokens"] == length, meta
        assert meta["completion_tokens"] == 1024, meta
        assert meta["num_retractions"] == 0, meta
        assert meta["spec_verify_ct"] > 0, meta
        return {
            "rid": rid,
            "prompt_tokens": length,
            "completion_tokens": meta["completion_tokens"],
            "verify_steps": meta["spec_verify_ct"],
            "accepted_drafts": meta["spec_num_correct_drafts"],
            "elapsed_seconds": time.monotonic() - started,
        }

    with requests.Session() as client:
        client.trust_env = False
        for concurrency, length in cases:
            flush = client.post(
                url + "/flush_cache", params={"timeout": 30}, timeout=35
            )
            flush.raise_for_status()
            observer = output / "service-observation/resident-graphs.jsonl"
            before = len(observer.read_text().splitlines()) if observer.exists() else 0
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                records = list(
                    executor.map(
                        lambda index, concurrency=concurrency, length=length: generate(
                            index, concurrency, length
                        ),
                        range(concurrency),
                    )
                )
            samples = [
                json.loads(line) for line in observer.read_text().splitlines()[before:]
            ]
            case = {
                "concurrency": concurrency,
                "input_length": length,
                "completed": len(records),
                "elapsed_seconds": time.monotonic() - started,
                "observed_max_decode_residency": max(
                    (sample["resident_batch"] for sample in samples), default=0
                ),
                "exact_resident_samples": sum(
                    sample["resident_batch"] == concurrency for sample in samples
                ),
                "requests": records,
                "graph_samples": samples,
            }
            case["residency_qualified"] = (
                case["observed_max_decode_residency"] >= concurrency
                and case["exact_resident_samples"] >= 3
            )
            report["cases"].append(case)
            report["completed_requests"] += len(records)
            report["scope"] = (
                "Real input tokens, chunked prefill and greedy acceptance; sampled memory includes actual prefill. "
                "Saturated near-1M concurrency requires its own workload; no fake-service latency inference."
            )
            (output / "real-capacity-summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                f"Real prefill C{concurrency}, {length} input: {len(records)} complete; "
                f"decode residency {case['observed_max_decode_residency']}",
                flush=True,
            )
            assert case[
                "residency_qualified"
            ], "real decode residency was not qualified"
    return report
