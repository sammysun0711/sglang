"""Synthetic fake-transfer lifecycle qualification, separate from accuracy."""

import concurrent.futures
import json
import time
from pathlib import Path

import requests


def summarize_fake_integration(output, *, compact, eager, owner_count):
    output = Path(output)
    report = json.loads((output / "fake-workload-summary.json").read_text())
    directory = output / "fake-admission"
    forwards = [
        json.loads(line) for line in (directory / "forwards.jsonl").read_text().splitlines()
    ]
    assert forwards, "missing fake-forward observations"
    assert all(
        all(length == 4 for length in forward["accept_lens"])
        for forward in forwards
    ), "unexpected simulated per-step acceptance"
    graph_forwards = sum(forward["graph_replay"] for forward in forwards)
    assert bool(graph_forwards) != eager, "resolved graph mode was not exercised"
    summary = {
        "passed": True,
        "completed_requests": report["completed_requests"],
        "owner_count": owner_count,
        "eager": eager,
        "observed_batches": sorted({forward["batch_size"] for forward in forwards}),
        "graph_forwards": graph_forwards,
        "per_request_steps": sum(forward["batch_size"] for forward in forwards),
        "simulated_acceptance": 4,
    }
    if compact:
        admissions = [
            json.loads(line)
            for line in (directory / "admissions.jsonl").read_text().splitlines()
        ]
        owners = sorted({admission["owner"] for admission in admissions})
        assert owners == list(range(1, owner_count + 1)), "incomplete owner coverage"
        assert all(admission["zero_suffix_verified"] for admission in admissions)
        max_prefix = max(admission["prefix_len"] for admission in admissions)
        assert max_prefix >= 1047000, "missing full-context-boundary fake prefix"
        summary.update(
            observed_owners=owners,
            admissions=len(admissions),
            max_prefix=max_prefix,
            zero_suffix="verified on all TP8 ranks before rank-0 admission recording",
        )
    (output / "fake-integration-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def evaluate_fake_prefill(url, output, max_running):
    output = Path(output)
    responses = output / "fake-responses"
    responses.mkdir()
    records = []

    def generate(rid, length, count):
        ids = ([791, 6864, 315, 9822, 374] * ((length + 4) // 5))[:length]
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(
                url + "/generate",
                json={
                    "rid": rid,
                    "input_ids": ids,
                    "bootstrap_host": "2.2.2.2",
                    "bootstrap_room": 0,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": count,
                        "ignore_eos": True,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            value = response.json()
        (responses / f"{rid}.json").write_text(json.dumps(value, indent=2) + "\n")
        meta = value["meta_info"]
        assert meta["completion_tokens"] == count, value
        assert meta["num_retractions"] == 0, value
        # Completion count includes the fake prefill token and a final partial
        # block; its ratio to verify calls need not be exactly four.
        assert meta["spec_num_correct_drafts"] == 3 * meta["spec_verify_ct"], value
        histogram = meta["spec_correct_drafts_histogram"]
        assert histogram[3] == meta["spec_verify_ct"], value
        assert sum(histogram) == histogram[3], value
        assert value["output_ids"][1:] == [100] * (count - 1), value
        return {"rid": rid, "input_length": length, "output_length": count}

    with requests.Session() as client:
        client.trust_env = False
        started = time.monotonic()
        for wave in range(4):
            lengths = [1023, 1024, 1025, 1087, 8193, 65536, 262145, 1047000]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        generate, f"fake-wave-{wave}-{i}", length, 96 + i * 8
                    )
                    for i, length in enumerate(lengths)
                ]
                records.extend(f.result() for f in futures)
            print(
                f"Fake wave {wave + 1}: {len(lengths)} requests completed", flush=True
            )
        for size in (1, 2, 3, 4, 7, 8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=size) as executor:
                records.extend(
                    executor.map(
                        lambda i, size=size: generate(
                            f"fake-batch-{size}-{i}", 2048 + i, 65
                        ),
                        range(size),
                    )
                )
        # More outstanding requests than running slots exercise the complete
        # preallocation domain and reuse banks after mixed completion lengths.
        count = max_running * 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            records.extend(
                executor.map(
                    lambda i: generate(
                        f"fake-prealloc-{i}", 1024 + i, 1024 + (i % 8) * 16
                    ),
                    range(count),
                )
            )
        flush = client.post(url + "/flush_cache", params={"timeout": 30}, timeout=35)
        flush.raise_for_status()
        records.append(generate("fake-after-flush", 1047000, 96))
        report = {
            "passed": True,
            "completed_requests": len(records),
            "simulated_acceptance": 4,
            "elapsed_seconds": time.monotonic() - started,
            "requests": records,
            "scope": "fake-transfer synthetic lifecycle/graph/preallocation; accuracy is established by real GSM8K separately",
        }
        (output / "fake-workload-summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report
