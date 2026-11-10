#!/usr/bin/env python3
"""Exercise representative prefill shapes before benchmark timing."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# Paired requests exercise both workers in the usual DP2 deployment. Varied
# lengths cover the FP8 GEMM tile-order and split-reduction variants missed by
# a uniform warmup. This is coverage for the measured MiMo workload, not a
# guarantee against compilation for every future serving configuration.
INPUT_LENGTHS = (
    128,
    257,
    513,
    6656,
    2049,
    3073,
    4095,
    10240,
    6145,
    8191,
    12288,
    14784,
    15744,
    15873,
    16383,
    16384,
)


def request(base_url, route, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(base_url + route, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run(args):
    if args.requests < 0 or args.timeout <= 0:
        raise ValueError("requests must be non-negative and timeout must be positive")
    report = {
        "status": "running",
        "requested": args.requests,
        "completed": 0,
        "mode": "representative",
        "started": time.time(),
        "batches": [],
        "measured": False,
    }
    try:
        if not args.requests:
            report["status"] = "skipped"
            return report
        host = (
            f"[{args.host}]"
            if ":" in args.host and not args.host.startswith("[")
            else args.host
        )
        base_url = f"http://{host}:{args.port}"
        info = request(base_url, "/server_info", args.timeout)
        context = info.get("context_length")
        if not isinstance(context, int) or context < 2:
            raise ValueError("server_info must report context_length >= 2")
        report["server"] = {
            key: info.get(key)
            for key in (
                "model_path",
                "context_length",
                "tp_size",
                "dp_size",
                "ep_size",
                "moe_dense_tp_size",
                "tokenizer_worker_num",
                "tokenizer_backend",
                "enable_prefill_delayer",
                "prefill_delayer_max_delay_passes",
            )
        }
        for index in range((args.requests + 1) // 2):
            count = min(2, args.requests - report["completed"])
            length = min(INPUT_LENGTHS[index % len(INPUT_LENGTHS)], context - 1)
            payload = {
                "input_ids": [[100] * length for _ in range(count)],
                "sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            }
            started = time.time()
            outputs = request(base_url, "/generate", args.timeout, payload)
            if not isinstance(outputs, list) or len(outputs) != count:
                raise RuntimeError("Warmup response does not match the requested batch")
            for output in outputs:
                meta = output.get("meta_info", {})
                if (
                    meta.get("prompt_tokens") != length
                    or meta.get("completion_tokens") != 1
                    or meta.get("finish_reason", {}).get("type") == "abort"
                ):
                    raise RuntimeError(
                        "Warmup request did not complete with the expected token counts"
                    )
            report["completed"] += count
            report["batches"].append(
                {
                    "input_tokens": length,
                    "requests": count,
                    "started": started,
                    "finished": time.time(),
                }
            )
        report["status"] = "complete"
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished"] = time.time()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        print(f"Prefill warmup failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Representative prefill warmup: {report['completed']}/{report['requested']} requests; {report['status']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
