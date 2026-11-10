#!/usr/bin/env python3
"""Exercise representative prefill shapes before benchmark timing."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from sglang.srt.entrypoints.mimo_prefill_warmup import (
    representative_batches,
    validate_warmup_response,
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
        for payload in representative_batches(args.requests, context):
            count = len(payload["input_ids"])
            lengths = [len(ids) for ids in payload["input_ids"]]
            started = time.time()
            outputs = request(base_url, "/generate", args.timeout, payload)
            validate_warmup_response(outputs, payload)
            report["completed"] += count
            report["batches"].append(
                {
                    "input_tokens": lengths,
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
