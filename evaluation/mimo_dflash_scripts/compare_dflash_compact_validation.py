"""Compare generated tokens and per-request draft observations, never just scores."""

import argparse
import json
from pathlib import Path

import torch


def traces(directory):
    result = {}
    # Startup warmup requests have generated IDs and are not matched workloads.
    case_ids = set()
    for case in directory.glob("*.json"):
        value = json.loads(case.read_text())
        if isinstance(value, dict) and "output_ids" in value:
            case_ids.add(case.stem)
    for path in sorted((directory / "draft-traces").glob("*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=True)
        n = len(batch["rids"])
        hidden = batch["draft_hidden"].reshape(n, 7, -1)
        draft = batch["draft_ids"].reshape(n, 7)
        output = batch["output_ids"].reshape(n, 8)
        lengths = batch.get("committed_lengths")
        offset = 0
        for i, rid in enumerate(batch["rids"]):
            length = int(lengths[i]) if lengths is not None else 0
            if rid not in case_ids:
                offset += length
                continue
            key = (rid, int(batch["prefix_lens"][i]))
            if key in result:
                raise ValueError(f"duplicate trace key: {key}")
            observation = {
                "hidden": hidden[i],
                "draft": draft[i],
                "commit": int(batch["commit_lens"][i]),
                "output": output[i],
            }
            if "draft_logits_rank0" in batch:
                observation["logits"] = batch["draft_logits_rank0"].reshape(n, 7, -1)[i]
            if lengths is not None:
                observation["k"] = batch["committed_k"][:, offset : offset + length]
                observation["v"] = batch["committed_v"][:, offset : offset + length]
                offset += length
            result[key] = observation
    return result


def prefill_probes(directory):
    probes = {}
    for path in sorted((directory / "prefill-traces").glob("*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=True)
        if len(batch["rids"]) != 1:
            continue
        rid = batch["rids"][0]
        for materialization in batch["materialization_inputs"]:
            for pos, row in zip(
                materialization["positions"].tolist(), materialization["target_hidden"]
            ):
                probes[(rid, pos)] = row
    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-traces", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Diagnostic comparison only; does not establish a completed serving gate.",
    )
    args = parser.parse_args()
    report = {"cases": {}, "trace": {}}
    failed = False
    run_status = {}
    for name, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
        result_path = directory / "result.json"
        status = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {"status": "missing"}
        )
        run_status[name] = status
        if not args.allow_incomplete and status.get("status") != "passed":
            failed = True
    report["run_status"] = run_status
    report["diagnostic_only"] = args.allow_incomplete
    pool_configs = {}
    for name, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
        info_path = directory / "server-info.json"
        if not info_path.exists():
            failed = True
            continue
        info = json.loads(info_path.read_text())
        pool_configs[name] = {
            key: info.get(key)
            for key in (
                "context_length",
                "max_total_num_tokens",
                "max_running_requests",
                "swa_full_tokens_ratio",
                "cuda_graph_config",
                "kv_cache_dtype",
                "speculative_draft_kv_cache_dtype",
                "aiter_mxfp4_stage2_output_dtype",
            )
        }
    report["storage_ab_config_equal"] = (
        len(pool_configs) == 2 and pool_configs["baseline"] == pool_configs["candidate"]
    )
    if not report["storage_ab_config_equal"] and not args.allow_incomplete:
        failed = True
    for path in sorted(args.baseline.glob("*.json")):
        base = json.loads(path.read_text())
        if not isinstance(base, dict) or "output_ids" not in base:
            continue
        candidate_path = args.candidate / path.name
        if not candidate_path.exists():
            report["cases"][path.stem] = {"missing": True}
            failed = True
            continue
        candidate = json.loads(candidate_path.read_text())
        match = base["output_ids"] == candidate["output_ids"]
        report["cases"][path.stem] = {
            "output_ids_equal": match,
            "baseline_tokens": len(base["output_ids"]),
            "candidate_tokens": len(candidate["output_ids"]),
            "baseline_accepted_drafts": base["meta_info"].get("spec_accepted_drafts"),
            "candidate_accepted_drafts": candidate["meta_info"].get(
                "spec_accepted_drafts"
            ),
        }
        failed |= not match
    if not report["cases"]:
        raise ValueError("no generation cases to compare")
    candidate_cases = set()
    for path in args.candidate.glob("*.json"):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and "output_ids" in value:
            candidate_cases.add(path.stem)
    extra_cases = sorted(candidate_cases - set(report["cases"]))
    report["extra_candidate_cases"] = extra_cases
    failed |= bool(extra_cases)
    base_probes = prefill_probes(args.baseline)
    candidate_probes = prefill_probes(args.candidate)
    shared_probes = sorted(set(base_probes) & set(candidate_probes))
    input_differences = []
    for key in shared_probes:
        a, b = base_probes[key], candidate_probes[key]
        if not torch.equal(a, b):
            input_differences.append(
                {
                    "request_position": key,
                    "max_abs": (a.float() - b.float()).abs().max().item(),
                    "close": torch.allclose(a.float(), b.float(), atol=0.02, rtol=0.02),
                }
            )
    report["prefill_inputs"] = {
        "compared_probes": len(shared_probes),
        "differences": input_differences,
    }
    base_traces = traces(args.baseline)
    candidate_traces = traces(args.candidate)
    if args.require_traces and (not base_traces or not candidate_traces):
        raise ValueError("both runs must provide nonempty draft traces")
    if base_traces or candidate_traces:
        missing = sorted(set(base_traces) - set(candidate_traces))
        extra = sorted(set(candidate_traces) - set(base_traces))
        mismatches = []
        max_abs = 0.0
        for key in sorted(set(base_traces) & set(candidate_traces)):
            b, c = base_traces[key], candidate_traces[key]
            bh, bd, bc, bo = b["hidden"], b["draft"], b["commit"], b["output"]
            ch, cd, cc, co = c["hidden"], c["draft"], c["commit"], c["output"]
            diff = (bh.float() - ch.float()).abs()
            max_abs = max(max_abs, diff.max().item())
            close = torch.allclose(bh.float(), ch.float(), atol=0.02, rtol=0.02)
            tensor_checks = {}
            for name in ("logits", "k", "v"):
                if name not in b or name not in c:
                    if args.require_traces:
                        tensor_checks[name] = {"missing": True, "close": False}
                    continue
                shape_equal = b[name].shape == c[name].shape
                delta = (
                    (b[name].float() - c[name].float()).abs() if shape_equal else None
                )
                tensor_checks[name] = {
                    "shape_equal": shape_equal,
                    "bitwise_equal": shape_equal and torch.equal(b[name], c[name]),
                    "close": shape_equal
                    and torch.allclose(
                        b[name].float(), c[name].float(), atol=0.02, rtol=0.02
                    ),
                    "max_abs": delta.max().item() if delta is not None else None,
                }
            if (
                not close
                or not torch.equal(bd, cd)
                or bc != cc
                or not torch.equal(bo[:bc], co[:cc])
                or any(not check["close"] for check in tensor_checks.values())
            ):
                mismatches.append(
                    {
                        "request_prefix": key,
                        "hidden_close": close,
                        "hidden_max_abs": diff.max().item(),
                        "draft_ids_equal": torch.equal(bd, cd),
                        "baseline_commit": bc,
                        "candidate_commit": cc,
                        "tensor_checks": tensor_checks,
                    }
                )
        report["trace"] = {
            "baseline_steps": len(base_traces),
            "candidate_steps": len(candidate_traces),
            "missing": missing,
            "extra": extra,
            "mismatches": mismatches,
            "hidden_max_abs": max_abs,
        }
        failed |= bool(missing or extra or mismatches)
    report["passed"] = not failed
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
