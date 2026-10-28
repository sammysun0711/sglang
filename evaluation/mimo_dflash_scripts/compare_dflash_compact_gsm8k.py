#!/usr/bin/env python3
"""Compare completed paired score runs and retain every changed answer/output."""

import argparse
import json
from itertools import zip_longest
from pathlib import Path


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    summaries = [
        json.loads((p / "summary.json").read_text()) for p in (baseline, candidate)
    ]
    server_fields = (
        "context_length",
        "max_total_num_tokens",
        "max_running_requests",
        "swa_full_tokens_ratio",
        "cuda_graph_config",
        "kv_cache_dtype",
        "speculative_draft_kv_cache_dtype",
        "model_path",
        "disable_custom_all_reduce",
        "enable_fp32_lm_head",
        "aiter_mxfp4_stage2_output_dtype",
    )
    server_configs = []
    for directory in (baseline, candidate):
        path = directory.parent / "server-info.json"
        if not path.is_file():
            raise ValueError(f"paired score run requires server configuration: {path}")
        info = json.loads(path.read_text())
        server_configs.append({key: info.get(key) for key in server_fields})
    same_server_config = server_configs[0] == server_configs[1]
    completed_runs = all(
        json.loads((p.parent / "result.json").read_text())["status"] == "passed"
        for p in (baseline, candidate)
    )
    keys = (
        "dataset_sha256",
        "prompt_sha256",
        "excluded_example_indices",
        "scored_questions",
        "workers",
        "max_new_tokens",
        "temperature",
        "thinking",
        "scorer",
    )
    matching_protocol = all(summaries[0][key] == summaries[1][key] for key in keys)
    differences = []
    regressions = []
    improvements = []
    equal_outputs = 0
    answer_equal = 0
    base_files = {p.name: p for p in baseline.glob("question-*.json")}
    candidate_files = {p.name: p for p in candidate.glob("question-*.json")}
    paired_indices = sorted(set(base_files) & set(candidate_files))
    record_errors = []
    observed_correct = [0, 0]
    for name in paired_indices:
        a, b = [
            json.loads(files[name].read_text())
            for files in (base_files, candidate_files)
        ]
        if a["dataset_index"] != b["dataset_index"] or a["label"] != b["label"]:
            record_errors.append(
                {"file": name, "error": "question index or ground-truth label differs"}
            )
        if "error" in a or "error" in b:
            differences.append(
                {"dataset_index": a["dataset_index"], "request_error": True}
            )
            continue
        for side, record in enumerate((a, b)):
            actual_correct = record["answer"] == record["label"]
            observed_correct[side] += actual_correct
            if record["correct"] != actual_correct:
                record_errors.append(
                    {
                        "file": name,
                        "side": side,
                        "error": "record correctness disagrees with parsed answer",
                    }
                )
        output_equal = a["response"]["output_ids"] == b["response"]["output_ids"]
        answers_equal = a["answer"] == b["answer"]
        equal_outputs += output_equal
        answer_equal += answers_equal
        if not output_equal:
            a_ids, b_ids = a["response"]["output_ids"], b["response"]["output_ids"]
            first_difference = next(
                i
                for i, (a_id, b_id) in enumerate(zip_longest(a_ids, b_ids))
                if a_id != b_id
            )
            differences.append(
                {
                    "dataset_index": a["dataset_index"],
                    "baseline_answer": a["answer"],
                    "candidate_answer": b["answer"],
                    "baseline_correct": a["correct"],
                    "candidate_correct": b["correct"],
                    "answer_equal": answers_equal,
                    "baseline_truncated": a["truncated"],
                    "candidate_truncated": b["truncated"],
                    "first_differing_output_position": first_difference,
                    "baseline_token": (
                        a_ids[first_difference]
                        if first_difference < len(a_ids)
                        else None
                    ),
                    "candidate_token": (
                        b_ids[first_difference]
                        if first_difference < len(b_ids)
                        else None
                    ),
                    "baseline_completion_tokens": len(a_ids),
                    "candidate_completion_tokens": len(b_ids),
                    "baseline_accepted_drafts": a["response"]["meta_info"].get(
                        "spec_accepted_drafts"
                    ),
                    "candidate_accepted_drafts": b["response"]["meta_info"].get(
                        "spec_accepted_drafts"
                    ),
                }
            )
        if a["correct"] and not b["correct"]:
            regressions.append(a["dataset_index"])
        if b["correct"] and not a["correct"]:
            improvements.append(a["dataset_index"])
    complete = (
        len(base_files)
        == len(candidate_files)
        == len(paired_indices)
        == summaries[0]["scored_questions"]
    )
    transport_passed = all(
        s["transport_passed"] and s["completed"] == s["scored_questions"]
        for s in summaries
    )
    score_non_regression = summaries[1]["correct"] >= summaries[0]["correct"]
    verified_summary = (
        observed_correct == [s["correct"] for s in summaries] and not record_errors
    )
    return {
        "baseline": summaries[0],
        "candidate": summaries[1],
        "matching_protocol": matching_protocol,
        "matching_server_config": same_server_config,
        "completed_serving_runs": completed_runs,
        "complete_pairs": complete,
        "transport_passed": transport_passed,
        "record_errors": record_errors,
        "verified_summary": verified_summary,
        "equal_outputs": equal_outputs,
        "equal_answers": answer_equal,
        "regressions": regressions,
        "improvements": improvements,
        "output_differences": differences,
        "score_delta": summaries[1]["score"] - summaries[0]["score"],
        "score_non_regression": score_non_regression,
        "score_gate_passed": complete
        and matching_protocol
        and same_server_config
        and completed_runs
        and transport_passed
        and verified_summary
        and score_non_regression,
        "exact_output_equivalence": complete and equal_outputs == len(paired_indices),
        "cache_gate": "score comparison supplements same-input KV/proposal gates; differing outputs require investigation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("output_differences", "baseline", "candidate")
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["score_gate_passed"] else 1)
