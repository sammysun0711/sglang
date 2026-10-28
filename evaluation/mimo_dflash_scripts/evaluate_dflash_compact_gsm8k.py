#!/usr/bin/env python3
"""Paired greedy GSM8K evidence with pinned prompts and every response retained.

Uses the repository's answer parser and five-shot protocol. The five prompt
examples are excluded from scoring. HTTP errors, truncated generations and
retractions are reported explicitly rather than converted to empty answers.
"""

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests
from sglang.test.simple_eval_gsm8k import (
    INVALID,
    get_answer_value,
    get_few_shot_examples,
)
from transformers import AutoTokenizer


def evaluate_gsm8k(
    url, output, dataset, model_path, *, count=None, workers=8, max_tokens=2048
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = Path(dataset).read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    shots = 5
    prefix = get_few_shot_examples(rows, shots)
    questions = rows[shots:]
    if count is not None:
        questions = questions[:count]
    if not questions:
        raise ValueError("no evaluation questions after excluding five prompt examples")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = []
    for i, row in enumerate(questions):
        messages = [
            {
                "role": "user",
                "content": prefix + f"Question: {row['question']}\nAnswer:",
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = list(ids)
        prompts.append(
            {
                "dataset_index": i + shots,
                "label": get_answer_value(row["answer"]),
                "input_ids": ids,
                "question": row["question"],
            }
        )
    if any(p["label"] == INVALID for p in prompts):
        raise ValueError("unparseable dataset answer")
    prompt_bytes = json.dumps(
        prompts, ensure_ascii=False, separators=(",", ":")
    ).encode()
    (output / "prompts.json").write_bytes(prompt_bytes)
    manifest = {
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "dataset_count": len(rows),
        "excluded_example_indices": list(range(shots)),
        "scored_questions": len(prompts),
        "workers": workers,
        "max_new_tokens": max_tokens,
        "temperature": 0,
        "thinking": False,
        "model_path": str(model_path),
        "scorer": "sglang.test.simple_eval_gsm8k.get_answer_value",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()

    def run(prompt):
        index = prompt["dataset_index"]
        record = {"dataset_index": index, "label": prompt["label"]}
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.post(
                url + "/generate",
                json={
                    "rid": f"gsm8k-{index}",
                    "input_ids": prompt["input_ids"],
                    "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
                },
                timeout=900,
            )
            response.raise_for_status()
            value = response.json()
            meta = value["meta_info"]
            answer = get_answer_value(value["text"])
            record.update(
                {
                    "response": value,
                    "answer": answer,
                    "correct": answer == prompt["label"],
                    "invalid": answer == INVALID,
                    "truncated": meta["finish_reason"]["type"] == "length",
                    "retractions": meta["num_retractions"],
                }
            )
        except Exception as error:  # noqa: BLE001
            # Preserve failed requests in the report and score the remaining ones.
            record["error"] = repr(error)
        finally:
            session.close()
        (output / f"question-{index:04d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record

    records = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        for record in executor.map(run, prompts):
            records.append(record)
            if len(records) % 32 == 0 or len(records) == len(prompts):
                print(
                    f"GSM8K {len(records)}/{len(prompts)}: {sum(r.get('correct', False) for r in records)} correct",
                    flush=True,
                )
    except BaseException:
        # Let the owned-server harness clean up immediately on cancellation;
        # do not drain thousands of queued requests against a failed model.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    errors = [r["dataset_index"] for r in records if "error" in r]
    retractions = sum(r.get("retractions", 0) for r in records)
    summary = {
        **manifest,
        "completed": len(records) - len(errors),
        "correct": sum(r.get("correct", False) for r in records),
        "score": sum(r.get("correct", False) for r in records) / len(records),
        "invalid": sum(r.get("invalid", False) for r in records),
        "truncated": sum(r.get("truncated", False) for r in records),
        "errors": errors,
        "retractions": retractions,
        "elapsed_seconds": time.monotonic() - started,
        "transport_passed": not errors and retractions == 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["transport_passed"]:
        raise RuntimeError(f"GSM8K requests failed: {summary}")
    return summary
