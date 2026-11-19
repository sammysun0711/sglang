"""Profile fake decode with complete request batches at a fixed concurrency.

Used by run_sglang_profile.sh with PROFILE_BATCH_API=1. Payload preparation and
warmup precede profiling; every measured wave waits for all of its responses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import orjson
import requests

from sglang.bench_serving import flush_server_cache, get_tokenizer
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("backend", "model", "host", "dataset-name", "dataset-path", "profile-prefix"):
        parser.add_argument("--" + name, required=True)
    for name in ("port", "random-input-len", "random-output-len", "seed", "num-prompts", "warmup-requests", "max-concurrency"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--random-range-ratio", type=float, required=True)
    parser.add_argument("--served-model-name")
    for name in ("flush-cache", "tokenize-prompt", "fake-prefill", "profile"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    assert args.backend == "sglang" and args.dataset_name == "random"
    assert args.fake_prefill and args.tokenize_prompt and args.profile and args.flush_cache
    assert args.random_range_ratio == 1.0
    assert args.num_prompts >= args.max_concurrency > 0
    assert args.num_prompts % args.max_concurrency == 0
    assert 0 <= args.warmup_requests <= args.num_prompts

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = sample_random_requests(
        args.random_input_len, args.random_output_len, args.num_prompts,
        args.random_range_ratio, get_tokenizer(args.model), args.dataset_path,
        random_sample=True, return_text=False,
    )
    assert len(rows) == args.num_prompts
    assert all(len(row.prompt) == args.random_input_len for row in rows)

    def payload(batch, room):
        return orjson.dumps(dict(
            input_ids=[row.prompt for row in batch],
            sampling_params=dict(temperature=0.0, top_p=1.0,
                                 max_new_tokens=args.random_output_len, ignore_eos=True),
            stream=False, bootstrap_host=FAKE_BOOTSTRAP_HOST, bootstrap_room=room,
        ))

    warms = [(payload(rows[i:min(i+args.max_concurrency, args.warmup_requests)], i),
              min(args.max_concurrency, args.warmup_requests-i))
             for i in range(0, args.warmup_requests, args.max_concurrency)]
    waves = [payload(rows[i:i+args.max_concurrency], i)
             for i in range(0, args.num_prompts, args.max_concurrency)]
    del rows
    session = requests.Session()
    base = f"http://{args.host}:{args.port}"

    def generate(data, expected):
        response = session.post(base+'/generate', data=data,
                                headers={'Content-Type': 'application/json'}, timeout=600)
        response.raise_for_status()
        outputs = response.json()
        assert isinstance(outputs, list) and len(outputs) == expected
        for output in outputs:
            meta = output['meta_info']
            assert meta['prompt_tokens'] == args.random_input_len
            assert meta['completion_tokens'] == args.random_output_len
            assert meta.get('finish_reason', {}).get('type') != 'abort', meta
        return len(outputs)

    for i, (data, count) in enumerate(warms):
        generate(data, count)
        print(f'Warmup batch {i+1}/{len(warms)}: {count} requests completed', flush=True)
    flush_server_cache(base, 'sglang', 60)
    time.sleep(1)
    trace_dir = Path(os.environ['SGLANG_TORCH_PROFILER_DIR']) / str(time.time())
    trace_dir.mkdir(parents=True, exist_ok=True)
    profile_body = dict(output_dir=str(trace_dir), profile_prefix=args.profile_prefix,
                        activities=['CPU', 'GPU'], with_stack=True, record_shapes=True)
    response = session.post(base+'/start_profile', json=profile_body, timeout=120)
    response.raise_for_status()
    print('Profiler started:', response.text, flush=True)
    completed = 0
    timings = []
    started = time.time()
    try:
        for i, data in enumerate(waves):
            begin = time.perf_counter()
            completed += generate(data, args.max_concurrency)
            seconds = time.perf_counter()-begin
            timings.append(seconds)
            print(f'Profiled wave {i+1}/{len(waves)}: {args.max_concurrency} requests in {seconds:.3f}s', flush=True)
    finally:
        finished = time.time()
        response = session.post(base+'/stop_profile', json={}, timeout=600)
        response.raise_for_status()
        print('Profiler stopped:', response.text, flush=True)
    record = dict(completed=completed, total_input_tokens=completed*args.random_input_len,
                  total_output_tokens=completed*args.random_output_len,
                  max_concurrency=args.max_concurrency, input_tokens=args.random_input_len,
                  output_tokens=args.random_output_len, warmup_requests=args.warmup_requests,
                  waves=len(waves), wave_seconds=timings, profile_start=started,
                  profile_requests_end=finished, trace_dir=str(trace_dir),
                  payload_sha256=hashlib.sha256(b''.join(waves)).hexdigest(),
                  scope='Diagnostic profile with batched API submission; profiler overhead applies.')
    Path('profile-batch-result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
