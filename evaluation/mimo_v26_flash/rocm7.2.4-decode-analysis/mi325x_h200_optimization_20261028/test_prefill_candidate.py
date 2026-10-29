"""Exercise existing adversarial BF16 page-tail checks against a candidate."""

import argparse
import functools
import importlib.util
import json
from pathlib import Path
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--cooperative-v', type=int, default=1)
parser.add_argument('--manual-schedule', type=int, default=0)
parser.add_argument('--cta-multiplier', type=int, default=1)
parser.add_argument('--workgroup-waves', type=int, default=8)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert torch.cuda.get_device_properties().gcnArchName.startswith('gfx942')
from sglang.srt.layers.attention import flydsl_pa

flydsl_pa.PagedAttention = functools.partial(
    flydsl_pa.PagedAttention,
    cooperative_v=bool(args.cooperative_v),
    manual_schedule=bool(args.manual_schedule),
    cta_multiplier=args.cta_multiplier,
    workgroup_waves=args.workgroup_waves,
)
path = args.source_root / 'test/registered/attention/test_mimo_flypa_nan_tail.py'
spec = importlib.util.spec_from_file_location('mimo_tail_tests', path)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
passed = []
started = time.monotonic()
for case in tests.CASES:
    if case[-1] != 'bf16':
        continue
    print('Running', case, flush=True)
    tests.test_unused_nan_v_tail(*case)
    passed.append(str(case))
    print('Passed', flush=True)
tests.test_noncausal_nan_tail()
passed.append('noncausal_nan_tail')
tests.test_valid_nan_values_are_not_sanitized()
passed.append('valid_nan_values_are_not_sanitized')
args.output.write_text(json.dumps(dict(passed=passed, count=len(passed), elapsed_seconds=time.monotonic()-started, cooperative_v=args.cooperative_v, manual_schedule=args.manual_schedule, cta_multiplier=args.cta_multiplier, workgroup_waves=args.workgroup_waves), indent=2) + '\n')
print(len(passed), 'cases passed', flush=True)
