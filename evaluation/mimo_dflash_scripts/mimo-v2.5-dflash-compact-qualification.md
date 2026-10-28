# MiMo-V2.5 compact DFlash qualification — 2026-10-06

The opt-in `--speculative-dflash-compact-cache` replaces the context-sized
five-layer draft KV pool with two contiguous page-64 banks per possible
request owner. Each bank has 1152 rows. Target verification keeps target
addresses; draft forward uses private addresses, then accepted target hidden
rows overwrite speculative KV. Prefill selects the visible suffix before
projection. Request identity/retraction generations and GPU events protect
bank reuse. Compact mode uses torch BF16 QKV projection and RCCL, as qualified
here. The default-off path retains its original allocation and tuned projection.

M0–M5 passed within the scope below. M6 copy profiling supports keeping the
existing direct multi-layer copy; optional ring/support expansion is deferred.
The full experiment history remains in the local
`mimo-v2.5-dflash-memory-ratio-opt.md` milestone tracker and the source-checkpoint
archive described below.

## Commit preparation checks — 2026-10-28

The seven-commit series preserves the qualified cache/QKV implementation and
adds CI registration, executable pytest entry points, and a model-aware guard
for the known FP8 target profile. MiMo's MXFP4 checkpoint also declares
`quant_method=fp8`; its `store_dtype=mxfp4` keeps it distinct from the FP8 target
that requires `--enable-fp32-lm-head`.

The completed runtime stack passed **1,470 tests**, including compact-cache
CPU/GPU tests and existing pool, speculation-registry, sampling, and
disaggregation tests. Both QKV branch checks, all five CI registrations,
seven evaluation CLI help checks, and four launcher syntax checks passed.
The two pure geometry/configuration files register for CPU CI; integrations
that import the ROCm stack and the GPU suite register for AMD MI35x CI.
The three checkpoint-dependent GPU cases still need the declared checkpoint
or skip when it is unavailable.

These are fresh functional checks. The TP8 GSM8K, memory, and service results
below remain the historical October 6 qualification; no new model-server or
performance campaign was run while preparing the commits.

## Qualified deployment scope

- Eight MI350X/gfx950 GPUs, ROCm 7.2, TP8/PP1/DP1.
- MiMo-V2.5 MXFP4 target, or FP8 target with `--enable-fp32-lm-head`.
- Five uniform SWA draft layers, window 1024, block 8, BF16 target/draft KV,
  page 64, AITER/FlyDSL, RCCL (`--disable-custom-all-reduce`).
- Radix/HiCache, TBO, plan-stream overlap and context parallelism disabled.
- Greedy ordinary serving and separately qualified fake decode. Real
  disaggregation transfer remains rejected.

The archive includes required common AITER repairs, including FP8 MoE
activation-scale orientation, page-64 target D192/V128 CK specialization,
MTP D192/V192 flat/page-64 families and Opus duplicate dispatch repair. Serving
this source against an unpatched AITER checkout is outside the qualified setup.

## Accuracy and correctness

Per the user's clarification, GSM8K is sufficient for M3 numerical qualification.
Independent deterministic output identity is not an exit requirement.

| Profile | Legacy score | Compact score | Qualification |
| --- | ---: | ---: | --- |
| MXFP4 | 1272/1314 (96.8037%) | 1272/1314 (96.8037%) | Passed |
| FP8, FP32 target head | 1271/1314 (96.7275%) | 1275/1314 (97.0320%) | Passed |
| FP8, BF16 target head | 1273/1314 (96.8798%) | 1269/1314 (96.5753%) | Failed profile retained |

The representative FP8 MTP FP32-head baseline scored 1268/1314 (96.4992%),
with no invalid answers, truncations, transport errors or retractions. This
baseline uses the same prompt/scoring protocol but is not a paired storage gate.

Protocol: five-shot, thinking disabled, greedy, 8 workers, maximum 2048 output
tokens; all 1319 dataset rows loaded, first 5 demonstrations excluded.
Dataset SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`;
prompt SHA256 `8d919d1ddbbf6cfea3d6dad3f326211bad9003227fac39088f0cfa8397a8d6c2`.

Same-input TP8 storage/proposal checks include 675 MXFP4 production shadow
steps, 2599 FP8 shadow steps and 1223 forced sequential/eager steps: zero hidden
delta, bitwise committed KV and exact proposals. Graph/lifecycle workloads
cover long prefill, repeated window turnovers, mixed batches, all owner slots,
flush/reuse, early stop, abort and queued-client timeout cleanup.

Both qualified formats pass C18 fake integration in eager and graph modes:
112 requests per workload, all 54 non-dummy owners, zero visible suffixes on TP8,
absolute prefixes through 1,047,000. Each graph workload records 16,087 request
steps and 1144 graph forwards. Final verification: 1215 CPU cases and 196 GPU
cases pass, in `final-cpu-qualification.log` and `final-gpu-qualification.log`.

## Memory and service results

At equal target capacity F=20,080,576/S=80,320, admission C18 and identical
graphs 1/2/4/8/16/18, dtype, target format, Stage-2 dtype and requested fraction:

| Per GPU | Legacy | Compact |
| --- | ---: | ---: |
| Draft KV bytes | 51,406,438,400 | 324,567,040 |
| Draft request table bytes | 230,689,360 | 253,440 |
| Minimum sampled free VRAM | 0.907 GiB | 47.453 GiB |

Draft KV savings are **47.57370 GiB/GPU (99.3686%)**. Compact persistent
metadata totals 255,064 bytes, including its request table. The fresh service
pair completes 400 requests with no errors/retractions. Throughput changes
range −1.52% to +1.21% across C8/C9/C16/C17: similar throughput, with substantial
memory savings. This single pair does not establish a repeatable speedup.

Selected service settings use three fresh servers each, simulated acceptance 4,
1,047,488 input tokens, 1024 output tokens, four waves, and median TPOT ≤25 ms.
Actual request residency and graph buckets are sampled separately from client
concurrency; every selected case passes residency.

| Selected profile | Requested fraction / SWA ratio | Resident C | Mean output TPS | Mean median TPOT | TPS range | TPS CV |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| MXFP4 compact | .90 / .004 | 16 | 1466.29 | 10.197 ms | 1441.76–1483.13 | 1.48% |
| FP8 compact, FP32 head | 1.0 / .004 | 13 | 1194.50 | 10.204 ms | 1178.61–1212.13 | 1.41% |
| FP8 MTP, FP32 head | 1.0 / .004 | 14 | 1206.82 | 10.899 ms | 1200.98–1213.28 | 0.51% |

All use graphs 1/2/4/8/16/18. Requested fractions resolve through the existing
AITER adjustment: .90→.765 and 1.0→.85. MXFP4 true resident C17 fits at .90–1.0,
but C16 is faster. FP8 DFlash C16/C17 submissions reach at most 13 resident;
MTP reaches 14. Those higher client counts are queued service, not qualified
resident C16/C17 capacity.

FP8 compact is −1.02% versus tuned MTP at each method's selected concurrency.
MXFP4 compact is +21.50% versus FP8 MTP, with target weight format explicitly
changed. At equal resident C13, the single FP8 DFlash/MTP sweep showed +6.17%
DFlash throughput; selecting MTP C14 removes that best-concurrency advantage.

Padding materially helps some shapes: MXFP4 C9 graph 16 gives +20.11% versus
exact graph 9 in the controlled sweep. FP8 exact C13 measured 1033.65 TPS versus
padded C13's first 1212.13 TPS. AITER's next-power-of-two MoE row dispatch also
changes at these boundaries; these observations do not isolate every source
of iteration latency.

## Separate real-prefill settings

Actual 64K, 256K and 1,047,488-token prompts at C1, plus 16×64K prompts, each
generate 1024 tokens with real acceptance. Both formats pass completion,
no-retraction and sampled residency gates with target SWA ratio .016:

| Profile | Requested fraction / SWA ratio | Full / SWA rows | Minimum sampled free VRAM |
| --- | --- | --- | ---: |
| MXFP4 compact | .90 / .016 | 18,814,976 / 300,992 | 25.558 GiB |
| FP8 compact, FP32 head | 1.0 / .016 | 13,694,144 / 219,072 | 25.273 GiB |

Ordinary C18 serving reserves 18 draft owners: 112,230,400 KV bytes plus 88,312
metadata bytes. Fake C18 reserves 54 owners because decode preallocation adds
36 indices; budgets include dummy banks and an extra sentinel page.

MXFP4 real `.90/.004` returned all 19 responses but retracted one C16×64K
request when target SWA reached 99%. It fails the no-retraction gate and is
retained. Fake `.004` capacity must not be used as a general real-prefill
recommendation. Saturated real C17×near-1M prefill was not exercised.

Device VRAM is sampled every second through loading, graph capture and the
workload; subsecond peaks may be lower than the recorded free-memory minima.
The fake service numbers do not predict real-prefill latency.

## Reproduce the selected compact service workload

From `/root/workspace/mimo-opt/mimo-dflash-env-20261005`:

```bash
source env.sh
export GPU_ARCHS=gfx950 DISABLE_CUSTOM_ALL_REDUCE=1
export AITER_MXFP4_STAGE2_OUTPUT_DTYPE=fp8
python -u sglang/evaluation/mimo_dflash_scripts/validate_dflash_compact_server.py \
  --source-root sglang --output-dir logs/dflash-compact/new-mxfp4-service \
  --port 30126 --compact --target-variant mxfp4 --fake-prefill \
  --memory-fraction .90 --swa-ratio .004 --max-running 18 \
  --graph-sizes 1 2 4 8 16 18 --service-concurrencies 16
```

For selected FP8 fake service use `--target-variant fp8 --fp32-lm-head`,
`--memory-fraction 1.0` and `--service-concurrencies 13`. For the declared real
prefill workload remove `--fake-prefill --service-concurrencies ...`, add
`--real-capacity`, and set `--swa-ratio .016`. Use a fresh output directory and
available port. The harness stops only its owned server and saves configs,
source patches, ledger, VRAM samples, responses and graph/residency evidence.

## Evidence archive

All artifact names below are under `../../../logs/dflash-compact/`:

- `mxfp4-gsm8k-comparison.json`, `fp8-fp32-head-gsm8k-comparison.json`,
  `fp8-fixed-gsm8k-comparison.json`, `m0-fp8-mtp-fp32-head-gsm8k/`.
- `m5-mxfp4-fixed-capacity-service-comparison.json`,
  `m5-mxfp4-fraction-sweep.json`, `m5-mxfp4-padded-three-run-summary.json`.
- `m5-fp8-padded-c13-three-run-summary.json`,
  `m5-fp8-mtp-c14-three-run-summary.json`,
  `m5-best-measured-service-comparison.json`.
- `m5-mxfp4-real-prefill-capacity-fraction090/` (failed .004),
  `m5-mxfp4-real-prefill-capacity-fraction090-swa016/`,
  `m5-fp8-real-prefill-capacity-fraction100-swa016/`.
- `m6-bounded-copy-profile.json`: isolated warmed graph-copy median 34.94 µs
  at C16, excluding CPU planning/projection/attention/communication. Retain
  the qualified two-bank copy; optional page ring/TBO/real-transfer/sampling
  extensions require separate qualification.
- `source-checkpoint-final/`: pinned commits, complete tracked source patches,
  new compact/harness/test files, required AITER/CK patches and SHA256 manifest.
