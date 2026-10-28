# DFlash compact-cache review and commit plan — 2026-10-28

Created local branch **`feat/dflash-compact-cache`** in
`/root/workspace/mimo-opt/mimo-dflash-env-20261005/sglang`, based on
`228cdf7dc6311059862b424dbe2c971a85258fed`. The selected changes are organized
in seven local commits with Xiake Sun's sign-off only. No push was made.
Excluded setup, diagnostic, and historical files remain outside the series.

## Assessment

The core implementation remains supported by the declared October 6 evidence:
request-owned page-64 banks, target/draft address separation, accepted-prefix
commit, lifecycle protection, and allocation accounting. Fresh local tests pass.
This is the scoped MiMo implementation; it does not provide the broader
HiCache/real-PD integration of SGLang PR #40665.

At review start there were **17 modified tracked files and 38 untracked files**.
The **seven-commit** feature series covers **32 existing candidate files**,
plus this review. The other **23 files** remain local setup, optional
diagnostics, or historical archives.

## Review follow-ups applied in this series

1. **QKV computation remains an explicit, separate change.** Commit 4 uses
   `F.linear` in compact mode and retains the tuned `QKVParallelLinear` path
   when the flag is off. The qualification report now states this directly.
   A recorded service manifest has `repeatable_qkv=false` while server info
   reports `qkv_projection=torch_bf16_linear`: the production flag itself
   selects it. Removing this compute choice requires new model qualification.

2. **All five new tests have CI registrations and executable entry points.**
   The two pure layout/configuration modules register for CPU CI; budget,
   transfer, and GPU integrations use the AMD MI35x suite so their ROCm
   dependencies are available. The actual CI collector accepts all five.
   Model-weight-dependent GPU cases skip when the checkpoint is unavailable.

3. **Known FP8 target weights require the qualified FP32 LM head.** The
   model-aware check recognizes resolved/checkpoint `quant_method=fp8`, while
   exempting the qualified MXFP4 checkpoint's `store_dtype=mxfp4`. Eight new
   tests cover both metadata forms and both LM-head settings. This guard does
   not claim qualification for arbitrary other target formats.

4. **Custom speculative argument objects retain default-off behavior.** An
   existing registry test exposed direct access to a missing compact flag.
   The hook now uses `getattr(..., False)` so unrelated custom algorithms are
   unaffected.

5. **Setup and diagnostics remain separate.** The `round_normalized_k`
   diagnostic option, ROCm pyproject replacement, generated HIP files, and
   historical/diagnostic extras are excluded. Evaluation preparation only
   adds import ordering, an executable bit, explicit closure binding, and an
   explained broad catch for retaining per-request errors.

## Validation and report audit

At review start, the tracked SGLang patch and all 38 new files matched
`source-checkpoint-final/`, and every archive hash verified. The series adds
the follow-ups above; no cache algorithm or arithmetic was changed while
preparing the commits. The paired AITER and CK heads and patches match:

| Component | Base revision | Review baseline matches archive |
| --- | --- | --- |
| SGLang | `228cdf7dc6311059862b424dbe2c971a85258fed` | Yes |
| AITER | `90bb12270b5df9f207a0ef14d258baeb22168cca` | Yes |
| CK | `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` | Yes |

Fresh checks use the isolated `.venv` and `env.sh`: Torch 2.9.1/ROCm 7.2,
FlyDSL 0.3.2, and the paired AITER source. No model servers were launched.

| Fresh test scope | Result |
| --- | ---: |
| Compact layout/configuration/fake-transfer/budget unit tests | 1,215 passed |
| Compact cache GPU tests, including checkpoint materialization | 196 passed |
| Existing DFlash sampling/disaggregation tests | 4 passed |
| Final combined runtime regression suite | 1,470 passed |
| Final CI registration audit | All five accepted |
| QKV branch checks / evaluation CLI help checks | 2 / 7 passed |

The TP8 model and service figures below are historical results, verified against
stored artifacts; they were not rerun during this review:

- Fixed-capacity draft KV reduction: **47.57370 GiB/GPU (99.3686%)**.
- Matched service pair: throughput change **-1.52% to +1.21%**, with no claimed
  repeatable speedup. These are synthetic-acceptance service results.
- GSM8K: MXFP4 **1272/1314 -> 1272/1314**; FP8 with FP32 LM head
  **1271/1314 -> 1275/1314**.
- Preserve failed profiles: FP8/BF16 LM head **1273 -> 1269**, and real
  MXFP4 `.90/.004` C16 prefill with a retraction. The `.004` fake-workload
  setting should not become a general real-prefill recommendation.
- The qualified setup is TP8/gfx950, five SWA draft layers, W1024/block8,
  BF16 KV/page64, AITER/FlyDSL, RCCL, greedy requests, and disabled
  radix/HiCache/TBO/plan-stream overlap. Real PD transfer remains rejected.

The required AITER/CK fixes are separate dependencies. The SGLang branch is
based on this integration snapshot; it has not been rebased onto current
upstream SGLang.

## Commit sequence

| # | Suggested title | Scope |
| ---: | --- | --- |
| 1 | `feat(dflash): define compact-cache geometry and capability checks` | Pure bank geometry, owner counts, supported configuration checks, and unit tests. |
| 2 | `feat(dflash): add request-owned double-bank KV storage` | Copy/table kernels, allocator ownership, bank reuse/events, and GPU storage tests. |
| 3 | `feat(dflash): integrate bounded KV allocation and draft execution` | Budget solver, separate pools, worker prefill/decode/commit paths, public flag, and greedy request validation. |
| 4 | `fix(dflash): use the qualified BF16 QKV path in compact mode` | Isolate the existing torch-linear projection choice and its reporting; this is a compute change, not the bank allocator. |
| 5 | `feat(dflash): support fake admission and compact-cache telemetry` | Fake-transfer guard and initialization, owner admission, memory ledger, server info, and benchmark retraction/token accounting. |
| 6 | `test(dflash): add compact-cache qualification tooling` | Opt-in observers, evaluators/comparators, and matched DFlash/MTP launch controls. |
| 7 | `docs(dflash): record compact-cache qualification and limits` | Concise qualified profiles, exact evidence, failed profiles, dependency versions, and this review. |

The series splits feature hunks across shared files. The public flag appears
only after storage, budget, and worker integration are connected. Fake
disaggregation stays rejected until commit 5 adds transfer exclusion and prefix
initialization. CI registration accompanies each introduced test file.

Commit 4 is intentionally separate so the QKV execution change is visible to
reviewers. A later allocation-only upstream proposal could omit it only after
requalifying that resulting code and updating the report.

## Complete file allocation

Paths below are relative to the SGLang checkout. `M` means modified tracked
file; `A` means an existing untracked file proposed for addition.

| Status | File | Commit(s) |
| --- | --- | --- |
| M | `evaluation/launch_tp8_noep_aiter_mtp_accuracy_baseline.sh` | 6 |
| M | `evaluation/launch_tp8_noep_aiter_mtp_decode_fake_prefill_baseline.sh` | 6 |
| A | `evaluation/mimo_dflash_scripts/compact_trace/sitecustomize.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/compare_dflash_compact_gsm8k.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/compare_dflash_compact_validation.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/evaluate_dflash_compact_fake.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/evaluate_dflash_compact_gsm8k.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/evaluate_dflash_compact_real_capacity.py` | 6 |
| A | `evaluation/mimo_dflash_scripts/evaluate_dflash_compact_service.py` | 6 |
| M | `evaluation/mimo_dflash_scripts/launch_tp8_noep_aiter_dflash_accuracy_baseline.sh` | 6 |
| M | `evaluation/mimo_dflash_scripts/launch_tp8_noep_aiter_dflash_decode_fake_prefill_baseline.sh` | 6 |
| A | `evaluation/mimo_dflash_scripts/mimo-v2.5-dflash-compact-qualification.md` | 7 |
| A | `evaluation/mimo_dflash_scripts/validate_dflash_compact_server.py` | 6 |
| M | `python/sglang/bench_serving.py` | 5 |
| M | `python/sglang/srt/arg_groups/speculative_hook.py` | 3 |
| M | `python/sglang/srt/disaggregation/decode.py` | 5 |
| M | `python/sglang/srt/managers/scheduler.py` | 5 |
| M | `python/sglang/srt/managers/tokenizer_manager.py` | 3 |
| M | `python/sglang/srt/model_executor/model_runner.py` | 3 |
| M | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 3, 5 |
| M | `python/sglang/srt/model_executor/pool_configurator.py` | 3 |
| M | `python/sglang/srt/models/dflash.py` | 4 |
| M | `python/sglang/srt/server_args.py` | 3, 4, 5 |
| A | `python/sglang/srt/speculative/dflash_compact_cache.py` | 2, 4, 5 |
| A | `python/sglang/srt/speculative/dflash_compact_config.py` | 1, 5 |
| A | `python/sglang/srt/speculative/dflash_compact_layout.py` | 1 |
| M | `python/sglang/srt/speculative/dflash_worker_v2.py` | 3, 4 |
| A | `test/registered/attention/test_dflash_compact_cache.py` | 2, 5 |
| A | `test/registered/unit/model_executor/test_dflash_compact_budget.py` | 3 |
| A | `test/registered/unit/spec/test_dflash_compact_config.py` | 1, 5 |
| A | `test/registered/unit/spec/test_dflash_compact_fake_transfer.py` | 5 |
| A | `test/registered/unit/spec/test_dflash_compact_layout.py` | 1 |
| A | `evaluation/mimo_dflash_scripts/dflash-compact-cache-review-20261028.md` | 7 (new review) |

## Files to retain outside this feature series

These files have not been deleted or restored. They remain available locally.

| File | Reason |
| --- | --- |
| `evaluation/mimo_dflash_scripts/compare_dflash_numerical_replay.py` | Optional numerical diagnosis/replay tool; retain locally outside core feature series |
| `evaluation/mimo_dflash_scripts/diagnose_dflash_materialization.py` | Optional numerical diagnosis/replay tool; retain locally outside core feature series |
| `evaluation/mimo_dflash_scripts/diagnose_fp8_mimo_moe.py` | Optional numerical diagnosis/replay tool; retain locally outside core feature series |
| `evaluation/mimo_dflash_scripts/mimo-v2.5-dflash-memory-ratio-opt.md` | Historical experiment/ablation archive; link from concise qualification report |
| `evaluation/mimo_dflash_scripts/mimo-v2.5-fp4-dflash-ablation-test.md` | Historical experiment/ablation archive; link from concise qualification report |
| `python/pyproject.toml` | Local ROCm installation setup; byte-identical to pyproject_other.toml |
| `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py` | Diagnostic-only round_normalized_k option; production calls leave it false |
| `sgl-kernel/csrc/allreduce/deterministic_all_reduce_hip.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/allreduce/quick_all_reduce.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/allreduce/quick_all_reduce_hip.h` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/elementwise/activation.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/elementwise/deepseek_v4_topk.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/elementwise/dsv4_norm_rope.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/elementwise/pos_enc.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/elementwise/topk.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/grammar/apply_token_bitmask_inplace_hip.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/kvcacheio/transfer.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/moe/moe_align_kernel.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/csrc/speculative/eagle_utils.hip` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/include/hip/hip_act_and_mul_hip.cuh` | ROCm/HIP build artifact; unrelated to compact-cache implementation |
| `sgl-kernel/include/utils_hip.h` | ROCm/HIP build artifact; unrelated to compact-cache implementation |

The opt-in `compact_trace/sitecustomize.py` is included only with evaluation
tooling. Keep it out of normal production `PYTHONPATH`; its CPU observations
synchronize execution and its explicit flags can alter arithmetic for diagnosis.
The two long historical Markdown reports remain useful evidence archives and
can be linked rather than copied into a concise upstream description.

## Evidence location

Initial review backup, source comparisons, JUnit files, and file inventories:
`/root/workspace/mimo-opt/performance-results/dflash-compact-review-20261028-3b2zomt0`.

Historical complete source and qualification evidence:
`/root/workspace/mimo-opt/mimo-dflash-env-20261005/logs/dflash-compact/source-checkpoint-final`.

Commit preparation, intermediate test logs, final CI audit, and commit records:
`/root/workspace/mimo-opt/performance-results/dflash-compact-commits-20261028-3x9vu50x`.
