# MiMo FP8/FP4 DFlash Commit Split Plan

## Summary

  Create signed, reviewable commit series on:

- SGLang: /root/workspace/mimo-opt/sglang, branch mimo-opt
- AITER: /root/workspace/mimo-opt/aiter-mimo-fp4-dflash, branch integration/mimo-fp4-dflash

  Use git commit -s with:

  Signed-off-by: Xiake Sun <xiake.sun@amd.com>

  Do not add Co-authored-by, AI attribution, or GPG signing. Do not push. Keep MORI, MegaMoE, generated artifacts, customer logs, migration
  notes, and unrelated edits uncommitted.

## Preparation

1. Save binary working-tree and index patches plus git status under /tmp/mimo-commit-split-<timestamp></timestamp>/.
2. Record both starting SHAs and current staged/unstaged file lists.
3. In AITER, run a non-destructive mixed reset to clear the existing partial index; preserve all working-tree content.
4. Build each commit using whole-file staging where exclusive and curated git apply --cached patches where files contain multiple features.
5. Inspect every staged patch with git diff --cached --check and git diff --cached.
6. Verify each committed SHA in a detached temporary worktree so later uncommitted changes cannot mask missing dependencies.

## AITER Series

### 1. feat(mimo): enable gfx950 DFlash kernels

- Add native FP8 paged-prefill CSR routing, asymmetric D192/V128 paged decode, BF16 page-64 SWA, and Gluon fallback support.
- Add fused MiMo RMSNorm plus FP8 group quantization and the required public exports.
- Include paged-attention, SWA, asymmetric decode, graph replay, and fused-quant tests.
- Include only shared helper changes required by these kernels.

### 2. feat(mimo): enable native MXFP4 A8W4 MoE

- Add the TP8/EP1 MiMo A8W4 path using FlyDSL stage 1 and OPUS stage 2.
- Include selectable auto/fp8/bf16 stage-2 output and the E384 MiMo tuned rows.
- Include semantic changes corresponding to source commits 05056b7c, bcdcf762, and 977666f6.
- Include standard-path A8W4 correctness, output-format, and graph tests.
- Exclude E24/E48 sparse-EP rows and changes unique to 976e22e6/02319dda.

### 3. test(mimo): add DFlash and MXFP4 benchmark coverage

- Add reusable paged-decode benchmark entry points.
- Register stable paged-attention and SWA tests in AITER CI sharding.
- Keep correctness tests with their feature commits; this commit contains benchmark/CI wiring only.

### 4. perf(mimo): add optional PyHIP A8W8 kernels

- Add the xiaomi-355-opt PyHIP dependency and automatic MiMo GEMM/MoE dispatch.
- Add MiMo A8W8 tuned/untuned GEMM and BF16 FMoE CSVs.
- Include the prequantized scale-layout compatibility fix and tests.
- Keep this commit last and explicitly optional because current C16 FP8 measurements show stalls/regression.

  AITER exclusions:
- MegaMoE implementation and multigpu tests.
- MORI live-row/sparse-EP changes.
- profile_fmoe.csv.
- patches/.
- GLM CSV row deletions.
- Unrelated CI changes beyond registering the committed attention tests.

## SGLang Series

### 1. feat(mimo): enable FP8 target with DFlash

- Add DFlash target hidden-state capture, mask embedding, sinks, value scaling, simulated acceptance, and fake-prefill disaggregation.
- Add page-64 vectorized KV lifecycle, prefix-valid writes/moves, graph padding-page restoration, cached BF16 prefill, target verification,
  and draft SWA routing.
- Include fused RMS/QKV quantization and mixed-router support used by the optimized FP8 path.
- Include attention, DFlash, cache-lifetime, disaggregation, and fused-quant tests.

### 2. feat(mimo): enable native MXFP4 target experts

- Detect quant_method=fp8 plus store_dtype=mxfp4.
- Preserve packed FP4 weights and E8M0 scale bytes and map native scale parameter names.
- Route TP8/EP1 native MXFP4 through AITER with interleaved gate/up layout and configurable stage-2 output.
- Include loader, model-config, AITER runner, and server-argument tests.
- Exclude routed-only EP compatibility, MXFP8 MORI dispatch, and test_moriep_mxfp8_dispatch.py.

### 3. eval(mimo): add FP8 and FP4 DFlash performance gates

- Commit the maintained accuracy, fake-prefill decode, prefill benchmark, and profiling scripts.
- Preserve the fixed BF16-KV prefill gate: 64K/1, chunk 32K, C16, 64 prompts, four warmups, radix off, TBO off.
- Preserve the decode gate: 64K/1K, C96, 384 prompts, 32 warmups, fake prefill, simulated acceptance 4.
- Do not include check_acc_long_test.py.

  SGLang exclusions:
- /root/workspace/mimo-opt/sglang-mimo-fp4-dflash reference repository.
- MORI and MegaMoE integration.
- python/pyproject.toml and deletion of python/pyproject_other.toml.
- Untracked sgl-kernel HIP files.
- Customer documents, logs, profiles, archives, spreadsheets, and generated analysis output.
- Migration-plan Markdown.

## Commit Message Format

  Each commit uses a short body with two or three bullets:

  feat(mimo): enable native MXFP4 target experts

- preserve packed FP4 weights and E8M0 scales
- route TP8 experts through AITER A8W4 kernels
- add loader and dispatch coverage

  Signed-off-by: Xiake Sun <xiake.sun@amd.com>

  After every commit, verify that the message contains exactly one Signed-off-by trailer and no other attribution trailers.

## Validation

  AITER Milestone 1:

  python3 -m pytest -q
    op_tests/test_flydsl_paged_fmha.py
    op_tests/test_flydsl_paged_swa_bf16.py
    op_tests/test_flydsl_pa_decode.py
    op_tests/test_batch_prefill_asymmetric.py \

  AITER Milestone 2:

  python3 -m pytest -q
    op_tests/test_flydsl_moe_a8w4.py
    op_tests/test_moe_mxfp8_passthrough.py

  PyHIP optional commit:

- Run direct A8W8 GEMM shapes and MiMo MoE tokens 16384,32768,65536,131072.
- Regenerate /tmp/aiter_configs from empty state.
- Verify MiMo rows are selected automatically and source CSV hashes do not change.
- Record the known C16 performance concern in the commit summary.

  SGLang:
- Run model-config, MXFP4 loading, AITER runner, server-argument, DFlash, cache-lifetime, and vectorized-attention tests.
- Run target-only and DFlash TP8 server-ready smokes for FP8 and MXFP4.
- Run bash -n and dry runs for every committed evaluation script.
- After the primary series, rerun the fixed C16 prefill comparison and C96 decode gate.

  Final checks:
- git diff --check for each repository.
- Review each commit independently in a detached worktree.
- Confirm the final dirty state contains only intentionally excluded/deferred files.
- Produce a concise commit table with SHA, title, tests, and two-bullet summary.

## Progress

Safety snapshot: `/tmp/mimo-commit-split-20260919T133413Z`

| Repository | Order | Commit | Status | SHA | Validation |
|---|---:|---|---|---|---|
| AITER | 1 | `feat(mimo): enable gfx950 DFlash kernels` | committed | `a4874953` | Python compile; 424 tests collected with FlyDSL 0.3.2 |
| AITER | 2 | `feat(mimo): enable native MXFP4 A8W4 MoE` | committed | `29508acc` | Staged Python compile; focused GPU execution pending |
| AITER | 3 | `test(mimo): add DFlash and MXFP4 benchmark coverage` | committed | `b9a0e83a` | Benchmark dry-run and Python compile |
| AITER | 4 | `perf(mimo): add optional PyHIP A8W8 kernels` | committed | `d5dd21f3` | Python compile; 10 tests collected; C16 perf remains gating follow-up |
| SGLang | 1 | `feat(mimo): enable FP8 target with DFlash` | committed | `8208385df2` | Python compile; 102 tests collected before environment import failure |
| SGLang | 2 | `feat(mimo): enable native MXFP4 target experts` | committed | `70a370f8e3` | 85 tests and 10 subtests passed |
| SGLang | 3 | `eval(mimo): add FP8 and FP4 DFlash performance gates` | committed | this commit | `bash -n` on all five scripts |

The progress table is updated after each commit. Deferred and excluded work remains present in the original working trees and is not staged, deleted, or reverted.

Collection note: the SGLang DFlash collection environment imported AITER from
`/root/workspace/mimo-opt/aiter`, whose FlyDSL-facing modules expect a different
runtime API (`flydsl.expr.vector`) than installed FlyDSL 0.3.2. This is an
environment/worktree mismatch, not a Python syntax failure in the staged SGLang
files.
