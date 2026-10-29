# Consolidated FlyPA / CK handoff code

This package preserves the existing `sglang-mimo-ck-prefill` working state
and adds the files needed by `flypa-optimization-handoff.md`. GPU hardware
is unavailable; consolidation validation is CPU-only. The original
production-source edits were preserved.

When using the committed `feat/mimo-ck-linear-prefill` branch, the SGLang
changes and supporting files are already present: skip the SGLang patch
application below. Set `MIMO_PKG` to this directory to use the included
`patches/aiter-local.patch` and `patches/composable-kernel-local.patch`.
The separate consolidation archive also contains `sglang-handoff.patch`,
the source overlay and `validation.json` for restoring from the base revision.

`sglang-handoff.patch` includes all eight modified files and both untracked
files from the supplied status, plus handoff scripts/configuration/results.
It is based on SGLang `a15045ff186a5bbd39661a15b05cae676ea9dda9`.
That revision was verified at remote branch `feat/mimo-flash-dp-attention`.

| Files from the supplied status | Role |
|---|---|
|`aiter_backend.py`, `aiter_utils.py`|Tuned PA decode policy and CK linear-KV prefill integration; the handoff enables both flags.|
|`flydsl_pa.py`|Manual-scheduling switch and graph-safe work-counter initialization; existing experimental knobs retained unchanged.|
|`launch_mimo_v26_flash_dp_attention.sh`|Portable AITER path and existing executable mode.|
|`test/manual/test_mimo_ck_linear_prefill.py`|Required CK integration tests; included despite being untracked.|
|`communicator.py`, `dp_attention.py`|Existing optional DP transport experiments; retained for exact restoration, disabled in the handoff.|
|`scheduler.py`, `scheduler_components/profiler_manager.py`|Existing full-TP profiler-boundary support.|
|`batch_overlap/dp_collectives.py`|Untracked experimental provider, currently not selected by serving; retained rather than lost.|

The worktree previously lacked the handoff, noVS CSV, `test_prefill_candidate.py`,
and standalone reproducer. They are now included. The existing client scripts,
`attention/utils.py`, `flydsl_helpers.py`, and registered attention tests are
already in the base commit and are checked by the file manifest.

To restore on a healthy machine, use a fresh checkout and set these paths:

```bash
export MIMO_ROOT=/data/xisun/rocm7.2.4
export MIMO_SRC="$MIMO_ROOT/sglang-ck-restored"
export MIMO_PKG=/path/to/extracted/flypa_handoff_code
git clone --branch feat/mimo-flash-dp-attention \
  https://github.com/sammysun0711/sglang "$MIMO_SRC"
git -C "$MIMO_SRC" switch -c feat/mimo-ck-linear-prefill \
  a15045ff186a5bbd39661a15b05cae676ea9dda9
git -C "$MIMO_SRC" apply --check --index "$MIMO_PKG/sglang-handoff.patch"
git -C "$MIMO_SRC" apply --index "$MIMO_PKG/sglang-handoff.patch"
python "$MIMO_SRC/evaluation/mimo_v26_flash/check_flypa_handoff_files.py" \
  --repo "$MIMO_SRC"
```

`--index` stages the patch, including support files matched by ignore rules;
it does not create a commit. The checker accepts a later commit that contains
the base revision and still matches the recorded file hashes.

`sources/sglang/` is an overlay of the consolidated files, not a replacement
for the full base repository. `manifest.json` records every included path,
SHA-256, base revision and dependency. `validation.json` records the clean
checkout application and CPU checks performed for this package.

The external code state is included in two additional patches:

- AITER base: `2b81433bbff27020ce98271b992a1aab93a5f3db`, verified at remote
  branch `integration/mimo-fp4-dflash`. `patches/aiter-local.patch` preserves
  the existing LL128 scratch-stride change and its untracked test. This is
  a separate communication change, not a new FlyPA optimization.
- Composable Kernel base: `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`, the
  AITER submodule revision. `patches/composable-kernel-local.patch` preserves
  all three modified CK files, including MiMo page64/QK192/V128 support.

For a fresh AITER checkout:

```bash
git clone --branch integration/mimo-fp4-dflash \
  https://github.com/sammysun0711/aiter "$MIMO_ROOT/aiter"
git -C "$MIMO_ROOT/aiter" checkout 2b81433bbff27020ce98271b992a1aab93a5f3db
git -C "$MIMO_ROOT/aiter" submodule update --init 3rdparty/composable_kernel
git -C "$MIMO_ROOT/aiter" apply --check --index "$MIMO_PKG/patches/aiter-local.patch"
git -C "$MIMO_ROOT/aiter" apply --index "$MIMO_PKG/patches/aiter-local.patch"
git -C "$MIMO_ROOT/aiter/3rdparty/composable_kernel" apply --check --index \
  "$MIMO_PKG/patches/composable-kernel-local.patch"
git -C "$MIMO_ROOT/aiter/3rdparty/composable_kernel" apply --index \
  "$MIMO_PKG/patches/composable-kernel-local.patch"
```

The CK patch above is the full current submodule diff. Do not also apply the
older `mimo_page64_qk192_v128_batch_prefill.patch` to that patched tree.
Build/install the restored code in the matching environment on the healthy
node; this task did not import GPU libraries or rebuild them on the broken node.

Model weights, the 672,837,942-byte ShareGPT file, installed Python packages,
and the multi-GB `runtime724` Torch overlay are external and not included in
this code package. Their paths/hashes are recorded where applicable. Transfer
or recreate them, then check file completeness without touching the GPU:

```bash
export AITER_ROOT="$MIMO_ROOT/aiter"
export MIMO_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
python "$MIMO_SRC/evaluation/mimo_v26_flash/check_flypa_handoff_files.py" \
  --repo "$MIMO_SRC" --workspace "$MIMO_ROOT" --dependencies \
  --dataset "$MIMO_DATASET" --model-root /models/MiMo-V2.6-Flash-MOPD
```

Then follow `evaluation/mimo_v26_flash/flypa-optimization-handoff.md` in the
restored checkout. The historical 5 CK tests, 11 FlyPA cases and E2E results
are evidence from before the hardware problem; rerun those gates after restore.
