## 1. Prepare docker image

### gfx942

```bash
docker run -it --name mimo-v26-flash-opt-gfx942 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  -v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510
```

## 2. Environment setup

### Clean up pre-installed environment

```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang

```bash
cd /sgl-workspace
git clone https://github.com/sammysun0711/sglang -b mimo-opt-mxfp4-dflash
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```

### Install AITER & pyhip dependency

```bash
git clone https://github.com/sammysun0711/aiter -b integration/mimo-fp4-dflash
cd aiter
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL

Install FlyDSL runtime version that compatible with aiter

```bash
pip install flydsl==0.3.2
```

## 3. Run baseline single node prefill benchmark

Baseline prefill keeps quick-reduce disabled, mixed router disabled, FlyDSL prefill disabled, Gluon decode, BF16 KV cache

### Launch TBO-off server

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
CHUNKED_PREFILL_SIZE=16384 \
MEM_FRACTION_STATIC=0.70 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=0 \
./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_accuracy_baseline.sh
```

### Run prefill benchmark

```bash
cd /sgl-workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```

Stop the TBO-off server, verify that the GPUs are clear, then launch a fresh
TBO-on server and run the same client matrix:

```bash
cd /sgl-workspace/sglang/evaluation/mimo_dflash_scripts
CHUNKED_PREFILL_SIZE=16384 \
MEM_FRACTION_STATIC=0.70 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
DISABLE_RADIX_CACHE=1 \
ENABLE_TWO_BATCH_OVERLAP=0 \
./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_accuracy_baseline.sh
```

The qualified TBO-on memory fraction is `0.75`. The `0.80` configuration can
exhaust transient HSA memory even though the static KV pools themselves are
not full.

## 4. Run baseline single node decode benchmark with fake prefill

The standalone fake-prefill launcher defines the Flash baseline settings:
TP4 x DP2, EP1, DFlash block 8, load-time MXFP4-to-FP8 MoE conversion,
Gluon attention, BF16 target/draft KV, and disabled Quick Reduce/mixed router.
It adds `--disaggregation-mode decode --disaggregation-transfer-backend fake`.
Acceptance remains real, matching the accuracy baseline; simulated acceptance
is not forced. Fake prefill skips prompt computation, so this is a performance
test rather than an accuracy evaluation.

The fake-prefill decode capacity must fit both the full-token and SWA-token
pools:

```text
(C + 1) * (ISL + OSL) <= max_total_num_tokens
(C + 1) * OSL         <= swa_layer_tokens
```

Apply these bounds per DP replica, using its admitted request count. Read
`max_total_num_tokens` from `/server_info` and `swa_layer_tokens` from
the server initialization log. If SWA is the smaller bound for a short-ISL
case, increase `SWA_FULL_TOKENS_RATIO` only until the SWA slot count is at
least the full-token slot count (or the desired concurrency cap). Allocating
more than that reduces the full-token pool without increasing concurrency.

### Launch server

```bash
cd /root/workspace/sglang/evaluation/mimo_v26_flash
MEM_FRACTION_STATIC=0.70 \
DISABLE_RADIX_CACHE=1 \
SWA_FULL_TOKENS_RATIO=0.03 \
MAX_RUNNING_REQUESTS=96 \
  ./launch_mimo_v26_flash_dp2_tp4_ep1_dflash_decode_fake_prefill_baseline.sh
```

Use `DRY_RUN=1` to inspect the command without starting the server. Supported
overrides such as `MODEL_PATH`, `DFLASH_BLOCK_SIZE`, `AITER_ROOT`, and
`CUDA_GRAPH_BS_DECODE` are configured in this script. Decode-specific controls include
`MAX_TOTAL_TOKENS`, `DECODE_LOG_INTERVAL`, `WATCHDOG_TIMEOUT`, and
`SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS` (defaults to the request limit).

### Run the matching decode point

```bash
cd /root/workspace/sglang/evaluation
MODEL=/models/MiMo-V2.6-Flash-MOPD/ \
CASE_SPECS='65536|65536|64k|128' \
PROMPT_WAVES=4 \
WARMUP_REQUESTS=32 \
./run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

The client uses explicit token IDs, so no four-token subtraction is needed.

### Run decode throughput analysis

```bash
python3 analyze_server_output_throughput.py \
  <path-to-server-log> \
  --target-bs <C> \
  --exact-bs \
  --min-tps 40 \
  --max-tpot 25
```

Use the analyzer's filtered server median as the reported output throughput.
The analyzer reports the following median-consistent derived values:

```text
TPS/request = filtered server median / C
TPOT (ms)   = 1000 * C / filtered server median
```

The analyzer also retains `mean_tpot_ms` and `mean_tps_per_request` as explicit
mean-based reference values.

## 5. Profiling & Analysis

```bash
cd /sgl-workspace/sglang/evaluation
./run_sglang_profile.sh
```

## 8. Run real-MTP ShareGPT dataset accuracy gate

```bash
cd /sgl-workspace/sglang/evaluation
./run_sharegpt_mtp_accuracy_test.sh
```

## 9. Run swe-bench & accuracy benchmark test

Follow up customer's swe-bench accuracy verification guide.

## 10. H200 performance evaluation

Follow up customer's shared performance data
