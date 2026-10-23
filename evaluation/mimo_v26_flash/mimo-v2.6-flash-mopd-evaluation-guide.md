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
MEM_FRACTION_STATIC=0.75 \
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
MEM_FRACTION_STATIC=0.75 \
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
Acceptance is simulated at length 4 using `SGLANG_SIMULATE_ACC_LEN=4` and
`SGLANG_SIMULATE_ACC_METHOD=match-expected`, matching the Pro fake-prefill
reference. This gives a controlled acceptance length for decode throughput
comparisons. Fake prefill skips prompt computation, so these runs measure
performance; use the accuracy launcher with real prefill to evaluate accuracy.

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

Use `DRY_RUN=1` to inspect the command without starting the server.

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
At the 1M context boundary, reserve additional slots for the scheduler:
use `CASE_SPECS='1047548|1047548|1m|18'` for 1,024 output tokens. An input
length of 1,047,552 leaves no scheduler margin and is capped at 1,022 output
tokens. This margin is separate from tokenizer special tokens.

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

## 8. Run GSM8K accuracy and acceptance gate

Use the real-prefill accuracy launcher with acceptance simulation disabled.
The runner uses upstream `/server_info` statistics and requires no SGLang
source changes. It rejects decode-only servers.

Run one command for the complete
GSM8K `main` test split, 4-shot prompting, and each worker's mean acceptance length >= 4:

```bash
pip install evalscope==1.11.0
cd /sgl-workspace/sglang/evaluation
MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/ \
MIN_ACCEPT_LENGTH=4.0 \
python3 ./run_evalscope_gsm8k_accuracy.py
```

This runs EvalScope once, then reads each worker's upstream
`avg_spec_accept_length` (including the bonus token). Every worker's mean
must meet `MIN_ACCEPT_LENGTH`; their arithmetic mean is also reported and
is explicitly unweighted. Raw counts for an exact global weighted mean are
not available from this API.

`FLUSH_CACHE=1` is required for the acceptance check. Use a dedicated server
with no concurrent inference or statistics resets during evaluation. The
upstream API does not expose simulation settings, so real acceptance must
be configured in the server launcher.

`MIN_GSM8K_ACCURACY=0.97` optionally enforces 97% accuracy;
`MIN_ACCEPT_LENGTH=4.0` sets the acceptance threshold for every worker.

## 9. Run swe-bench & accuracy benchmark test

Follow up customer's swe-bench accuracy verification guide.

## 10. H200 performance evaluation

Follow up customer's shared performance data
