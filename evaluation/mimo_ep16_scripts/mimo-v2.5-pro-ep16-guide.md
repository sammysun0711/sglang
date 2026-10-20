## 1. Prepare docker image for gfx950 with AINIC
### dp0 container on node 0
```bash
docker run -it --name mimo-sgl-opt-gfx950-node0 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem --device=/dev/infiniband -v $HOME:/root/workspace  -v /data/models:/models  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/lib/x86_64-linux-gnu/libibverbs.so.1 -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so -v /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so -v /etc/libibverbs.d:/etc/libibverbs.d:ro rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```

### dp1 container on node 1
```bash
docker run -it --name mimo-sgl-opt-gfx950-node1 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem --device=/dev/infiniband -v $HOME:/root/workspace  -v /data/models:/models  -v /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/lib/x86_64-linux-gnu/libibverbs.so.1 -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so -v /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so -v /etc/libibverbs.d:/etc/libibverbs.d:ro rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```
## 2. Environment setup for both DP containers

### Clean up pre-installed environment
```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang
```bash
cd /sgl-workspace/
git clone https://github.com/sammysun0711/sglang -b mimo-opt
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```
### Install AITER & pyhip dependency
```bash
git clone https://github.com/sammysun0711/aiter -b mimo-opt
cd aiter
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL
FlyDSL installation should be after AITER installation, since it will update FlyDSL 0.2.4 runtime and FlyDSL FA & PA decode kernels.
```bash
git clone https://github.com/sammysun0711/FlyDSL -b mimo-opt
cd FlyDSL/wheels && python3 -m pip install --no-deps --force-reinstall flydsl-0.2.4-*.whl mimo_flydsl_kernels-0.1.3-*.whl
cd ../..
```

### Install MORI
Use the validated MORI 1.2.4 release package for better dispatch/combine performance.

```bash
pip uninstall -y amd_mori
pip install --no-deps amd_mori==1.2.4
```

### Check RDMA status
```bash
rdma link show
ibv_devinfo -v | head -30
```

### Setup passwordless ssh connection
```bash
git clone https://github.com/sammysun0711/llm-distributed-inference.git
cd llm-distributed-inference/sglang/
# Set up SSH access between the node 0 and node 1 containers in both directions.
./scripts/setup_docker_passwdless_ssh.sh mi355-gpu-14
```
Check that node 0 and node 1 can access each other via SSH without a password.
```shell
ssh mi355-gpu-14
```

## 3. Benchmark ep16 decode performance with fake-prefill

The two nodes form one TP16/EP16 service with DP2 and attention TP8 per node. Send requests to node 0 on port 30001.

Replace the example IPs, TCP interface, `MORI_RDMA_DEVICES` and `MORI_IB_GID_INDEX` with your cluster settings; both nodes must use the same `DIST_INIT_ADDR`.

### Start the benchmark server on both nodes

Run this setup in each container, using a new run ID for each experiment:
```bash
cd /workspace/sglang/evaluation/mimo_ep16_scripts
export DIST_INIT_ADDR=10.2.80.21:50000
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export RUN_ID=ep16_perf_01
export HIP_FORCE_DEV_KERNARG=1
```

On node 0:
```bash
NODE_RANK=0 SGLANG_HOST_IP=10.2.80.21 \
	bash ./launch_tp8_ep8_aiter_mori_mtp_decode_fake_prefill_baseline_2nodes.sh
```

On node 1, in a separate terminal:
```bash
NODE_RANK=1 SGLANG_HOST_IP=10.2.80.29 \
	bash ./launch_tp8_ep8_aiter_mori_mtp_decode_fake_prefill_baseline_2nodes.sh
```

The launcher enables dense TP1, DP LM head, fake expert routing and simulated MTP acceptance length3.

Wait for `The server is fired up and ready to roll!` on node 0. From a separate client terminal:
```bash
curl -fsS http://10.2.80.21:30001/health
```

### Run the performance matrix

Run on node 0:
```bash
cd /workspace/sglang/evaluation/mimo_ep16_scripts
HOST=10.2.80.21 PORT=30001 \
BENCHMARK_PRESET=ep16_baseline_highest \
OUTPUT_TOKENS=1024 WARMUP_REQUESTS=32 PROMPT_WAVES=4 \
LOG_DIR=./logs/ep16_perf_01/benchmark \
	bash ../run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

| Input tokens | Global concurrency | Full-load batch per DP |
| ---: | ---: | ---: |
| 65536 | 368 | 184 |
| 262144 | 128 | 64 |
| 524288 | 64 | 32 |
| 786432 | 46 | 23 |
| 1047548 | 32 | 16 |

The matrix uses seed12345, flushes after warmup, and sends four waves per case (2552 measured requests in total). Add `DRY_RUN=1` to print commands without sending requests.

Keep both nodes' server logs under `logs/decode_ep16_fake_prefill_baseline_<RUN_ID>/node<N>/`. For each case, extract its log interval from the successful cache flush to benchmark completion, separately for DP0/TP0 and DP1/TP8. Do not analyze a mixed five-case log with a smaller target batch. For a 64k DP0 slice:
```bash
python3 ../analyze_server_output_throughput.py ./logs/64k_dp0.log --target-bs 184
```

## 4. Accuracy validation

Stop the fake-prefill servers on both nodes first. Use fresh terminals so benchmark overrides do not leak into accuracy validation. [ep_baseline_gsm8k_acc_test_server.sh](ep_baseline_gsm8k_acc_test_server.sh) starts the server only; the EvalScope client below runs GSM8K.

### Start the real-prefill EP16 accuracy server

Run in both containers:
```bash
cd /workspace/sglang/evaluation/mimo_ep16_scripts
export EP_SIZE=16
export DIST_INIT_ADDR=10.2.80.21:50000
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export RUN_ID=ep16_gsm8k_01
export ENABLE_FAKE_EP_DISPATCH=0
export ENABLE_TBO=0
export MORI_EP_LAUNCH_CONFIG_MODE=MANUAL
unset MORI_EP_TUNING_CONFIG SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
```

On node 0:
```bash
NODE_RANK=0 SGLANG_HOST_IP=10.2.80.21 \
LOG_DIR=./logs/ep16_gsm8k_01/node0 \
	bash ./ep_baseline_gsm8k_acc_test_server.sh
```

On node 1:
```bash
NODE_RANK=1 SGLANG_HOST_IP=10.2.80.29 \
LOG_DIR=./logs/ep16_gsm8k_01/node1 \
	bash ./ep_baseline_gsm8k_acc_test_server.sh
```

### Run the complete GSM8K evaluation

Wait for server readiness and check `/health`. EvalScope 1.11.0 must be installed in the client environment (`python3 -m pip install evalscope==1.11.0` if absent). .

Run from a separate terminal on node 0:
```bash
cd /workspace/sglang/evaluation/mimo_ep16_scripts
BASE_URL=http://10.2.80.21:30001/v1 \
MODEL_PATH=/models/MiMo-V2.5-Pro/ \
MODEL_ID=MiMo-V2.5-Pro-EP16-MORI124 \
LIMIT=all EVAL_BATCH_SIZE=32 MAX_TOKENS=8192 SEED=12345 \
OUTPUT_DIR=./logs/ep16_gsm8k_01/gsm8k_full \
	bash ../run_evalscope_gsm8k_accuracy.sh
```

The client records server configuration, flushes the cache and evaluates all 1319 test examples with temperature0. Reports are under `gsm8k_full/evalscope/reports/`, with request/response artifacts in the EvalScope work directory.