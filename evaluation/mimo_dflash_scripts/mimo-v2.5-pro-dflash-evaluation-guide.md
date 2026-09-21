## 1. Prepare docker image
### gfx942
```bash
docker run -it --name mimo-sgl-opt-gfx942 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  -v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510
```

### gfx950
```bash
docker run -it --name mimo-sgl-opt-gfx950 --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  -v /data/models:/models  rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510
```

## 2. Environment setup

### Clean up pre-installed environment
```bash
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl mimo-flydsl-kernels -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang
```bash
cd /root/workspace
git clone https://github.com/sammysun0711/sglang -b 
cd sglang && pip install --upgrade pip && cd sgl-kernel && python3 setup_rocm.py install
cd ..  && rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"
cd ..
```
### Install AITER & pyhip dependency
```bash
git clone https://github.com/sammysun0711/aiter -b integration/mimo-fp4-dflash
cd aiter
git submodule update --init 3rdparty/composable_kernel
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

### Launch server
```bash
cd /root/workspace/sglang/evaluation/mimo_dflash_scripts
./launch_tp8_noep_aiter_dflash_accuracy_baseline.sh
```
### Run prefill benchmark
```bash
cd /root/workspace/sglang/evaluation/
./run_benchmark_mimo_pro_prefill.sh
```
## 4. Run baseline single node decode benchmark with fake prefill
Baseline fake-prefill decode keeps quick-reduce disabled, mixed router disabled, FlyDSL prefill disabled, Gluon decode, BF16 KV cache

### Launch server
```bash
cd /root/workspace/sglang/evaluation/mimo_dflash_scripts
./launch_tp8_noep_aiter_mtp_decode_fake_prefill_baseline.sh
```

### Run decode benchmark
```bash
cd /root/workspace/sglang/evaluation
./run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh
```

### Run decode throughput analysis
```python
python3 analyze_server_output_throughput.py <path-to-server-log>
```

## 5. Profiling & Analysis
```bash
cd /root/workspace/sglang/evaluation
./run_sglang_profile.sh
```


## 8. Run real-MTP ShareGPT dataset accuracy gate
```bash
cd /root/workspace/sglang/evaluation
./run_sharegpt_mtp_accuracy_test.sh
```

## 9. Run swe-bench & accuracy benchmark test
Follow up customer's swe-bench accuracy verification guide.

## 10. H200 performance evaluation
Follow up customer's shared performance data
