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
pip uninstall sglang sglang-kernel sgl-kernel amd-aiter flydsl -y
rm -rf /sgl-workspace/sglang /sgl-workspace/aiter
```

### Install sglang
```bash
cd /root/workspace
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
git -C 3rdparty/composable_kernel apply < patches/composable_kernel/mimo_fp8_page64_head192_batch_prefill.patch
pip install -e .
cd ..
```

### Install FlyDSL
FlyDSL installation should be after AITER installation, since it will update FlyDSL 0.2.4 runtime and FlyDSL FA & PA decode kernels.
```bash
git clone https://github.com/sammysun0711/FlyDSL -b mimo-opt
cd FlyDSL/wheels && python3 -m pip install --no-deps --force-reinstall flydsl-0.2.4-*.whl mimo_flydsl_kernels-*.whl
cd ../..
```

## 3. Run single node prefill benchmark
### Launch server
```bash
cd /root/workspace/sglang/evaluation
./launch_tp8_noep_aiter_mtp_accuracy.sh
```
### Run prefill benchmark
```bash
./run_benchmark_mimo_pro_prefill.sh
```

## 4. Run single node decode benchmark with fake prefill
### Launch server
```bash
cd /root/workspace/sglang/evaluation
./launch_tp8_noep_aiter_mtp_decode_fake_prefill.sh
```

### Run decode benchmark
```bash
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


## 6. Run real-MTP ShareGPT dataset accuracy gate
```bash
cd /root/workspace/sglang/evaluation
./run_sharegpt_mtp_accuracy_test.sh
```

## 7. Run swe-bench & accuracy benchmark test
Follow up customer's swe-bench accuracy verification guide.

## 8. H200 performance evaluation
Follow up customer's shared performance data
