# Build MiMo-V2.6-Flash for MI300X on ROCm 7.14.1

Build MiMo-V2.6-Flash-MOPD for eight 304-CU AMD Instinct MI300X GPUs (gfx942) with ROCm 7.14.1 and PyTorch 2.12. The image includes DFlash, AITER, PyHIP, FlyDSL, MoRI, the SGLang router for disaggregated serving, and image, audio and video input support. Mount the checkpoint at runtime. [Dockerfile](../../docker/rocm.Dockerfile)

## 1. Build the image

Run the build from the root of the SGLang checkout containing the MiMo Dockerfile. The runtime checkout installed inside the image comes from the merged `mimo-opt-mxfp4-dflash` branch.

```bash
IMAGE=mimo-flash:rocm7.14.1-gfx942-sgl-f2cab77-aiter-e595aec
docker build --progress=plain -f docker/rocm.Dockerfile \
  --target mimo-rocm7141 --build-arg GPU_ARCH=gfx942 \
  -t "$IMAGE" .
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
docker run --rm "$IMAGE" python -m pip check
```

The Docker target `mimo-rocm7141` selects the ROCm 7.14.1 multimodal image. `GPU_ARCH` selects the AMD hardware architecture for compilation and defaults to `gfx942` in this target. Select `--target mimo-rocm7141-serving` for the serving layer alone, or `--target mimo-rocm7141-toolchain` for the compiler environment. The build can run on a CPU-only host. The serving node JIT-compiles workload-specific AITER kernels on first use.

The Dockerfile fetches `mimo-opt-mxfp4-dflash` from the SGLang fork and `integration/mimo-fp4-dflash` from the AITER fork and checks out the full commit IDs in section 2. To use another revision, pass a commit reachable from the selected branch using `SGL_COMMIT` or `AITER_COMMIT`. The repository and branch arguments are `SGL_REPO` / `SGL_BRANCH` and `AITER_REPO` / `AITER_BRANCH`.

### Custom MoRI repo

The default uses the CPython 3.12 Linux x86-64 wheel for `amd_mori==1.2.4`. Its embedded build information reports HIP `7.14.60850`, its ROCm compatibility pattern is `7.14.*`, and the Dockerfile verifies the wheel SHA-256. Its native bindings also require Noble's `libgrpc++1.51t64` and its runtime dependencies. The source-build example below defaults to the upstream MoRI 1.2.4 release. To build a customer fork against the image's SDK, set `MORI_REPO` to its clone URL and `MORI_COMMIT` to a full commit SHA from that fork.

```bash
export MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
export MORI_COMMIT="${MORI_COMMIT:-f39fa765e897c95995957d0518908cc982a88fc2}"
docker build --progress=plain -f docker/rocm.Dockerfile --target mimo-rocm7141 --build-arg GPU_ARCH=gfx942 \
  --build-arg MORI_REPO="$MORI_REPO" --build-arg MORI_COMMIT="$MORI_COMMIT" \
  -t mimo-flash:rocm7141-gfx942-customer-mori .
```

## 2. Component comparison for ROCm 7.2.0 and 7.14.1

The comparison uses two immutable image between previous ROCm 7.2.0 base image and current target ROCm 7.14.1 target base image. The 7.2.0 image is AMD’s published SGLang development image. The 7.14.1 image is the **final MiMo multimodal image** built with `--target mimo-rocm7141`. The historical benchmarks also used packages installed after launching the old image. [7.2.0 MiMo deployment changes](#what-the-earlier-720-mimo-deployment-installed)

| Image identity                     | 7.2.0 image                                                                                        | 7.14.1 MiMo image                                                                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tag inspected                      | `rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510`                                                    | `mimo-flash:rocm7.14.1-gfx942-sgl-f2cab77-aiter-e595aec`                                                                                                                          |
| Local image ID                     | `sha256:bb9d2e5ab1a6b66bff693e9a5d972be378e6059dc2058066aed70ebc3ed3993b`                        | `sha256:7ac2b614a076bf636434ada6ef12a6ef960e1205ca70d40121afc57353b40afa`                                                                                                         |
| Registry provenance                | Published image digest `sha256:8f8dc55398855764d339ff4e2695b9c3c298704999051acaa1a67993fdff62e5` | Built locally from `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` at base digest `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1` |
| Platform / system C library        | Ubuntu 22.04.5 amd64 / glibc 2.35                                                                  | Ubuntu 24.04.4 amd64 / glibc 2.39                                                                                                                                                   |
| Python used by SGLang              | `3.10.12` in `/opt/venv`                                                                       | `3.12.3` in `/opt/venv`                                                                                                                                                         |
| ROCm / HIP build reported by Torch | 7.2.0 / `7.2.26015-fc0010cf6a`                                                                   | 7.14.1 / `7.14.60850`                                                                                                                                                             |
| SDK installation                   | Ubuntu ROCm packages, `/opt/rocm` → `/opt/rocm-7.2.0`                                         | TheRock wheels `rocm-sdk-core`, `rocm-sdk-libraries` and `rocm-sdk-devel` 7.14.1, `/opt/rocm` → `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`                 |

### Serving framework, kernels and communications

| Component                                | 7.2.0 installed version and source                                                               | 7.14.1 installed version and source                                                                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch                                  | `2.9.1+rocm7.2.0.lw.git7e1940d4` · `7e1940d4b11fd6128be4c42ba41567ec5ab87102`               | `2.12.0+rocm7.14.1` · `098172edc887399a593f0c74ae5c40b1fe043801`                                                                                                |
| Triton                                   | `3.6.0+git42270451` · `42270451990532c67e69d753fbd026f28fcc4840`                            | `3.8.0+git4cff872c.rocm7.14.1` · vendor wheel revision `4cff872c`                                                                                               |
| torchvision                              | `0.24.0+rocm7.2.0.gitb919bd0c` · `b919bd0c56abbb3c5ca056a3a458af9fd1cabf52`                 | `0.27.0+rocm7.14.1` · `b3d5b111f54044b216013162706ca107aa7a4ec4`                                                                                                |
| torchaudio                               | `2.9.0+rocm7.2.0.gite3c6ee2b` · `e3c6ee2b6588b7cd27a84182de74bf12fe043831`                  | `2.11.0+rocm7.14.1` · `34c52a67e8941bbd8e6adaca0eb0b9eabec11d78`                                                                                                |
| SGLang                                   | `0.5.11.dev20260510+g2f0686712` · workspace HEAD `59faf986b28b1efb5a0733075b627b452b976f1f` | `0.0.0.dev14220+gf2cab7731.d20261123` · `f2cab7731ce64a1b07799ed2d637f20c6dbb0830` (`sammysun0711/sglang`, `mimo-opt-mxfp4-dflash`)                         |
| SGL Kernel (`sglang-kernel`)           | `0.4.2.post1`                                                                                  | `0.4.3` built from the pinned SGLang checkout                                                                                                                      |
| AITER (`amd-aiter`)                    | `0.1.12.post2.dev150+ga6bb49937.d20260510` · `a6bb499375849eec45d68c5ccaebc8865fd422c0`     | `0.1.1.dev3145+ge595aec16.d20261123` · `e595aec16c3c77469ad9d02f2a7c665e5cef1543` (`sammysun0711/aiter`, `integration/mimo-fp4-dflash`)                     |
| Composable Kernel vendored by AITER      | `fdf4bb7fcc984811cef48ce817d89aac064b984a`                                                     | `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` plus `mimo_page64_qk192_v128_batch_prefill.patch`                                                                     |
| PyHIP                                    | —                                                                                               | `1.0.1.dev448+g05142fb77` · `05142fb773deee77ce024165dd4b986dc4f2cf23` (`tingqli/pyhip`)                                                                      |
| FlyDSL                                   | `0.1.5.dev504`                                                                                 | `0.3.2` · release tag `dd8373349e33638c8d158a9ea009b7b28598984a`                                                                                                |
| MoRI (`amd-mori`)                      | `1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee`                                        | `1.2.4` · release `f39fa765e897c95995957d0518908cc982a88fc2` · CPython 3.12 wheel SHA-256 `52dd1ee81e3f0485e4ddc90e99b42dba64ceb2500ee08564a7ba2f9d1a5ef98c` |
| SGLang Model Gateway (`sglang-router`) | `0.3.2` built from the old SGLang checkout                                                     | `0.3.2` PyPI abi3 wheel · SHA-256 `ba5a24951e5e0357fe390782fcd1abd861448a87216832dbf2e501a1dd1548eb`                                                            |
| AMD SMI Python (`amdsmi`)              | `26.2.1+fc0010cf6a`                                                                            | `26.5.0+ca887ee8` installed from the TheRock development SDK                                                                                                       |

### ROCm libraries, profiling and build tools

| Component                                               | 7.2.0 image                                                      | 7.14.1 MiMo image                                                                                                                                                                           |
| ------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RCCL (also verified through Torch)                      | `2.27.7`                                                       | `2.30.4`                                                                                                                                                                                  |
| ROCr HSA runtime / COMGR                                | `1.18.70200` / `3.0.0`                                       | `1.21.0` / `3.3.0`                                                                                                                                                                      |
| rocBLAS / hipBLAS / hipBLASLt                           | `5.2.0` / `3.2.0` / `1.2.1`                                | `5.5.0` / `3.5.0` / `1.4.1`                                                                                                                                                           |
| MIOpen                                                  | `3.5.1`                                                        | `3.5.2`                                                                                                                                                                                   |
| rocFFT / hipFFT                                         | `1.0.36` / `1.0.22`                                          | `1.0.38` / `1.0.24`                                                                                                                                                                     |
| rocRAND / hipRAND                                       | `4.2.0` / `3.1.0`                                            | `4.5.0` / `3.4.0`                                                                                                                                                                       |
| rocSOLVER / hipSOLVER                                   | `3.32.0` / `3.2.0`                                           | `3.35.0` / `3.5.0`                                                                                                                                                                      |
| rocSPARSE / hipSPARSE / hipSPARSELt                     | `4.2.0` / `4.2.0` / `0.2.6`                                | `4.7.0` / `4.6.0` / `0.2.9`                                                                                                                                                           |
| rocPRIM / hipCUB / rocWMMA                              | `4.2.0` / `4.2.0` / `2.2.0`                                | `4.5.0` / `4.5.0` / `2.2.1`                                                                                                                                                           |
| ROCprofiler-SDK / rocprofiler-register                  | `1.1.0` / `0.6.0`                                            | `1.3.2` / `0.6.0`                                                                                                                                                                       |
| Profiler commands on `PATH`                           | `rocprof`, `rocprofv2`, `rocprofv3`                        | `rocprofv3`                                                                                                                                                                               |
| Kineto registration / host BLAS                         | `ROCP_TOOL_LIBRARIES` unset                                    | `ROCP_TOOL_LIBRARIES` points to Torch’s `libtorch_cpu.so`. Its loader dependency `librocm-openblas.so.0` is in `/opt/rocm/lib/host-math/lib` (OpenBLAS SDK header `0.3.31.dev`). |
| AMD Clang                                               | `22.0.0git` · `7b800a19466229b8479a78de19143dc33c3ab9b5`    | `23.0.0git` · `f30ae3e6b680ad23dcf522e668c6d4c5c90838d0` with vendor patch `440716f8b87be9d8e20ed910e10e5b6d14d57cf6`                                                                |
| GCC / CMake on `PATH`                                 | 11.4.0 / 3.31.10                                                 | 13.3.0 / 3.28.3                                                                                                                                                                             |
| Ninja on `PATH`                                       | `1.13.0.git.kitware.jobserver-pipe-1`                          | `1.13.0.git.kitware.jobserver-pipe-1`                                                                                                                                                     |
| Go compiler                                             | 1.22.2 in the final image                                        | 1.23.8 in the Mooncake builder only                                                                                                                                                         |
| Native gRPC C++ runtime                                 | `libgrpc++1` `1.30.2-3build6`                                | `libgrpc++1.51t64` `1.51.1-4.1build5` for the MoRI wheel                                                                                                                                |
| InfiniBand verbs (`libibverbs1` / `libibverbs-dev`) | `39.0-1` / `39.0-1`                                          | `50.0-2ubuntu0.2` / `50.0-2ubuntu0.2`                                                                                                                                                   |
| Open MPI / libfabric Debian packages                    | `openmpi-bin` `4.1.2-2ubuntu1` / `libfabric1` `1.11.0-3` | —                                                                                                                                                                                          |

### Model loading, serving and multimedia dependencies

| Component (versions in the order shown) | 7.2.0 image                                               | 7.14.1 MiMo image                                    |
| --------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| Transformers / compressed-tensors       | `5.6.0` / `0.15.0.1`                                  | `5.8.1` / `0.16.0`                               |
| HF Hub / Safetensors                    | `1.14.0` / `0.8.0rc0`                                 | `1.33.0` / `0.8.0`                               |
| Tokenizers / SentencePiece / tiktoken   | `0.22.2` / `0.2.1` / `0.12.0`                       | `0.22.2` / `0.2.3` / `0.14.0`                  |
| NumPy / SciPy / TorchAO                 | `2.2.6` / `1.15.3` / `0.9.0`                        | `2.3.5` / `1.18.1` / `0.9.0`                   |
| xgrammar / outlines-core / llguidance   | `0.2.0` / `0.1.26` / `0.7.30`                       | `0.2.1` / `0.1.26` / `0.7.30`                  |
| OpenAI client / FastAPI / Uvicorn       | `2.6.1` / `0.136.1` / `0.46.0`                      | `2.6.1` / `0.165.0` / `0.54.0`                 |
| Python gRPC / protobuf / PyZMQ          | `1.80.0` / `6.33.6` / `27.1.0`                      | `1.85.3` / `7.36.2` / `27.3.0`                 |
| apache-tvm-ffi / z3-solver              | `0.5.11.dev20260510+g2f0686712` / `4.15.4.0`          | `0.1.9` / `4.15.4.0`                             |
| Pillow / PyAV (`av`) / SoundFile      | `11.3.0` / `17.0.1` / `0.13.1`                      | `12.3.0` / `19.0.2` / `0.13.1`                 |
| TorchCodec / Decord                     | Neither installed                                         | `0.13.0` / —                                      |
| OpenCV headless / MoviePy               | `4.10.0.84` / `2.2.1`                                 | Neither installed                                    |
| Diffusers / Accelerate / PEFT           | `0.37.0` / `1.13.0` / `0.18.1`                      | All three absent                                     |
| System FFmpeg / libavcodec              | CLI absent, `libavcodec58` `7:4.4.2-0ubuntu0.22.04.1` | CLI `6.1.1`, `libavcodec60` `7:6.1.1-3ubuntu5` |
| libavcodec reported by PyAV             | 62.11.100                                                 | 63.1.102                                             |
| eSpeak NG / DejaVu core fonts           | — / `2.37-2build1`                                     | `1.51+dfsg-12build1` / `2.37-8`                  |

The 7.14.1 vendor base supplies Torch, torchvision, torchaudio, Triton and Pillow. The MiMo build adds the serving packages and the CPU TorchCodec 0.13.0 wheel (release `6e692e9f9f55470cc013914c8cc69ec7ae2eefea`). The system FFmpeg packages and PyAV’s linked FFmpeg libraries have separate version numbers. A dash in these tables indicates the listed package or command is absent from the inspected image.

### Source checkouts and distributed-serving tools

Both images retain `sglang`, `aiter`, `Mooncake` and `fast-hadamard-transform` under `/sgl-workspace`. The 7.2.0 image also contains `triton-custom` and `mori` there. In the 7.14.1 image, Triton comes from the vendor wheel and the default MoRI comes from a PyPI wheel. Both images also carry `/opt/tilelang`.

| Location in the 7.2 image                   | Installed version and source                                                                                                                                                        | Location and use in the 7.14.1 image                                                                                                                                                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/sgl-workspace/mori`                     | `amd_mori 1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee`                                                                                                                  | Wheel **1.2.4** in the serving venv for `--moe-a2a-backend mori` (EP8). EP1 selects `none`.                                                                                                                           |
| `/sgl-workspace/Mooncake`                 | Git tag `v0.3.7.post2` · `b6a841dc78c707ec655a563453277d969fb8f38d`. The old image contains native `mooncake.engine` and `mooncake.store` modules and `mooncake_master`. | Included at the same commit, with HIP and etcd support, the transfer engine, distributed store and `mooncake_master`. Select `--disaggregation-transfer-backend mooncake` on both workers for Mooncake KV transfer.         |
| `/sgl-workspace/fast-hadamard-transform`  | `fast_hadamard_transform 1.0.4.post1` · HIP fork `46efb7d776d38638fc39f3c803eaee3dd7016bd1`                                                                                    | Included at the same commit, compiled against the image’s ROCm PyTorch for clients such as the DSA indexer.                                                                                                                    |
| `/opt/tilelang`                           | Installed metadata `0.1.7.post3+cuda.gita55a8230` · `a55a82302bf7f3c5af635b5c9146f728185cc900`                                                                                 | Included at the same commit with `USE_ROCM=ON` and `USE_CUDA=OFF`. The upstream version provider emits the historical `+cuda.gita55a8230` suffix for this CMake build. The Python runtime uses `apache-tvm-ffi==0.1.9`. |
| `/sgl-workspace/sglang/sgl-model-gateway` | `sglang-router 0.3.2`, built from the old SGLang tree at `59faf986b28b1efb5a0733075b627b452b976f1f`                                                                             | The MiMo image includes the 0.3.2 release wheel. Start the gateway for prefill/decode disaggregation. For ordinary single-node DP serving, clients can use the SGLang HTTP endpoint directly.                                   |

Mooncake is installed by CMake. On MI300X, the SGLang Mooncake wrapper transferred 4 KiB between two processes over TCP, and the native master and store completed a 4 KiB put/get. Hadamard matched a CPU reference for FP32 and BF16. TileLang compiled and ran a HIP vector-add with all 256 results correct. Multi-node GPU/RDMA transport requires a separate test on the deployment hosts.

## 3. Verify the TheRock SDK

The vendor base includes the gfx942 device libraries and a `hipcc` shim. The toolchain stage installs `rocm[devel]==7.14.1` from the AMD multi-arch index for compilation and runtime JIT. In this SDK, `rocm-sdk path --cmake` names the **CMake configuration directory** and `rocm-sdk path --root` names its installation prefix. The toolchain stage creates `/opt/rocm` as a symlink directly to that prefix. PyTorch extensions, AITER, MoRI, and PyHIP use this path for builds and runtime JIT. [AMD 7.14.1 development wheels](https://repo.amd.com/rocm/whl-multi-arch/rocm-sdk-devel/)

```bash
docker run --rm "$IMAGE" bash -ec '
  rocm-sdk path --root
  rocm-sdk path --cmake
  test -f "$(rocm-sdk path --cmake)/hip-lang/hip-lang-config.cmake"
  test -L /opt/rocm && test "$ROCM_HOME" = /opt/rocm
  test "$(readlink -f /opt/rocm)" = "$(rocm-sdk path --root)"
  hipcc --version
  python -m pip check
'
```

## 4. Start real-prefill serving

Use the `IMAGE` from section 1 and a complete MiMo-V2.6-Flash-MOPD checkpoint, including the tokenizer, processor config, `audio_tokenizer/` and `dflash/`. On the serving host, check the model mount and GPU access first.

```bash
MODEL_DIR=/data/models/MiMo-V2.6-Flash-MOPD
test -r "$MODEL_DIR/config.json" && test -r "$MODEL_DIR/dflash/config.json"
docker run --rm --device=/dev/kfd --device=/dev/dri "$IMAGE" python -c '
import torch, aiter, pyhip
gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
print([(g.name, g.gcnArchName, g.multi_processor_count) for g in gpus])
assert len(gpus) == 8 and all(g.gcnArchName.split(":")[0] == "gfx942" and g.multi_processor_count == 304 for g in gpus)
'
```

The EP1 launcher uses global TP8, DP2 attention, converted FP8 experts, BF16 KV and DFlash block size 8. For EP8 real serving, select `launch_mimo_v26_flash_dp2_tp8_ep8_dpa_accuracy_baseline.sh` below. The command sets `PERFORMANCE_MODE=0` for real expert routing on EP8 and publishes the API on host loopback.

```bash
LAUNCHER=launch_mimo_v26_flash_dp2_tp8_ep1_dpa_accuracy_baseline.sh
docker run -d --rm --name mimo-flash --ipc=host --device=/dev/kfd --device=/dev/dri \
  -p 127.0.0.1:30001:30001 -e SERVER_HOST=0.0.0.0 -e SERVER_PORT=30001 -e PERFORMANCE_MODE=0 \
  -v "$MODEL_DIR:/models/MiMo-V2.6-Flash-MOPD:ro" \
  -w /sgl-workspace/sglang/evaluation/mimo_v26_flash "$IMAGE" "./$LAUNCHER"
docker logs -f mimo-flash
```

After model loading, JIT compilation and warmup, check readiness and request a completion from a second host shell.

```bash
curl -fsS http://127.0.0.1:30001/health
curl -fsS http://127.0.0.1:30001/v1/models
curl -fsS http://127.0.0.1:30001/generate -H 'Content-Type: application/json' \
  -d '{"text":"Complete the sequence: 1, 2, 3,","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

**Native MXFP4 EP8 decode.** The section 1 image ships A8W4 MOE path in SGLang `f2cab7731c` with PyHIP `05142fb773` and MoRI 1.2.4. All 10,368 measured fake-prefill requests completed across 4K to 1M, with a worst-worker TPOT of 21.19 ms. Preview the launcher below, then use the dedicated decode server for the benchmark.

```bash
docker run --rm -e DRY_RUN=1 -e MAX_TOTAL_TOKENS=13000000 \
  -w /sgl-workspace/sglang/evaluation/mimo_v26_flash "$IMAGE" \
  ./launch_mimo_v26_flash_dp2_tp8_ep8_dpa_decode_fake_prefill_baseline.sh
```

## 5. Check multimodal serving

The full image decodes media on the CPU and runs model inference on the GPU.

| Input | Components                                                           | Functional check                                           |
| ----- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| Image | Pillow, ROCm torchvision, DejaVu fonts for fixtures                  | Read text and colors from single and multiple images       |
| Audio | torchaudio, SoundFile, TorchCodec 0.13.0, `espeak-ng` for fixtures | Recognize spoken content in WAV and MP3                    |
| Video | PyAV, TorchCodec 0.13.0, FFmpeg                                      | Identify frame order and speech in an embedded audio track |

With the section 4 server ready, run the client from the host.

```bash
docker exec -w /sgl-workspace/sglang/evaluation/mimo_v26_flash mimo-flash \
  python run_mimo_multimodal_feature_checks.py --output /tmp/mimo-mm-results
```

A successful run reports **`18/18 checks passed`**. The suite checks image, audio, video and concurrent mixed requests on both DP workers. The output directory also contains `summary.json` and the per-case responses. The rebuilt image passed all 18 checks on both EP1 and EP8 with real routing.

<details>
<summary>Check the media decoders before loading the model</summary>

Run this fixture check from the host. It uses the full image and requires no GPU or checkpoint.

```bash
docker run --rm -w /sgl-workspace/sglang/evaluation/mimo_v26_flash "$IMAGE" bash -ec '
  python run_mimo_multimodal_feature_checks.py --prepare-only
  cd .cache/mimo_multimodal_media
  python - <<"CHECK"
from PIL import Image
from torchcodec.decoders import AudioDecoder, VideoDecoder
assert Image.open("apple.png").size == (512, 512)
assert VideoDecoder("forward.mp4")[0].shape == (3, 512, 512)
assert len(VideoDecoder("reverse.mp4")) == 8
for media in ("secret.mp3", "short_audio_tail.mp4"):
    assert AudioDecoder(media).get_all_samples().data.numel() > 0
print("Image, audio and video decoders passed")
CHECK
'
```

</details>

## 6. Capture GPU kernels with the SGLang profiler

With the section 4 server ready, run the profiler client inside the serving container and copy the traces to the host. Open a per-rank `.trace.json.gz` in Perfetto and inspect a GPU stream track.

```bash
docker exec -w /sgl-workspace/sglang/evaluation \
  -e HOST=127.0.0.1 -e PORT=30001 -e PROFILE_OUTPUT_DIR=/tmp/mimo-profiles \
  mimo-flash ./run_sglang_profile.sh
docker cp mimo-flash:/tmp/mimo-profiles ./mimo-profiles
```

A matmul/add probe on the rebuilt image captured three GPU kernel events with HIP initialized before PyTorch. In the `f2cab77` EP1 full-server run, four 8,192-input / 20-output requests produced **219,930 kernel events across eight ranks**, including **28,066 on TP0**. [Profiling defaults in the Dockerfile](../../docker/rocm.Dockerfile)

 `ROCP_TOOL_LIBRARIES` registers PyTorch's Kineto tool when HIP initializes. The host-math directory supplies its loader dependency `librocm-openblas.so.0`. The stock ROCm 7.14.1 base reproduced the same initialization-order issue without the `/opt/rocm` symlink.

```bash
export ROCP_TOOL_LIBRARIES=/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
export LD_LIBRARY_PATH="/opt/rocm/lib/host-math/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
