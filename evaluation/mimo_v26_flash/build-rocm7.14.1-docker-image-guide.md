# Build MiMo-V2.6-Flash for MI300X on ROCm 7.14.1

Build MiMo-V2.6-Flash-MOPD for eight 304-CU AMD Instinct MI300X GPUs (gfx942) with ROCm 7.14.1 and PyTorch 2.12. The image includes DFlash, AITER, PyHIP, FlyDSL, MoRI, the SGLang router for disaggregated serving, and image, audio and video input support. Mount the checkpoint at runtime. [Dockerfile](./rocm.Dockerfile)

## 1. Build the image

Run the build from the SGLang repository root using Docker BuildKit.

```bash
IMAGE=mimo-flash:rocm7.14.1-gfx942-sglang
docker build --progress=plain -f docker/rocm.Dockerfile \
  --target mimo-rocm7141 --build-arg GPU_ARCH=gfx942 \
  -t "$IMAGE" -t mimo-flash:rocm7141-gfx942-sglang \
  -t mimo-flash:rocm7.14.1-gfx942-sgl-0acf8c1-aiter-e595aec .
docker image inspect "$IMAGE" --format '{{.Id}} {{json .Config.Labels}}'
docker run --rm "$IMAGE" python -m pip check
```

The Docker target `mimo-rocm7141` selects the ROCm 7.14.1 multimodal image. `GPU_ARCH` selects the AMD hardware architecture for compilation and defaults to `gfx942` in this target. Select `--target mimo-rocm7141-serving` for the serving layer alone, or `--target mimo-rocm7141-toolchain` for the compiler environment. The build can run on a CPU-only host. The serving node JIT-compiles workload-specific AITER kernels on first use.

The Dockerfile fetches `mimo-opt-mxfp4-dflash` from the SGLang fork and `integration/mimo-fp4-dflash` from the AITER fork and checks out the full commit IDs in section 2. To use another revision, pass a commit reachable from the selected branch using `SGL_COMMIT` or `AITER_COMMIT`. The repository and branch arguments are `SGL_REPO` / `SGL_BRANCH` and `AITER_REPO` / `AITER_BRANCH`.

### Custom MoRI repo

The default uses the CPython 3.12 Linux x86-64 wheel for `amd_mori==1.2.4`. Its embedded build information reports HIP `7.14.60850`, its ROCm compatibility pattern is `7.14.*`, and the Dockerfile verifies the wheel SHA-256. Its native bindings also require Noble's `libgrpc++1.51t64` and its runtime dependencies. To build a fork against the image's own SDK, set both arguments below. Supply the customer's clone URL and a full commit SHA from that repository.

```bash
: "${MORI_REPO:?Set MORI_REPO to your fork clone URL}"
: "${MORI_COMMIT:?Set MORI_COMMIT to a full commit SHA}"
docker build --progress=plain -f docker/rocm.Dockerfile --target mimo-rocm7141 --build-arg GPU_ARCH=gfx942 \
  --build-arg MORI_REPO="$MORI_REPO" --build-arg MORI_COMMIT="$MORI_COMMIT" \
  -t mimo-flash:rocm7141-gfx942-customer-mori .
```

## 2. Component comparison for ROCm 7.2.0 and 7.14.1

The comparison uses two immutable image IDs inspected on 22 November 2026. The 7.2.0 image is AMD’s published SGLang development image. The 7.14.1 image is the **final MiMo multimodal image** built with `--target mimo-rocm7141`. The historical benchmarks also used packages installed after launching the old image. [7.2.0 MiMo deployment changes](#what-the-earlier-720-mimo-deployment-installed)

| Image identity | 7.2.0 image | 7.14.1 MiMo image |
| --- | --- | --- |
| Tag inspected | `rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510` | `mimo-flash:rocm7.14.1-gfx942-sglang` |
| Local image ID | `sha256:bb9d2e5ab1a6b66bff693e9a5d972be378e6059dc2058066aed70ebc3ed3993b` | `sha256:8692fde3d0411b33e47f90f008d4b0cd329ab6e83b148a1169166c87dc0a588d` |
| Registry provenance | Published image digest `sha256:8f8dc55398855764d339ff4e2695b9c3c298704999051acaa1a67993fdff62e5` | Built locally from `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` at base digest `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1` |
| Platform / system C library | Ubuntu 22.04.5 amd64 / glibc 2.35 | Ubuntu 24.04.4 amd64 / glibc 2.39 |
| Python used by SGLang | `3.10.12` in `/opt/venv` | `3.12.3` in `/opt/venv` |
| ROCm / HIP build reported by Torch | 7.2.0 / `7.2.26015-fc0010cf6a` | 7.14.1 / `7.14.60850` |
| SDK installation | Ubuntu ROCm packages, `/opt/rocm` → `/opt/rocm-7.2.0` | TheRock wheels `rocm-sdk-core`, `rocm-sdk-libraries` and `rocm-sdk-devel` 7.14.1, `/opt/rocm` → `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel` |

The filterable inventory contains **1,650 side-by-side records**. Each row identifies the package, export or checkout, its version or Git revision in each image, and whether it changed. An empty cell means that the corresponding record is absent from that image. [Complete component inventory (TSV)](./rocm7.2.0-vs-7.14.1-components.tsv)

| Inventory source | 7.2.0 | 7.14.1 | Distinct records across both |
|---|---:|---:|---:|
| Python distributions visible to the serving interpreter | 226 | 263 | 317 |
| Installed Debian packages, keyed by package and architecture | 1,046 | 558 | 1,233 |
| ROCm and LLVM SDK CMake package exports | 43 | 58 | 65 |
| Git checkouts and materialized submodules under `/sgl-workspace` and `/opt` | 35 | 23 | 35 |

The Python rows were checked against `python3 -m pip list`. The Debian rows come from `dpkg-query` with status `installed`. A ROCm library packaged as a Debian package in 7.2.0 can be supplied by a TheRock wheel in 7.14.1. Use the CMake rows to compare the native library versions across that packaging change.

### Serving framework, kernels and communications

| Component | 7.2.0 installed version and source | 7.14.1 installed version and source |
| --- | --- | --- |
| PyTorch | `2.9.1+rocm7.2.0.lw.git7e1940d4` · `7e1940d4b11fd6128be4c42ba41567ec5ab87102` | `2.12.0+rocm7.14.1` · `098172edc887399a593f0c74ae5c40b1fe043801` |
| Triton | `3.6.0+git42270451` · `42270451990532c67e69d753fbd026f28fcc4840` | `3.8.0+git4cff872c.rocm7.14.1` · vendor wheel revision `4cff872c` |
| torchvision | `0.24.0+rocm7.2.0.gitb919bd0c` · `b919bd0c56abbb3c5ca056a3a458af9fd1cabf52` | `0.27.0+rocm7.14.1` · `b3d5b111f54044b216013162706ca107aa7a4ec4` |
| torchaudio | `2.9.0+rocm7.2.0.gite3c6ee2b` · `e3c6ee2b6588b7cd27a84182de74bf12fe043831` | `2.11.0+rocm7.14.1` · `34c52a67e8941bbd8e6adaca0eb0b9eabec11d78` |
| SGLang | `0.5.11.dev20260510+g2f0686712` · workspace HEAD `59faf986b28b1efb5a0733075b627b452b976f1f` | `0.0.0.dev14219+g0acf8c172.d20261120` · `0acf8c172e9e035f47ac9cd0e806ee415c34e07c` (`sammysun0711/sglang`, `mimo-opt-mxfp4-dflash`) |
| SGL Kernel (`sglang-kernel`) | `0.4.2.post1` | `0.4.3` built from the pinned SGLang checkout |
| AITER (`amd-aiter`) | `0.1.12.post2.dev150+ga6bb49937.d20260510` · `a6bb499375849eec45d68c5ccaebc8865fd422c0` | `0.1.1.dev3145+ge595aec16.d20261120` · `e595aec16c3c77469ad9d02f2a7c665e5cef1543` (`sammysun0711/aiter`, `integration/mimo-fp4-dflash`) |
| Composable Kernel vendored by AITER | `fdf4bb7fcc984811cef48ce817d89aac064b984a` | `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` plus `mimo_page64_qk192_v128_batch_prefill.patch` |
| PyHIP | — | `1.0.1.dev447+gdb0b1cfc4` · `db0b1cfc498d470765dc970b44328192fa6ecd6b` (`tingqli/pyhip`) |
| FlyDSL | `0.1.5.dev504` | `0.3.2` · release tag `dd8373349e33638c8d158a9ea009b7b28598984a` |
| MoRI (`amd-mori`) | `1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee` | `1.2.4` · release `f39fa765e897c95995957d0518908cc982a88fc2` · CPython 3.12 wheel SHA-256 `52dd1ee81e3f0485e4ddc90e99b42dba64ceb2500ee08564a7ba2f9d1a5ef98c` |
| SGLang Model Gateway (`sglang-router`) | `0.3.2` built from the old SGLang checkout | `0.3.2` PyPI abi3 wheel · SHA-256 `ba5a24951e5e0357fe390782fcd1abd861448a87216832dbf2e501a1dd1548eb` |
| AMD SMI Python (`amdsmi`) | `26.2.1+fc0010cf6a` | `26.5.0+ca887ee8` installed from the TheRock development SDK |

The 7.2.0 SGLang distribution reports `g2f0686712` in its package version. Its workspace checkout is at `59faf986b2`. The 7.14.1 native EP8 decode performance run used PR49’s runtime changes in a separate container with MoRI 1.2.4. [Qualification report](./rocm7.14-qualification-report.md#ep8-decode)

### ROCm libraries, profiling and build tools

| Component | 7.2.0 image | 7.14.1 MiMo image |
| --- | --- | --- |
| RCCL (also verified through Torch) | `2.27.7` | `2.30.4` |
| ROCr HSA runtime / COMGR | `1.18.70200` / `3.0.0` | `1.21.0` / `3.3.0` |
| rocBLAS / hipBLAS / hipBLASLt | `5.2.0` / `3.2.0` / `1.2.1` | `5.5.0` / `3.5.0` / `1.4.1` |
| MIOpen | `3.5.1` | `3.5.2` |
| rocFFT / hipFFT | `1.0.36` / `1.0.22` | `1.0.38` / `1.0.24` |
| rocRAND / hipRAND | `4.2.0` / `3.1.0` | `4.5.0` / `3.4.0` |
| rocSOLVER / hipSOLVER | `3.32.0` / `3.2.0` | `3.35.0` / `3.5.0` |
| rocSPARSE / hipSPARSE / hipSPARSELt | `4.2.0` / `4.2.0` / `0.2.6` | `4.7.0` / `4.6.0` / `0.2.9` |
| rocPRIM / hipCUB / rocWMMA | `4.2.0` / `4.2.0` / `2.2.0` | `4.5.0` / `4.5.0` / `2.2.1` |
| ROCprofiler-SDK / rocprofiler-register | `1.1.0` / `0.6.0` | `1.3.2` / `0.6.0` |
| Profiler commands on `PATH` | `rocprof`, `rocprofv2`, `rocprofv3` | `rocprofv3` |
| Kineto registration / host BLAS | `ROCP_TOOL_LIBRARIES` unset | `ROCP_TOOL_LIBRARIES` points to Torch’s `libtorch_cpu.so`. Its loader dependency `librocm-openblas.so.0` is in `/opt/rocm/lib/host-math/lib` (OpenBLAS SDK header `0.3.31.dev`). |
| AMD Clang | `22.0.0git` · `7b800a19466229b8479a78de19143dc33c3ab9b5` | `23.0.0git` · `f30ae3e6b680ad23dcf522e668c6d4c5c90838d0` with vendor patch `440716f8b87be9d8e20ed910e10e5b6d14d57cf6` |
| GCC / CMake on `PATH` | 11.4.0 / 3.31.10 | 13.3.0 / 3.28.3 |
| Ninja on `PATH` | `1.13.0.git.kitware.jobserver-pipe-1` | `1.13.0.git.kitware.jobserver-pipe-1` |
| Go compiler | 1.22.2 in the final image | 1.23.8 in the Mooncake builder only |
| Native gRPC C++ runtime | `libgrpc++1` `1.30.2-3build6` | `libgrpc++1.51t64` `1.51.1-4.1build5` for the MoRI wheel |
| InfiniBand verbs (`libibverbs1` / `libibverbs-dev`) | `39.0-1` / `39.0-1` | `50.0-2ubuntu0.2` / `50.0-2ubuntu0.2` |
| Open MPI / libfabric Debian packages | `openmpi-bin` `4.1.2-2ubuntu1` / `libfabric1` `1.11.0-3` | — |

### Model loading, serving and multimedia dependencies

| Component (versions in the order shown) | 7.2.0 image | 7.14.1 MiMo image |
| --- | --- | --- |
| Transformers / compressed-tensors | `5.6.0` / `0.15.0.1` | `5.8.1` / `0.16.0` |
| HF Hub / Safetensors | `1.14.0` / `0.8.0rc0` | `1.33.0` / `0.8.0` |
| Tokenizers / SentencePiece / tiktoken | `0.22.2` / `0.2.1` / `0.12.0` | `0.22.2` / `0.2.3` / `0.14.0` |
| NumPy / SciPy / TorchAO | `2.2.6` / `1.15.3` / `0.9.0` | `2.3.5` / `1.18.1` / `0.9.0` |
| xgrammar / outlines-core / llguidance | `0.2.0` / `0.1.26` / `0.7.30` | `0.2.1` / `0.1.26` / `0.7.30` |
| OpenAI client / FastAPI / Uvicorn | `2.6.1` / `0.136.1` / `0.46.0` | `2.6.1` / `0.165.0` / `0.54.0` |
| Python gRPC / protobuf / PyZMQ | `1.80.0` / `6.33.6` / `27.1.0` | `1.85.3` / `7.36.2` / `27.3.0` |
| apache-tvm-ffi / z3-solver | `0.5.11.dev20260510+g2f0686712` / `4.15.4.0` | `0.1.9` / `4.15.4.0` |
| Pillow / PyAV (`av`) / SoundFile | `11.3.0` / `17.0.1` / `0.13.1` | `12.3.0` / `19.0.2` / `0.13.1` |
| TorchCodec / Decord | Neither installed | `0.13.0` / — |
| OpenCV headless / MoviePy | `4.10.0.84` / `2.2.1` | Neither installed |
| Diffusers / Accelerate / PEFT | `0.37.0` / `1.13.0` / `0.18.1` | All three absent |
| System FFmpeg / libavcodec | CLI absent, `libavcodec58` `7:4.4.2-0ubuntu0.22.04.1` | CLI `6.1.1`, `libavcodec60` `7:6.1.1-3ubuntu5` |
| libavcodec reported by PyAV | 62.11.100 | 63.1.102 |
| eSpeak NG / DejaVu core fonts | — / `2.37-2build1` | `1.51+dfsg-12build1` / `2.37-8` |

The 7.14.1 vendor base supplies Torch, torchvision, torchaudio, Triton and Pillow. The MiMo build adds the serving packages and the CPU TorchCodec 0.13.0 wheel (release `6e692e9f9f55470cc013914c8cc69ec7ae2eefea`). The system FFmpeg packages and PyAV’s linked FFmpeg libraries have separate version numbers. A dash in these tables indicates the listed package or command is absent from the inspected image.

### What the earlier 7.2.0 MiMo deployment installed

The earlier evaluation guide started from the 7.2.0 image above and replaced its SGLang and AITER trees. These are the deployment changes relevant when comparing historical measurements with a new container. [7.2.0 evaluation guide](https://github.com/sammysun0711/sglang/blob/9f954b61345ea9fb4deaa273d5c5a0982cd55724/evaluation/mimo_v26_flash/mimo-v2.6-flash-mopd-evaluation-guide.md)

| Component | 7.2.0 MiMo deployment | 7.14.1 MiMo image |
| --- | --- | --- |
| SGLang / AITER / PyHIP | Installed the `mimo-opt-mxfp4-dflash` and `integration/mimo-fp4-dflash` branches, AITER’s CK patch and its PyHIP dependency. The guide names moving branches. PR52 records SGLang `a547dcd4d518c6a84f119207f37b7d3a90c4e512` and AITER `108940c2b0aa6bc767b456f6f4c103046e651aef` for its run. [PR52 environment](https://github.com/sammysun0711/sglang/pull/52) | Installs the pinned SGLang, AITER, CK and PyHIP revisions in the serving stage. |
| FlyDSL / MoRI | Installed `0.3.2` / `1.2.4` over the original versions. | Ships `0.3.2` / `1.2.4`. |
| Media decoding | Built CPU TorchCodec `0.9.0` at `ce82c7cab6b19f0aeb58c4f85ee990756cfc9f86`, installed Decord `0.6.0`, FFmpeg and eSpeak NG, and used PyAV `17.0.1` and SoundFile `0.13.1`. | Ships CPU TorchCodec `0.13.0`, FFmpeg, eSpeak NG, PyAV `19.0.2` and SoundFile `0.13.1`. The multimodal tests use TorchCodec. |

### Source checkouts and distributed-serving tools

Both images retain `sglang`, `aiter`, `Mooncake` and `fast-hadamard-transform` under `/sgl-workspace`. The 7.2.0 image also contains `triton-custom` and `mori` there. In the 7.14.1 image, Triton comes from the vendor wheel and the default MoRI comes from a PyPI wheel. Both images also carry `/opt/tilelang`.

| Location in the 7.2 image | Installed version and source | Location and use in the 7.14.1 image |
|---|---|---|
| `/sgl-workspace/mori` | `amd_mori 1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee` | Wheel **1.2.4** in the serving venv for `--moe-a2a-backend mori` (EP8). EP1 selects `none`. |
| `/sgl-workspace/Mooncake` | Git tag `v0.3.7.post2` · `b6a841dc78c707ec655a563453277d969fb8f38d`. The old image contains native `mooncake.engine` and `mooncake.store` modules and `mooncake_master`. | Included at the same commit, with HIP and etcd support, the transfer engine, distributed store and `mooncake_master`. Select `--disaggregation-transfer-backend mooncake` on both workers for Mooncake KV transfer. |
| `/sgl-workspace/fast-hadamard-transform` | `fast_hadamard_transform 1.0.4.post1` · HIP fork `46efb7d776d38638fc39f3c803eaee3dd7016bd1` | Included at the same commit, compiled against the image’s ROCm PyTorch for clients such as the DSA indexer. |
| `/opt/tilelang` | Installed metadata `0.1.7.post3+cuda.gita55a8230` · `a55a82302bf7f3c5af635b5c9146f728185cc900` | Included at the same commit with `USE_ROCM=ON` and `USE_CUDA=OFF`. The upstream version provider emits the historical `+cuda.gita55a8230` suffix for this CMake build. The Python runtime uses `apache-tvm-ffi==0.1.9`. |
| `/sgl-workspace/sglang/sgl-model-gateway` | `sglang-router 0.3.2`, built from the old SGLang tree at `59faf986b28b1efb5a0733075b627b452b976f1f` | The MiMo image includes the 0.3.2 release wheel. Start the gateway for prefill/decode disaggregation. For ordinary single-node DP serving, clients can use the SGLang HTTP endpoint directly. |

Mooncake is installed by CMake. Check its version with `git -C /sgl-workspace/Mooncake describe --tags --exact-match`. On MI300X, the SGLang Mooncake wrapper transferred 4 KiB between two processes over TCP, and the native master and store completed a 4 KiB put/get. Hadamard matched a CPU reference for FP32 and BF16. TileLang compiled and ran a HIP vector-add with all 256 results correct. Multi-node GPU/RDMA transport requires a separate test on the deployment hosts.

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

Point `MODEL_DIR` at a complete MiMo-V2.6-Flash-MOPD checkpoint, including its tokenizer, processor configuration, `audio_tokenizer/`, and `dflash/` draft weights. The following commands publish the API on host loopback and use the TP8/DP2/EP1 launcher with DP attention, BF16 KV cache, and DFlash block size 8. It converts the MXFP4 experts to FP8 at load time by default. Keep the same model revision and launch settings when comparing releases.

```bash
MODEL_DIR=/data/models/MiMo-V2.6-Flash-MOPD
test -r "$MODEL_DIR/config.json" && test -r "$MODEL_DIR/dflash/config.json"
docker run --rm --device=/dev/kfd --device=/dev/dri "$IMAGE" python -c '
import torch, aiter, pyhip
gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
print([(d.name, d.gcnArchName, d.multi_processor_count) for d in gpus])
assert len(gpus) == 8 and all(d.gcnArchName.split(":")[0] == "gfx942" and d.multi_processor_count == 304 for d in gpus)
'
docker run -d --name mimo-flash --ipc=host --device=/dev/kfd --device=/dev/dri \
  -p 127.0.0.1:30001:30001 -e SERVER_HOST=0.0.0.0 -e SERVER_PORT=30001 \
  -v "$MODEL_DIR:/models/MiMo-V2.6-Flash-MOPD:ro" \
  "$IMAGE" bash /sgl-workspace/sglang/evaluation/mimo_v26_flash/launch_mimo_v26_flash_dp2_tp8_ep1_dpa_accuracy_baseline.sh
docker logs -f mimo-flash
```

In another shell, check readiness, then send a completion. Cold-start time includes model loading, JIT compilation and launcher warmup.

```bash
curl -fsS http://127.0.0.1:30001/health
curl -fsS http://127.0.0.1:30001/v1/models
curl -fsS http://127.0.0.1:30001/generate -H 'Content-Type: application/json' \
  -d '{"text":"Complete the sequence: 1, 2, 3,","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

For MoRI EP8, use `launch_mimo_v26_flash_dp2_tp8_ep8_dpa_accuracy_baseline.sh` in the same directory and pass `-e PERFORMANCE_MODE=0` to Docker. That launcher's default `PERFORMANCE_MODE=1` enables fake expert routing for benchmarks. The separate fake-prefill decode launchers simulate prompt processing and acceptance. Use the real-prefill launcher for customer requests and accuracy comparisons. [Launchers and evaluation commands](https://github.com/sammysun0711/sglang/tree/0acf8c172e9e035f47ac9cd0e806ee415c34e07c/evaluation/mimo_v26_flash)

**EP8 native-MXFP4 fake-prefill decode is blocked at the pinned revision.** During the first decode graph capture, all eight schedulers exited with `AttributeError: 'MoriEPNormalDispatchOutput' object has no attribute 'topk_output'`. PR48 measured a source tree that included PR49’s native MoRI A8W4 implementation. The image pins SGLang `0acf8c172e`, which lacks that implementation. Upgrading only `amd-mori` to 1.2.5 reproduced the same exception on all eight ranks. **Keep MoRI 1.2.4 for this issue.** With PR49’s SGLang runtime changes applied to the same image and `MAX_TOTAL_TOKENS=13000000` for draft graph memory, native-MXFP4 EP8 completed all ten 4K–1M fake-prefill decode points. Both DP workers met 25 ms TPOT at 10/10 points. The EP8 real-prefill launcher with converted FP8 weights passed the multimodal and accuracy tests. [Qualification report](./rocm7.14-qualification-report.md) · [PR48 dependencies](https://github.com/sammysun0711/sglang/pull/48) · [PR49](https://github.com/sammysun0711/sglang/pull/49)

The EP8 fake-prefill launcher is also packaged with mode `0644`. Use Bash to inspect its native configuration. Direct execution exits 126 before Python starts.

```bash
cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
DRY_RUN=1 bash launch_mimo_v26_flash_dp2_tp8_ep8_dpa_decode_fake_prefill_baseline.sh
```

## 5. Multimodal dependencies and feature checks

| Input or check | Reused from base or serving layer | Added in the multimodal layer |
|---|---|---|
| Image and vision preprocessing | Base Pillow 12.3.0 and ROCm torchvision | DejaVu fonts for the test images |
| Video frames and embedded audio tracks | PyAV 19.0.2 (`av`) from SGLang's HIP runtime | TorchCodec 0.13.0 CPU decoders, FFmpeg shared libraries and CLI |
| WAV, MP3 and mel features | Base torchaudio and runtime SoundFile 0.13.1 | TorchCodec audio decoder, `espeak-ng` for the synthetic speech fixtures |

The image uses the branch's preferred TorchCodec path for video and audio. Model inference uses the GPU. [TorchCodec compatibility](https://github.com/pytorch/torchcodec/tree/v0.13.0) · [MiMo audio processor](https://github.com/sammysun0711/sglang/blob/0acf8c172e9e035f47ac9cd0e806ee415c34e07c/python/sglang/srt/multimodal/processors/mimo_audio.py)

To check the codecs without loading the weights, run the fixture generator and decode its output in the same ephemeral container.

```bash
docker run --rm "$IMAGE" bash -ec '
  cd /sgl-workspace/sglang/evaluation/mimo_v26_flash
  python run_mimo_multimodal_feature_checks.py --prepare-only
  cd .cache/mimo_multimodal_media
  python - <<"CHECK"
from torchcodec.decoders import AudioDecoder, VideoDecoder
from PIL import Image
assert Image.open("apple.png").size == (512, 512)
assert VideoDecoder("forward.mp4")[0].shape == (3, 512, 512)
assert len(VideoDecoder("reverse.mp4")) == 8
assert AudioDecoder("secret.mp3").get_all_samples().data.numel() > 0
assert AudioDecoder("short_audio_tail.mp4").get_all_samples().data.numel() > 0
CHECK
'
```

To exercise model inference after the section 4 server is ready, send the bundled image, audio, video and mixed requests to its loopback endpoint. Both EP1 and EP8 with real routing passed all 18 checks in the ROCm 7.14.1 qualification. The client writes per-case responses and a summary inside the container. [Qualification report](./rocm7.14-qualification-report.md)

```bash
docker exec -w /sgl-workspace/sglang/evaluation/mimo_v26_flash mimo-flash \
  python3 run_mimo_multimodal_feature_checks.py --output /tmp/mimo-mm-results
```

## 6. Capture GPU kernels with the SGLang profiler

Start the server from the full image or the `mimo-rocm7141-serving` stage. In a second shell in that container, run the profiling client after the server is ready.

```bash
cd /sgl-workspace/sglang/evaluation
HOST=127.0.0.1 PORT=30001 PROFILE_OUTPUT_DIR=/tmp/mimo-profiles ./run_sglang_profile.sh
```

Open a per-rank `.trace.json.gz` in Perfetto and inspect the GPU stream tracks. On eight MI300X GPUs, the TP8/DP2/EP1 real-prefill launcher and four 8,192-input / 20-output requests produced **219,930 GPU kernel events across eight traces**, including **28,066 on TP0**. Two earlier captures collected CPU operators and zero GPU kernels.

The Dockerfile sets `ROCP_TOOL_LIBRARIES` to PyTorch’s `libtorch_cpu.so` so ROCprofiler-SDK can register Kineto when HIP initializes. The ROCm host-math directory supplies its loader dependency `librocm-openblas.so.0`. An eager matmul/add on the stock ROCm 7.14.1 PyTorch image, which has no `/opt/rocm` symlink, also lost kernel events when HIP initialized before PyTorch. The SDK symlink can stay in place. [Serving-stage defaults](./rocm.Dockerfile)

For a container created from an earlier build, export both settings in the **server shell**, then relaunch SGLang and run the same client.

```bash
export ROCP_TOOL_LIBRARIES=/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
export LD_LIBRARY_PATH="/opt/rocm/lib/host-math/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
