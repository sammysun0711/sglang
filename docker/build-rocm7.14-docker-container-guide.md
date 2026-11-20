# Build MiMo-V2.6-Flash for MI300X on ROCm 7.14.1

Build MiMo-V2.6-Flash-MOPD for eight 304-CU AMD Instinct MI300X GPUs (gfx942) with ROCm 7.14.1 and PyTorch 2.12. The image includes DFlash, AITER, PyHIP, FlyDSL, MoRI, the SGLang router for disaggregated serving, and image, audio and video input support. Mount the checkpoint at runtime. [Dockerfile](./rocm.Dockerfile)

## 1. Build the image

Run the build from the SGLang repository root using Docker BuildKit.

```bash
IMAGE=mimo-flash:rocm7.14.1-gfx942-sgl-0acf8c1-aiter-e595aec
docker build --progress=plain -f docker/rocm.Dockerfile \
  --target mimo-rocm7141 --build-arg GPU_ARCH=gfx942 \
  -t "$IMAGE" -t mimo-flash:rocm7141-gfx942-sglang .
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

## 2. Component versions and provenance

The first column of versions is the unmodified 7.2.0 image. The previous MiMo deployment installed additional packages on top of it, recorded below the table. The last column identifies what this recipe selects. The Triton wheel reports a shortened Git revision. The target base digest identifies the exact wheel set.

| Component | Previous base | 7.14.1 base or build selection |
|---|---|---|
| Docker image | `rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510` | `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` |
| Docker repository digest | `sha256:8f8dc55398855764d339ff4e2695b9c3c298704999051acaa1a67993fdff62e5` | `sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1` |
| OS / Python | Ubuntu 22.04.5 / 3.10.12 | Ubuntu 24.04.4 / 3.12.3 |
| ROCm / Torch HIP build | 7.2.0 / `7.2.26015-fc0010cf6a` | 7.14.1 / `7.14.60850` |
| PyTorch | `2.9.1+rocm7.2.0.lw.git7e1940d4` · `7e1940d4b11fd6128be4c42ba41567ec5ab87102` | `2.12.0+rocm7.14.1` · `098172edc887399a593f0c74ae5c40b1fe043801` |
| Triton | `3.6.0+git42270451` · `42270451990532c67e69d753fbd026f28fcc4840` | `3.8.0+git4cff872c.rocm7.14.1` · `4cff872c` from the vendor build |
| RCCL (reported by Torch) | 2.27.7 | 2.30.4 |
| AMD Clang | 22.0.0git · `7b800a19466229b8479a78de19143dc33c3ab9b5` | 23.0.0git · `f30ae3e6b680ad23dcf522e668c6d4c5c90838d0` with vendor patch `440716f8b87be9d8e20ed910e10e5b6d14d57cf6` |
| AMD SMI Python | `26.2.1+fc0010cf6a` | `26.5.0+ca887ee8`, installed from the discovered SDK |
| torchvision | `0.24.0+rocm7.2.0.gitb919bd0c` · `b919bd0c56abbb3c5ca056a3a458af9fd1cabf52` | `0.27.0+rocm7.14.1` · `b3d5b111f54044b216013162706ca107aa7a4ec4` |
| torchaudio | `2.9.0+rocm7.2.0.gite3c6ee2b` · `e3c6ee2b6588b7cd27a84182de74bf12fe043831` | `2.11.0+rocm7.14.1` · `34c52a67e8941bbd8e6adaca0eb0b9eabec11d78` |
| SGLang | `0.5.11.dev20260510+g2f0686712` · checkout `59faf986b28b1efb5a0733075b627b452b976f1f` | `sammysun0711/sglang`, branch `mimo-opt-mxfp4-dflash` · `0acf8c172e9e035f47ac9cd0e806ee415c34e07c` |
| SGL Kernel (`sglang-kernel`) | `0.4.2.post1` | `0.4.3`, built from the same SGLang commit |
| AITER (`amd-aiter`) | `0.1.12.post2.dev150+ga6bb49937.d20260510` · `a6bb499375849eec45d68c5ccaebc8865fd422c0` | `sammysun0711/aiter`, branch `integration/mimo-fp4-dflash` · `e595aec16c3c77469ad9d02f2a7c665e5cef1543` |
| Composable Kernel | Gitlink `fdf4bb7fcc984811cef48ce817d89aac064b984a` | Gitlink `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` plus AITER's `mimo_page64_qk192_v128_batch_prefill.patch` |
| PyHIP | Added by the MiMo deployment | AITER's `tingqli/pyhip` dependency, `xiaomi-355-opt` resolved to `db0b1cfc498d470765dc970b44328192fa6ecd6b` |
| FlyDSL | `0.1.5.dev504` | `0.3.2` · release tag `dd8373349e33638c8d158a9ea009b7b28598984a` |
| MoRI | `1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee` | `1.2.4` · release commit `f39fa765e897c95995957d0518908cc982a88fc2` · CPython 3.12 wheel SHA-256 `52dd1ee81e3f0485e4ddc90e99b42dba64ceb2500ee08564a7ba2f9d1a5ef98c` |
| SGLang Model Gateway (`sglang-router`) | `0.3.2`, wheel built from the old SGLang checkout | PyPI `0.3.2`, CPython abi3 Linux x86-64 wheel SHA-256 `ba5a24951e5e0357fe390782fcd1abd861448a87216832dbf2e501a1dd1548eb` |
| Transformers / compressed-tensors | `5.6.0` / `0.15.0.1` | `5.8.1` / `0.16.0` |
| TorchCodec | Added by the MiMo deployment | CPU media decoding, `0.13.0` · release `6e692e9f9f55470cc013914c8cc69ec7ae2eefea` |

The 7.2 deployment guide replaced the baked SGLang and AITER with branch checkouts, installed FlyDSL 0.3.2 and MoRI 1.2.4, and built CPU TorchCodec 0.9.0 (`ce82c7cab6b19f0aeb58c4f85ee990756cfc9f86`) against Torch 2.9. It also used PyAV 17.0.1, Decord 0.6.0 and SoundFile 0.13.1. The branch names in that guide could resolve to different commits on different build dates. [Previous deployment guide](https://github.com/sammysun0711/sglang/blob/9f954b61345ea9fb4deaa273d5c5a0982cd55724/evaluation/mimo_v26_flash/mimo-v2.6-flash-mopd-evaluation-guide.md)

### Which old workspace packages does MiMo need?

The unmodified `rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510` image uses **`/sgl-workspace`**. Alongside `sglang`, `aiter` and `triton-custom`, it contains three checkouts. The single-node EP8 launcher uses **MoRI** for expert dispatch. The image also ships the three HIP components listed below for customers using their respective backends. Two related components installed elsewhere in the old image are included below for customers comparing environments.

| Location in the 7.2 image | Installed version and source | Action for ROCm 7.14.1 |
|---|---|---|
| `/sgl-workspace/mori` | `amd_mori 1.1.1` · `b8af93f856422f508aa7de928f7b19737fdf50ee` | Include **1.2.4** for `--moe-a2a-backend mori` (EP8). EP1 selects `none`. |
| `/sgl-workspace/Mooncake` | Git tag `v0.3.7.post2` · `b6a841dc78c707ec655a563453277d969fb8f38d`. The old image contains native `mooncake.engine` and `mooncake.store` modules and `mooncake_master`. | Included at the same commit, with HIP and etcd support, the transfer engine, distributed store and `mooncake_master`. Select `--disaggregation-transfer-backend mooncake` on both workers for Mooncake KV transfer. |
| `/sgl-workspace/fast-hadamard-transform` | `fast_hadamard_transform 1.0.4.post1` · HIP fork `46efb7d776d38638fc39f3c803eaee3dd7016bd1` | Included at the same commit, compiled against the image’s ROCm PyTorch for clients such as the DSA indexer. |
| `/opt/tilelang` | Installed metadata `0.1.7.post3+cuda.gita55a8230` · `a55a82302bf7f3c5af635b5c9146f728185cc900` | Included at the same commit with `USE_ROCM=ON` and `USE_CUDA=OFF`. The upstream version provider emits the historical `+cuda.gita55a8230` suffix for this CMake build. The Python runtime uses `apache-tvm-ffi==0.1.9`. |
| `/sgl-workspace/sglang/sgl-model-gateway` | `sglang-router 0.3.2`, built from the old SGLang tree at `59faf986b28b1efb5a0733075b627b452b976f1f` | The MiMo image includes the 0.3.2 release wheel. Start the gateway for prefill/decode disaggregation. For ordinary single-node DP serving, clients can use the SGLang HTTP endpoint directly. |

Mooncake is installed by CMake. Check its version with `git -C /sgl-workspace/Mooncake describe --tags --exact-match`. On MI300X, the SGLang Mooncake wrapper transferred 4 KiB between two processes over TCP, and the native master and store completed a 4 KiB put/get. Hadamard matched a CPU reference for FP32 and BF16. TileLang compiled and ran a HIP vector-add with all 256 results correct. Multi-node GPU/RDMA transport requires a separate test on the deployment hosts.


The target base already supplies PyTorch, Triton, torchvision, torchaudio, the gfx942 device libraries and Pillow. Its `hipcc` shim exists even when the development package is missing. The Dockerfile installs `rocm[devel]==7.14.1` from the AMD multi-arch index and makes the SDK visible to both build-time and runtime JIT processes. In this SDK, `rocm-sdk path --cmake` names the **CMake configuration directory** and `rocm-sdk path --root` names its installation prefix. The toolchain stage creates `/opt/rocm` as a symlink directly to that prefix. PyTorch extensions, AITER, MoRI, and PyHIP use this path for builds and runtime JIT. [AMD 7.14.1 development wheels](https://repo.amd.com/rocm/whl-multi-arch/rocm-sdk-devel/)

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