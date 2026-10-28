#!/usr/bin/env python3
"""Run an owned TP8 server and retain reproducible real-generation evidence."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--port", type=int, default=30106)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--target-variant", choices=("fp8", "mxfp4"))
    parser.add_argument("--method", choices=("dflash", "mtp"), default="dflash")
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--swa-ratio", type=float, default=0.016)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument(
        "--fmoe-config",
        type=Path,
        help="Explicit experimental MoE dispatch configuration; copied and hashed.",
    )
    parser.add_argument(
        "--fp32-lm-head",
        action="store_true",
        help="Use the existing FP32 target head option for numerical isolation.",
    )
    parser.add_argument("--ledger-only", action="store_true")
    parser.add_argument("--fake-prefill", action="store_true")
    parser.add_argument("--graph-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--service-concurrencies", type=int, nargs="+")
    parser.add_argument("--service-input-len", type=int, default=1047488)
    parser.add_argument("--service-output-len", type=int, default=1024)
    parser.add_argument("--service-waves", type=int, default=4)
    parser.add_argument(
        "--real-capacity",
        action="store_true",
        help="Measure real chunked prefill at 64K/256K/near-1M and resident decode separately from fake service.",
    )
    parser.add_argument(
        "--allocation-trace",
        action="store_true",
        help="Report actual draft tensor shapes and bytes through the validation observer.",
    )
    parser.add_argument("--gsm8k-data", type=Path)
    parser.add_argument("--gsm8k-count", type=int)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--replay-prompts",
        type=Path,
        help="Replay saved input_ids for numerical mismatch investigation.",
    )
    parser.add_argument("--replay-indices", type=int, nargs="+")
    parser.add_argument("--replay-repeats", type=int, default=3)
    parser.add_argument("--replay-max-tokens", type=int, default=2048)
    parser.add_argument("--long", action="store_true")
    parser.add_argument(
        "--lifecycle",
        action="store_true",
        help="Exercise in-flight abort, flush and immediate reuse.",
    )
    parser.add_argument(
        "--queued-lifecycle",
        action="store_true",
        help="Fill admission, abort a queued request and disconnect a timed-out client.",
    )
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument(
        "--repeatable-qkv",
        action="store_true",
        help="Use torch BF16 draft QKV GEMM to isolate cache parity from tuned atomic split-K (validation only).",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Compare the compact draft with legacy addresses using the same target features and proposals (eager validation only).",
    )
    parser.add_argument(
        "--shadow-after-warmup",
        action="store_true",
        help="Allocate the validation shadow pool after ordinary server warmup.",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--numerical-trace",
        action="store_true",
        help="Retain target top-two logits and accepted outputs for mismatch diagnosis.",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--materialization", choices=("auto", "sequential", "fused"), default="auto"
    )
    parser.add_argument(
        "--prepare-block", choices=("auto", "eager", "triton"), default="auto"
    )
    args = parser.parse_args()
    if args.method == "mtp" and (
        args.compact
        or args.trace
        or args.numerical_trace
        or args.shadow
        or args.repeatable_qkv
    ):
        parser.error("MTP baseline does not use compact draft or DFlash observers")
    if args.shadow and not (args.compact and args.trace and args.eager):
        parser.error("--shadow requires --compact --trace --eager")
    if args.shadow_after_warmup and not args.shadow:
        parser.error("--shadow-after-warmup requires --shadow")
    if args.fake_prefill and (args.gsm8k_data or args.replay_prompts or args.shadow):
        parser.error(
            "fake-prefill is a separate DFlash synthetic qualification workload"
        )
    root = args.source_root.resolve()
    if args.service_concurrencies and not args.fake_prefill:
        parser.error("service campaign currently requires --fake-prefill")
    if args.real_capacity and (args.fake_prefill or args.ledger_only):
        parser.error("real capacity requires an ordinary real-prefill server")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    env = dict(os.environ)
    if args.target_variant:
        env["TARGET_MODEL_VARIANT"] = args.target_variant
        env["MODEL"] = {
            "fp8": "/models/MiMo-V2.5-Pro",
            "mxfp4": "/models/MiMo-V2.5-Pro-FP4-DFlash",
        }[args.target_variant]
        env.pop("AITER_CONFIG_FMOE", None)
    env.update(
        {
            "SGLANG_ROOT": str(root),
            "PYTHONPATH": f"{root}/python:{env['AITER_ROOT']}",
            "HOST": "127.0.0.1",
            "PORT": str(args.port),
            "LOG_DIR": str(output),
            "MEM_FRACTION_STATIC": str(args.memory_fraction),
            "SWA_FULL_TOKENS_RATIO": str(args.swa_ratio),
            "MAX_RUNNING_REQUESTS": str(args.max_running),
            "CHUNKED_PREFILL_SIZE": "4096",
            "CUDA_GRAPH_BS_DECODE": " ".join(map(str, args.graph_sizes)),
            "SERVER_RANDOM_SEED": "42",
            "ENABLE_TWO_BATCH_OVERLAP": "0",
            "SGLANG_DFLASH_PERFORMANCE_MODE": "0",
            "SPECULATIVE_DFLASH_COMPACT_CACHE": "1" if args.compact else "0",
            "DISABLE_CUDA_GRAPH": "1" if args.eager else "0",
            "ENABLE_FP32_LM_HEAD": "1" if args.fp32_lm_head else "0",
            "DFLASH_VALIDATION_MATERIALIZATION": args.materialization,
            "DFLASH_VALIDATION_PREPARE_BLOCK": args.prepare_block,
            "DFLASH_VALIDATION_SHADOW": "1" if args.shadow else "0",
            "DFLASH_VALIDATION_SHADOW_AFTER_WARMUP": "1"
            if args.shadow_after_warmup
            else "0",
            "DFLASH_VALIDATION_REPEATABLE_QKV": "1" if args.repeatable_qkv else "0",
            "DISABLE_RADIX_CACHE": "1",
            "AITER_MXFP4_STAGE2_OUTPUT_DTYPE": env.get(
                "AITER_MXFP4_STAGE2_OUTPUT_DTYPE", "fp8"
            ),
        }
    )
    if args.fake_prefill:
        env["MM_ATTENTION_BACKEND"] = ""
        env["SGLANG_USE_AITER_UNIFIED_ATTN"] = "0"
        if args.service_concurrencies:
            env["DFLASH_VALIDATION_SERVICE_DIR"] = str(output / "service-observation")
        else:
            env["DFLASH_VALIDATION_FAKE_DIR"] = str(output / "fake-admission")
        if args.method == "mtp":
            if not (args.service_concurrencies or args.ledger_only):
                parser.error(
                    "MTP fake transfer requires a service campaign or ledger-only"
                )
            env["SGLANG_SIMULATE_ACC_LEN"] = "4"
            env["SGLANG_SIMULATE_ACC_METHOD"] = "match-expected"
    if args.real_capacity:
        env["DFLASH_VALIDATION_SERVICE_DIR"] = str(output / "service-observation")
    if args.method == "mtp":
        env["SGLANG_FLYDSL_MIMO_PREFILL"] = "0"
        env["AITER_CONFIG_FMOE"] = str(
            Path(env["AITER_ROOT"])
            / "aiter/configs/model_configs"
            / (
                "mimo_v2_5_pro_tuned_fmoe.csv"
                if env["TARGET_MODEL_VARIANT"] == "mxfp4"
                else "mimo_v2_5_pro_b16_tuned_fmoe.csv"
            )
        )
        env["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] = str(
            Path(env["AITER_ROOT"])
            / "aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_mimo_v2_5_pro.csv"
        )
        env["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"] = str(
            Path(env["AITER_ROOT"])
            / "aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_mimo_v2_5_pro.csv"
        )
    if args.fmoe_config:
        shutil.copyfile(args.fmoe_config, output / "experimental-fmoe.csv")
        env["AITER_CONFIG_FMOE"] = str(output / "experimental-fmoe.csv")
    if args.max_total_tokens:
        env["MAX_TOTAL_TOKENS"] = str(args.max_total_tokens)
    else:
        env.pop("MAX_TOTAL_TOKENS", None)
    if args.trace:
        env["DFLASH_VALIDATION_TRACE_DIR"] = str(output / "draft-traces")
    if args.numerical_trace:
        env["DFLASH_VALIDATION_NUMERICAL_DIR"] = str(output / "numerical-traces")
    if args.allocation_trace:
        env["DFLASH_VALIDATION_ALLOCATION_TRACE"] = "1"
    if (
        args.trace
        or args.numerical_trace
        or args.allocation_trace
        or args.repeatable_qkv
        or args.fake_prefill
        or args.real_capacity
    ):
        env["PYTHONPATH"] += ":" + str(
            Path(__file__).resolve().parent / "compact_trace"
        )
    manifest = {
        "source_root": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "compact": args.compact,
        "requested_memory_fraction": args.memory_fraction,
        "swa_ratio": args.swa_ratio,
        "method": args.method,
        "ledger_only": args.ledger_only,
        "fake_prefill": args.fake_prefill,
        "graph_sizes": args.graph_sizes,
        "service_concurrencies": args.service_concurrencies,
        "service_input_len": args.service_input_len,
        "service_output_len": args.service_output_len,
        "service_waves": args.service_waves,
        "real_capacity": args.real_capacity,
        "context_length": 1048576,
        "max_running_requests": args.max_running,
        "max_total_tokens": args.max_total_tokens,
        "graphs": not args.eager,
        "materialization": args.materialization,
        "prepare_block": args.prepare_block,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shadow": args.shadow,
        "shadow_after_warmup": args.shadow_after_warmup,
        "repeatable_qkv": args.repeatable_qkv,
        "numerical_trace": args.numerical_trace,
        "allocation_trace": args.allocation_trace,
        "fp32_lm_head": args.fp32_lm_head,
        "experimental_fmoe_config": str(args.fmoe_config.resolve())
        if args.fmoe_config
        else None,
        "experimental_fmoe_sha256": hashlib.sha256(
            args.fmoe_config.read_bytes()
        ).hexdigest()
        if args.fmoe_config
        else None,
        "lifecycle": args.lifecycle,
        "queued_lifecycle": args.queued_lifecycle,
        "long": args.long,
        "boundaries": args.boundaries,
        "gsm8k_data": str(args.gsm8k_data.resolve()) if args.gsm8k_data else None,
        "gsm8k_count": args.gsm8k_count,
        "gsm8k_max_tokens": args.gsm8k_max_tokens,
        "replay_prompts": (
            str(args.replay_prompts.resolve()) if args.replay_prompts else None
        ),
        "replay_indices": args.replay_indices,
        "replay_repeats": args.replay_repeats,
        "replay_max_tokens": args.replay_max_tokens,
        "target_model_path": env["MODEL"],
        "target_variant": env["TARGET_MODEL_VARIANT"],
        "stage2_dtype": env.get("AITER_MXFP4_STAGE2_OUTPUT_DTYPE", "fp8"),
        "custom_all_reduce_disabled": env.get("DISABLE_CUSTOM_ALL_REDUCE", "0") == "1",
        "runtime_environment": {
            key: value
            for key, value in env.items()
            if key.startswith(("SGLANG_", "AITER_", "FLYDSL_"))
            or key
            in (
                "GPU_ARCHS",
                "AMDGPU_TARGET",
                "PYTORCH_ROCM_ARCH",
                "NCCL_MIN_NCHANNELS",
                "PYHIP_SIMPLE_GEN_FILENAME",
            )
        },
        "ck_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(env["AITER_ROOT"]) / "3rdparty/composable_kernel"),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "aiter_commit": subprocess.check_output(
            ["git", "-C", env["AITER_ROOT"], "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    manifest["config_hashes"] = {}
    manifest["aiter_config_hashes"] = {}
    for config in (Path(env["AITER_ROOT"]) / "aiter/configs/model_configs").glob(
        "*mimo_v2_5_pro*.csv"
    ):
        manifest["aiter_config_hashes"][str(config)] = hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
    for config in (
        Path(env["MODEL"]) / "config.json",
        Path(
            env["MODEL"] if args.method == "mtp" else env["SPECULATIVE_DRAFT_MODEL"]
        )
        / "config.json",
    ):
        manifest["config_hashes"][str(config)] = hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "source.patch").write_text(
        subprocess.check_output(["git", "-C", str(root), "diff"], text=True)
    )
    (output / "aiter-source.patch").write_text(
        subprocess.check_output(["git", "-C", env["AITER_ROOT"], "diff"], text=True)
    )
    ck_root = Path(env["AITER_ROOT"]) / "3rdparty/composable_kernel"
    (output / "ck-source.patch").write_text(
        subprocess.check_output(["git", "-C", str(ck_root), "diff"], text=True)
    )
    launcher = (
        Path(__file__)
        .resolve()
        .with_name("launch_tp8_noep_aiter_dflash_accuracy_baseline.sh")
    )
    if args.method == "mtp":
        launcher = Path(__file__).resolve().parent.parent / (
            "launch_tp8_noep_aiter_mtp_decode_fake_prefill_baseline.sh"
            if args.fake_prefill
            else "launch_tp8_noep_aiter_mtp_accuracy_baseline.sh"
        )
    elif args.fake_prefill:
        launcher = (
            Path(__file__)
            .resolve()
            .with_name("launch_tp8_noep_aiter_dflash_decode_fake_prefill_baseline.sh")
        )
    shutil.copyfile(launcher, output / "launch-server.sh")
    manifest["launcher_sha256"] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    evaluator = Path(__file__).resolve().with_name("evaluate_dflash_compact_gsm8k.py")
    if args.gsm8k_data:
        shutil.copyfile(evaluator, output / "gsm8k-evaluator.py")
        manifest["gsm8k_evaluator_sha256"] = hashlib.sha256(
            evaluator.read_bytes()
        ).hexdigest()
    if args.fake_prefill:
        fake_evaluator = (
            Path(__file__)
            .resolve()
            .with_name(
                "evaluate_dflash_compact_service.py"
                if args.service_concurrencies
                else "evaluate_dflash_compact_fake.py"
            )
        )
        shutil.copyfile(fake_evaluator, output / "fake-evaluator.py")
        manifest["fake_evaluator_sha256"] = hashlib.sha256(
            fake_evaluator.read_bytes()
        ).hexdigest()
        if args.service_concurrencies:
            benchmark = (
                Path(__file__).resolve().parents[2] / "python/sglang/bench_serving.py"
            )
            shutil.copyfile(benchmark, output / "benchmark-client.py")
            manifest["benchmark_client_sha256"] = hashlib.sha256(
                benchmark.read_bytes()
            ).hexdigest()
    if args.real_capacity:
        capacity_evaluator = Path(__file__).resolve().with_name(
            "evaluate_dflash_compact_real_capacity.py"
        )
        shutil.copyfile(capacity_evaluator, output / "real-capacity-evaluator.py")
        manifest["real_capacity_evaluator_sha256"] = hashlib.sha256(
            capacity_evaluator.read_bytes()
        ).hexdigest()
    new_sources = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "python",
        ],
        text=True,
    ).splitlines()
    manifest["new_source_hashes"] = {}
    for relative in new_sources:
        if not relative.endswith(".py"):
            continue
        source = root / relative
        destination = output / "new-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest["new_source_hashes"][relative] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    if (
        args.trace
        or args.numerical_trace
        or args.allocation_trace
        or args.repeatable_qkv
        or args.fake_prefill
        or args.real_capacity
    ):
        observer = Path(__file__).resolve().parent / "compact_trace/sitecustomize.py"
        shutil.copyfile(observer, output / "validation-observer.py")
        manifest["observer_sha256"] = hashlib.sha256(observer.read_bytes()).hexdigest()
    if args.replay_prompts:
        replay_source = args.replay_prompts.read_bytes()
        shutil.copyfile(args.replay_prompts, output / "replay-prompts.json")
        manifest["replay_prompt_sha256"] = hashlib.sha256(replay_source).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    url = f"http://127.0.0.1:{args.port}"
    session = requests.Session()
    session.trust_env = False
    started = time.monotonic()
    status = {"status": "starting"}
    memory_samples = []
    stop_sampling = threading.Event()
    gpu_paths = [
        p
        for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")
        if (p.parent / "mem_info_vram_total").is_file()
    ]

    def sample_memory():
        while not stop_sampling.is_set():
            sample = {"elapsed_seconds": time.monotonic() - started, "gpus": []}
            for path in gpu_paths:
                total = int((path.parent / "mem_info_vram_total").read_text())
                used = int(path.read_text())
                sample["gpus"].append(
                    {
                        "card": path.parent.parent.name,
                        "unique_id": (path.parent / "unique_id").read_text().strip(),
                        "total_bytes": total,
                        "used_bytes": used,
                        "free_bytes": total - used,
                    }
                )
            memory_samples.append(sample)
            stop_sampling.wait(1)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    with (output / "launcher.log").open("w") as log:
        proc = subprocess.Popen(
            [
                "bash",
                str(launcher),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (output / "launcher.pid").write_text(str(proc.pid))
        print(f"Started {proc.pid}; logs: {output}", flush=True)
        try:
            for _ in range(360):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited with {proc.returncode}")
                try:
                    if session.get(url + "/health", timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("server startup exceeded 30 minutes")
            info_response = session.get(url + "/server_info", timeout=30)
            info_response.raise_for_status()
            info = info_response.json()
            (output / "server-info.json").write_text(json.dumps(info, indent=2))
            assert info["context_length"] == 1048576
            assert info["speculative_algorithm"] == (
                "EAGLE" if args.method == "mtp" else "DFLASH"
            )
            assert (
                info["cuda_graph_config"]["decode"]["backend"] == "disabled"
            ) == args.eager
            assert (
                bool(info.get("speculative_dflash_compact_cache", False))
                == args.compact
            )
            if args.compact:
                owners = args.max_running * (
                    3 if args.fake_prefill and args.max_running <= 32 else 1
                )
                compact_info = info["internal_states"][0]["dflash_compact_cache"]
                assert compact_info["owner_count"] == owners
                assert compact_info["request_table_shape"] == [
                    owners + 1,
                    1152,
                ]
                assert (
                    compact_info["target_full_tokens"] == info["max_total_num_tokens"]
                )
                assert compact_info["minimum_free_vram_bytes"] > 0
            full_tokens = info["max_total_num_tokens"]
            swa_tokens = int(full_tokens * args.swa_ratio) // 64 * 64
            from sglang.srt.speculative.dflash_compact_layout import draft_owner_count

            request_owners = draft_owner_count(
                args.max_running,
                decode=args.fake_prefill,
                preallocated_requests=int(
                    env.get("SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS", "128")
                ),
            )
            components = {
                "target_full_kv_bytes": (full_tokens + 64) * 6400,
                "target_swa_kv_bytes": (swa_tokens + 64) * 38400,
            }
            if args.method == "dflash":
                if args.compact:
                    components["draft_kv_bytes"] = compact_info["kv_bytes"]
                    components["draft_persistent_metadata_bytes"] = compact_info[
                        "persistent_metadata_bytes"
                    ]
                else:
                    components["draft_kv_bytes"] = (full_tokens + 64) * 2560
                    components["draft_request_table_bytes"] = (
                        (request_owners + 1) * (1048576 + 12) * 4
                    )
            else:
                # MiMoV2MTP uses its inherited target SWA capacity. Its NHD
                # attention pads V128 to D192, so price K192 + V192.
                mtp_pools = (
                    info["speculative_num_steps"]
                    if info["enable_multi_layer_eagle"]
                    else 1
                )
                components["mtp_kv_bytes"] = (swa_tokens + 64) * 768 * mtp_pools
            if args.allocation_trace:
                observed = info["internal_states"][0]["validation_draft_allocation"]
                expected = components[
                    "draft_kv_bytes" if args.method == "dflash" else "mtp_kv_bytes"
                ]
                assert observed["kv_tensor_bytes"] == expected, (observed, expected)
                (output / "observed-draft-allocation.json").write_text(
                    json.dumps(observed, indent=2) + "\n"
                )
            (output / "allocation-ledger.json").write_text(
                json.dumps(
                    {
                        "target_full_tokens": full_tokens,
                        "target_swa_tokens": swa_tokens,
                        "requested_memory_fraction": args.memory_fraction,
                        "effective_memory_fraction": info["mem_fraction_static"],
                        "request_owner_count": request_owners,
                        "components": components,
                        "scope": "per-rank tensor geometry; metadata and measured stage residuals are recorded separately in logs/server_info",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Healthy after {time.monotonic() - started:.1f}s", flush=True)
            if args.ledger_only:
                status = {
                    "status": "passed",
                    "elapsed_seconds": time.monotonic() - started,
                    "gate": "startup_only",
                }
                return
            if args.fake_prefill:
                if args.service_concurrencies:
                    from evaluate_dflash_compact_service import evaluate_service

                    report = evaluate_service(url, output, args, env, root)
                else:
                    from evaluate_dflash_compact_fake import (
                        evaluate_fake_prefill,
                        summarize_fake_integration,
                    )

                    report = evaluate_fake_prefill(url, output, args.max_running)
                    summarize_fake_integration(
                        output,
                        compact=args.compact,
                        eager=args.eager,
                        owner_count=request_owners,
                    )
                status = {
                    "status": "passed",
                    "gate": "service_campaign"
                    if args.service_concurrencies
                    else "fake_prefill_integration",
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_requests": report["completed_requests"],
                }
                return
            if args.real_capacity:
                from evaluate_dflash_compact_real_capacity import evaluate_real_capacity

                report = evaluate_real_capacity(url, output, args.max_running)
                status = {
                    "status": "passed",
                    "gate": "real_prefill_capacity",
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_requests": report["completed_requests"],
                }
                return

            def generate(name, input_ids, count, *, sampling=None):
                response = session.post(
                    url + "/generate",
                    json={
                        "rid": name,
                        "input_ids": input_ids,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": count,
                            "ignore_eos": True,
                            **(sampling or {}),
                        },
                    },
                    timeout=900,
                )
                response.raise_for_status()
                result = response.json()
                (output / f"{name}.json").write_text(json.dumps(result, indent=2))
                assert result["meta_info"]["completion_tokens"] == count, result
                assert result["meta_info"]["num_retractions"] == 0, result
                assert result["meta_info"].get("spec_verify_ct", 0) > 0 or bool(
                    sampling and sampling.get("stop_token_ids")
                ), result
                print(
                    f"{name}: {count} tokens, accepted length {result['meta_info'].get('spec_accept_length')}",
                    flush=True,
                )
                return result

            # Fixed vocabulary IDs preserve the input across source snapshots.
            seed_ids = [791, 6864, 315, 9822, 374]
            generate("short", seed_ids, 32)
            if args.long or args.boundaries:
                for length in (
                    (1023, 1024, 1025, 8193, 65536) if args.long else (1023, 1024, 1025)
                ):
                    ids = (seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[
                        :length
                    ]
                    generate(f"prompt-{length}", ids, 64)
            if args.long:
                generate("multiple-windows", seed_ids * 205, 2304)
                for wave in range(4):
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8
                    ) as executor:
                        list(
                            executor.map(
                                lambda i, wave=wave: generate(
                                    f"wave-{wave}-{i}", seed_ids * (i + 1), 24 + i * 8
                                ),
                                range(8),
                            )
                        )
                flush = session.post(url + "/flush_cache", timeout=30)
                flush.raise_for_status()
                assert "Cache flushed." in flush.text
                generate("after-flush", seed_ids, 32)
                # Force termination on a previously observed first target token,
                # then immediately reuse a request slot.
                first_token = generate("eos-seed", seed_ids, 16)["output_ids"][0]
                generate(
                    "early-stop",
                    seed_ids,
                    1,
                    sampling={
                        "max_new_tokens": 64,
                        "stop_token_ids": [first_token],
                        "ignore_eos": False,
                    },
                )
                generate("after-early-stop", seed_ids, 32)
            if args.lifecycle:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        session.post,
                        url + "/generate",
                        json={
                            "rid": "abort-in-flight",
                            "input_ids": seed_ids * 205,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": 8192,
                                "ignore_eos": True,
                            },
                        },
                        timeout=900,
                    )
                    time.sleep(1)
                    response = session.post(
                        url + "/abort_request",
                        json={"rid": "abort-in-flight"},
                        timeout=30,
                    )
                    response.raise_for_status()
                    aborted = future.result(timeout=60)
                    aborted.raise_for_status()
                    value = aborted.json()
                    (output / "abort-response.json").write_text(
                        json.dumps(value, indent=2)
                    )
                    assert value["meta_info"]["finish_reason"]["type"] == "abort", value
                    assert value["meta_info"]["completion_tokens"] < 8192
                flush = session.post(
                    url + "/flush_cache", params={"timeout": 30}, timeout=35
                )
                flush.raise_for_status()
                generate("after-abort", seed_ids, 32)
            if args.queued_lifecycle:
                load_samples = []

                def get_load():
                    response = session.get(
                        url + "/v1/loads", params={"include": "core,queues"}, timeout=10
                    )
                    response.raise_for_status()
                    value = response.json()
                    load_samples.append(value)
                    return value["loads"][0]

                def wait_load(predicate):
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        value = get_load()
                        if predicate(value):
                            return value
                        time.sleep(0.1)
                    raise TimeoutError(
                        "queued lifecycle load condition was not observed"
                    )

                def long_request(rid, timeout=120):
                    with requests.Session() as client:
                        client.trust_env = False
                        return client.post(
                            url + "/generate",
                            json={
                                "rid": rid,
                                "input_ids": seed_ids * 205,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 32768,
                                    "ignore_eos": True,
                                },
                            },
                            timeout=timeout,
                        )

                def abort(rid):
                    response = session.post(
                        url + "/abort_request", json={"rid": rid}, timeout=10
                    )
                    response.raise_for_status()

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.max_running + 1
                ) as executor:
                    held = [
                        executor.submit(long_request, f"queue-hold-{i}")
                        for i in range(args.max_running)
                    ]
                    try:
                        saturated = wait_load(
                            lambda value: (
                                value["num_running_reqs"] > 0
                                and value["num_running_reqs"]
                                + value["num_waiting_reqs"]
                                == args.max_running
                            )
                        )
                        baseline_waiting = saturated["num_waiting_reqs"]
                        queued = executor.submit(long_request, "queue-abort")
                        wait_load(
                            lambda value: value["num_waiting_reqs"] > baseline_waiting
                        )
                        abort("queue-abort")
                        response = queued.result(timeout=30)
                        if response.ok:
                            value = response.json()
                            assert (
                                value["meta_info"]["finish_reason"]["type"] == "abort"
                            )
                            assert value["meta_info"]["completion_tokens"] == 0
                        else:
                            assert response.status_code == 400, response.text
                            assert "abort" in response.text.lower(), response.text
                        (output / "queued-abort.json").write_text(
                            json.dumps(
                                {
                                    "status_code": response.status_code,
                                    "response": response.json(),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        wait_load(
                            lambda value: value["num_waiting_reqs"] == baseline_waiting
                        )
                        timed = executor.submit(long_request, "queue-client-timeout", 1)
                        wait_load(
                            lambda value: value["num_waiting_reqs"] > baseline_waiting
                        )
                        try:
                            timed.result(timeout=5)
                        except requests.ReadTimeout:
                            pass
                        else:
                            raise AssertionError("queued client did not time out")
                        # Allow the server's disconnect watcher to remove the
                        # waiting request, with no manual abort masking a leak.
                        wait_load(
                            lambda value: value["num_waiting_reqs"] == baseline_waiting
                        )
                    finally:
                        # TokenizerManager checks exact live IDs before it
                        # forwards an abort; a prefix alone is not forwarded.
                        for i in range(args.max_running):
                            abort(f"queue-hold-{i}")
                        (output / "queued-lifecycle-loads.json").write_text(
                            json.dumps(load_samples, indent=2) + "\n"
                        )
                    for future in held:
                        response = future.result(timeout=30)
                        response.raise_for_status()
                        value = response.json()
                        assert value["meta_info"]["finish_reason"]["type"] == "abort"
                        meta = value["meta_info"]
                        if "num_retractions" in meta:
                            assert meta["num_retractions"] == 0
                        else:
                            # Abort before prefill returns only tokenizer
                            # metadata; there is no draft owner/allocator use.
                            assert meta["completion_tokens"] == 0, value
                wait_load(
                    lambda value: (
                        value["num_running_reqs"] == value["num_waiting_reqs"] == 0
                    )
                )
                flush = session.post(
                    url + "/flush_cache", params={"timeout": 30}, timeout=35
                )
                flush.raise_for_status()
                generate("after-queued-timeout", seed_ids, 32)
                print(
                    "Queued abort, client timeout cleanup and immediate reuse passed",
                    flush=True,
                )
            if args.gsm8k_data:
                from evaluate_dflash_compact_gsm8k import evaluate_gsm8k

                score = evaluate_gsm8k(
                    url,
                    output / "gsm8k",
                    args.gsm8k_data,
                    env["MODEL"],
                    count=args.gsm8k_count,
                    max_tokens=args.gsm8k_max_tokens,
                )
                print(
                    f"GSM8K score: {score['correct']}/{score['scored_questions']} = {score['score']:.6f}",
                    flush=True,
                )
            if args.replay_prompts:
                replay = output / "replay"
                replay.mkdir()
                prompts = json.loads((output / "replay-prompts.json").read_text())
                requested = set(args.replay_indices) if args.replay_indices else None
                selected = [
                    p
                    for p in prompts
                    if requested is None or p["dataset_index"] in requested
                ]
                if (
                    requested is not None
                    and {p["dataset_index"] for p in selected} != requested
                ):
                    raise ValueError("replay indices missing from saved prompts")
                for repeat in range(args.replay_repeats):
                    for prompt in selected:
                        flush = session.post(
                            url + "/flush_cache", params={"timeout": 30}, timeout=35
                        )
                        flush.raise_for_status()
                        response = session.post(
                            url + "/generate",
                            json={
                                "rid": f"replay-{repeat}-{prompt['dataset_index']}",
                                "input_ids": prompt["input_ids"],
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": args.replay_max_tokens,
                                },
                            },
                            timeout=900,
                        )
                        response.raise_for_status()
                        value = response.json()
                        (
                            replay
                            / f"repeat-{repeat}-question-{prompt['dataset_index']:04d}.json"
                        ).write_text(json.dumps(value, indent=2) + "\n")
                        assert value["meta_info"]["num_retractions"] == 0
                    print(
                        f"Replay wave {repeat + 1}: {len(selected)} prompts completed",
                        flush=True,
                    )
            status = {"status": "passed", "elapsed_seconds": time.monotonic() - started}
        except BaseException as error:
            status = {
                "status": "failed",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
            }
            raise
        finally:
            stop_sampling.set()
            sampler.join(timeout=5)
            (output / "gpu-memory-samples.json").write_text(
                json.dumps(memory_samples, indent=2) + "\n"
            )
            minima = {}
            for sample in memory_samples:
                for gpu in sample["gpus"]:
                    uid = gpu["unique_id"]
                    if (
                        uid not in minima
                        or gpu["free_bytes"] < minima[uid]["free_bytes"]
                    ):
                        minima[uid] = {
                            **gpu,
                            "elapsed_seconds": sample["elapsed_seconds"],
                        }
            (output / "gpu-memory-summary.json").write_text(
                json.dumps(
                    {
                        "sampling_interval_seconds": 1,
                        "per_gpu_minimum_free": list(minima.values()),
                        "minimum_free_bytes": min(
                            (g["free_bytes"] for g in minima.values()), default=None
                        ),
                        "scope": "device VRAM sampled through load, graph capture and workload; subsecond peaks may be lower",
                    },
                    indent=2,
                )
                + "\n"
            )
            (output / "result.json").write_text(json.dumps(status, indent=2))
            try:
                children = psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            print(f"Stopped owned server {proc.pid}", flush=True)


if __name__ == "__main__":
    main()
