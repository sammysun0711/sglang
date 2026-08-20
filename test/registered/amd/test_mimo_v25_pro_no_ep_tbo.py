"""MiMo-V2.5-Pro TP8 no-EP TBO accuracy and performance gate on MI35x.

The test intentionally shares one expensive server launch. It checks GSM8K
for semantic accuracy and real-MTP acceptance, a fixed-length random workload
for prefill performance, and server logs for proof that all eight ranks entered
the no-EP prefill TBO path while decode used CUDA graphs.

Registry: nightly-amd-8-gpu-mi35x suite.
"""

import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-8-gpu-mi35x",
    nightly=True,
)

MIMO_V25_PRO_MODEL_PATH = os.environ.get(
    "MIMO_V25_PRO_MODEL_PATH", "XiaomiMiMo/MiMo-V2.5-Pro"
)
SERVER_LAUNCH_TIMEOUT = int(os.environ.get("MIMO_TBO_SERVER_TIMEOUT", "3600"))
NUM_GSM8K_EXAMPLES = int(os.environ.get("MIMO_TBO_GSM8K_EXAMPLES", "200"))
NUM_BENCHMARK_PROMPTS = int(os.environ.get("MIMO_TBO_NUM_PROMPTS", "64"))
PERFORMANCE_INPUT_LEN = int(os.environ.get("MIMO_TBO_INPUT_LEN", "16384"))
GSM8K_DATA_PATH = os.environ.get("MIMO_TBO_GSM8K_DATA_PATH")
SHAREGPT_DATASET_PATH = os.environ.get("MIMO_TBO_SHAREGPT_DATASET", "")
SPECULATIVE_STEPS = 3

# Portable MI35x gates account for the observed MI350X/MI355X performance gap.
MIN_GSM8K_ACCURACY = float(os.environ.get("MIMO_TBO_MIN_GSM8K_ACCURACY", "0.95"))
MIN_ACCEPT_LENGTH = float(os.environ.get("MIMO_TBO_MIN_ACCEPT_LENGTH", "3.0"))
MIN_ACCEPT_RATE = float(
    os.environ.get(
        "MIMO_TBO_MIN_ACCEPT_RATE",
        str((MIN_ACCEPT_LENGTH - 1.0) / SPECULATIVE_STEPS),
    )
)
MIN_INPUT_THROUGHPUT = float(
    os.environ.get("MIMO_TBO_MIN_INPUT_THROUGHPUT", "38000")
)
MAX_MEAN_TTFT_MS = float(os.environ.get("MIMO_TBO_MAX_MEAN_TTFT_MS", "1750"))

TBO_LOG_MARKER = "Running MiMo TP8 non-EP prefill TBO"
FATAL_LOG_PATTERNS = (
    r"Scheduler hit an exception",
    r"OutOfMemoryError",
    r"Traceback \(most recent call last\)",
    r"Memory access fault",
    r"Aborted \(core dumped\)",
    r"HSA_STATUS_ERROR",
    r"RCCL.*(?:error|failed|abort)",
    r"watchdog.*(?:timed out|timeout occurred|hang detected)",
)

SERVER_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_MOE_PADDING": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "MC_GID_INDEX": "3",
    "MC_TE_METRIC": "1",
    "SGLANG_SPEC_NAN_DETECTION": "1",
    "SGLANG_SPEC_OOB_DETECTION": "1",
    "SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY": "1",
    "SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE": "1",
    "SGLANG_MIMO_MIXED_ROUTER": "0",
    "SGLANG_FLYPA_MIMO_PREFILL": "1",
    "SGLANG_FLYDSL_MIMO_PREFILL": "1",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    "SGLANG_AITER_KV_CACHE_LAYOUT": "vectorized_5d",
    "SGLANG_AITER_PA_DECODE_IMPL": "flydsl",
    "SGLANG_FLYDSL_PA_NUM_PARTITIONS": "16",
    "SGLANG_MIMO_FUSED_RMS_MOE_QUANT": "1",
    "SGLANG_MIMO_FUSED_RMS_QKV_QUANT": "1",
    "SGLANG_AITER_MIMO_FRESH_BF16_ASM": "1",
    "SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN": "1",
    "SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN": "1",
}

# These knobs can turn the acceptance check into a simulated or lossy path if
# inherited from a developer shell. The gate must always launch real MTP.
SERVER_ENV_TO_UNSET = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "SGLANG_ABLATION_ENABLE_QUICK_REDUCE",
    "SGLANG_SIMULATE_ACC_LEN",
    "SGLANG_SIMULATE_ACC_METHOD",
)


@contextmanager
def temporarily_unset_env(names):
    saved = {name: os.environ[name] for name in names if name in os.environ}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(saved)


class TestMiMoV25ProNoEpTBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MIMO_V25_PRO_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="mimo_no_ep_tbo_gate_")
        cls.stdout_path = Path(cls._tmpdir.name) / "server.stdout.log"
        cls.stderr_path = Path(cls._tmpdir.name) / "server.stderr.log"
        cls.stdout_log = cls.stdout_path.open("w+", buffering=1)
        cls.stderr_log = cls.stderr_path.open("w+", buffering=1)

        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "8",
            "--dp-size",
            "1",
            "--ep-size",
            "1",
            "--moe-a2a-backend",
            "none",
            "--max-running-requests",
            "96",
            "--reasoning-parser",
            "mimo",
            "--tool-call-parser",
            "mimo",
            "--random-seed",
            "655500610",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.01",
            "--context-length",
            "1048576",
            "--chunked-prefill-size",
            "16384",
            "--max-prefill-tokens",
            "1048576",
            "--attention-backend",
            "aiter",
            "--page-size",
            "64",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            str(SPECULATIVE_STEPS),
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--enable-multi-layer-eagle",
            "--enable-two-batch-overlap",
            "--cuda-graph-backend-decode",
            "full",
            "--decode-log-interval",
            "1",
            "--watchdog-timeout",
            "1200",
        ]

        try:
            with temporarily_unset_env(SERVER_ENV_TO_UNSET):
                cls.process = popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=SERVER_LAUNCH_TIMEOUT,
                    other_args=other_args,
                    env=SERVER_ENV,
                    return_stdout_stderr=(cls.stdout_log, cls.stderr_log),
                )
        except BaseException:
            cls.stdout_log.close()
            cls.stderr_log.close()
            cls._tmpdir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        cls.stdout_log.close()
        cls.stderr_log.close()
        cls._tmpdir.cleanup()

    @classmethod
    def _read_server_log(cls):
        cls.stdout_log.flush()
        cls.stderr_log.flush()
        return "\n".join(
            (
                cls.stdout_path.read_text(errors="replace"),
                cls.stderr_path.read_text(errors="replace"),
            )
        )

    def test_accuracy_and_performance_gate(self):
        flush_response = requests.post(self.base_url + "/flush_cache", timeout=60)
        flush_response.raise_for_status()

        accuracy = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=NUM_GSM8K_EXAMPLES,
                num_threads=4,
                num_shots=5,
                gsm8k_data_path=GSM8K_DATA_PATH,
                api="completion",
                max_tokens=512,
                temperature=0.0,
            )
        )

        accuracy_server_info = requests.get(
            self.base_url + "/server_info", timeout=60
        ).json()
        if "decode" in accuracy_server_info:
            accuracy_server_info = accuracy_server_info["decode"][0]
        accuracy_internal_states = accuracy_server_info.get("internal_states") or []
        accept_length = (
            accuracy_internal_states[0].get("avg_spec_accept_length")
            if accuracy_internal_states
            else None
        )
        accept_rate = (
            (accept_length - 1.0) / SPECULATIVE_STEPS
            if accept_length is not None
            else None
        )

        benchmark_args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            dataset_path=SHAREGPT_DATASET_PATH,
            tokenizer=self.model,
            num_prompts=NUM_BENCHMARK_PROMPTS,
            random_input_len=PERFORMANCE_INPUT_LEN,
            random_output_len=1,
            request_rate=float("inf"),
            seed=12345,
            max_concurrency=4,
        )
        benchmark_args.random_range_ratio = 1.0
        benchmark_args.flush_cache = True
        benchmark_args.warmup_requests = 4
        benchmark_args.output_details = True
        benchmark_args.output_file = str(Path(self._tmpdir.name) / "benchmark.jsonl")
        benchmark = run_benchmark(benchmark_args)

        errors = [error for error in benchmark.get("errors", []) if error]
        input_lens = benchmark.get("input_lens", [])

        server_info = benchmark.get("server_info") or {}
        if "decode" in server_info:
            server_info = server_info["decode"][0]
        log_text = self._read_server_log()
        tbo_ranks = sorted(
            set(
                re.findall(
                    rf"\b(TP[0-7])\].*{re.escape(TBO_LOG_MARKER)}", log_text
                )
            )
        )
        decode_graph_states = re.findall(
            r"Decode batch.*cuda graph: (True|False)", log_text
        )
        fatal_markers = [
            pattern
            for pattern in FATAL_LOG_PATTERNS
            if any(
                re.search(pattern, line, re.IGNORECASE)
                for line in log_text.splitlines()
                if "server_args=ServerArgs(" not in line
            )
        ]

        summary = (
            "### MiMo-V2.5-Pro TP8 no-EP TBO gate\n\n"
            "| Metric | Result | Gate |\n"
            "|---|---:|---:|\n"
            f"| GSM8K accuracy ({NUM_GSM8K_EXAMPLES} examples) "
            f"| {accuracy['score']:.3f} | >= {MIN_GSM8K_ACCURACY:.3f} |\n"
            f"| Completed requests | {benchmark['completed']}/{NUM_BENCHMARK_PROMPTS} "
            f"| {NUM_BENCHMARK_PROMPTS}/{NUM_BENCHMARK_PROMPTS} |\n"
            f"| Request errors | {len(errors)} | 0 |\n"
            f"| Real-MTP accept length | {accept_length or 0:.3f} "
            f"| >= {MIN_ACCEPT_LENGTH:.3f} |\n"
            f"| Real-MTP accept rate | {accept_rate or 0:.3f} "
            f"| >= {MIN_ACCEPT_RATE:.3f} |\n"
            f"| Performance input length | {min(input_lens or [0])}–"
            f"{max(input_lens or [0])} | {PERFORMANCE_INPUT_LEN} |\n"
            f"| Input throughput | {benchmark['input_throughput']:.2f} tok/s "
            f"| >= {MIN_INPUT_THROUGHPUT:.2f} tok/s |\n"
            f"| Mean TTFT | {benchmark['mean_ttft_ms']:.2f} ms "
            f"| <= {MAX_MEAN_TTFT_MS:.2f} ms |\n"
            f"| TBO ranks | {', '.join(tbo_ranks) or 'none'} | TP0-TP7 |\n"
            f"| Decode CUDA-graph logs | true={decode_graph_states.count('True')}, "
            f"false={decode_graph_states.count('False')} | true>0, false=0 |\n"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary)

        failures = []
        if accuracy["score"] < MIN_GSM8K_ACCURACY:
            failures.append(
                f"GSM8K accuracy {accuracy['score']:.3f} < {MIN_GSM8K_ACCURACY:.3f}"
            )
        if benchmark["completed"] != NUM_BENCHMARK_PROMPTS:
            failures.append(
                f"completed {benchmark['completed']}/{NUM_BENCHMARK_PROMPTS} requests"
            )
        if errors:
            failures.append(f"benchmark returned {len(errors)} request errors")
        if len(input_lens) != NUM_BENCHMARK_PROMPTS or any(
            input_len != PERFORMANCE_INPUT_LEN for input_len in input_lens
        ):
            failures.append(
                f"performance input lengths were {sorted(set(input_lens))}, "
                f"expected {PERFORMANCE_INPUT_LEN}"
            )
        if accept_length is None:
            failures.append("server did not report avg_spec_accept_length")
        else:
            if accept_length < MIN_ACCEPT_LENGTH:
                failures.append(
                    f"accept length {accept_length:.3f} < {MIN_ACCEPT_LENGTH:.3f}"
                )
            if accept_rate < MIN_ACCEPT_RATE:
                failures.append(
                    f"accept rate {accept_rate:.3f} < {MIN_ACCEPT_RATE:.3f}"
                )
        if benchmark["input_throughput"] < MIN_INPUT_THROUGHPUT:
            failures.append(
                f"input throughput {benchmark['input_throughput']:.2f} < "
                f"{MIN_INPUT_THROUGHPUT:.2f} tok/s"
            )
        if benchmark["mean_ttft_ms"] > MAX_MEAN_TTFT_MS:
            failures.append(
                f"mean TTFT {benchmark['mean_ttft_ms']:.2f} > "
                f"{MAX_MEAN_TTFT_MS:.2f} ms"
            )

        expected_ranks = [f"TP{rank}" for rank in range(8)]
        if tbo_ranks != expected_ranks:
            failures.append(
                f"TBO path ranks were {tbo_ranks}, expected {expected_ranks}"
            )
        if not decode_graph_states or "True" not in decode_graph_states:
            failures.append("no decode CUDA-graph replay was logged")
        if "False" in decode_graph_states:
            failures.append(
                "decode ran without CUDA graph "
                f"{decode_graph_states.count('False')} times"
            )
        if fatal_markers:
            failures.append(f"server log contains fatal markers: {fatal_markers}")

        expected_server_config = {
            "tp_size": 8,
            "dp_size": 1,
            "ep_size": 1,
            "moe_a2a_backend": "none",
            "enable_two_batch_overlap": True,
            "chunked_prefill_size": 16384,
            "cuda_graph_backend_decode": "full",
        }
        for key, expected in expected_server_config.items():
            if server_info.get(key) != expected:
                failures.append(
                    f"server_info[{key!r}]={server_info.get(key)!r}, "
                    f"expected {expected!r}"
                )

        if failures:
            self.fail("MiMo no-EP TBO gate failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    unittest.main()
