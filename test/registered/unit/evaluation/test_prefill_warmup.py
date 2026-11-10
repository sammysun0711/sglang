"""CPU-only integration checks for the prefill benchmark warmup boundary."""

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "evaluation/run_benchmark_mimo_pro_prefill.sh"
HELPER = ROOT / "evaluation/warmup_mimo_prefill.py"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.events.append(("info", None))
        self.reply({"context_length": self.server.context_length})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        lengths = [len(ids) for ids in body["input_ids"]]
        self.server.events.append(("generate", lengths))
        self.reply(
            [
                {
                    "meta_info": {
                        "prompt_tokens": length,
                        "completion_tokens": 0 if self.server.fail else 1,
                        "finish_reason": {"type": "length"},
                    }
                }
                for length in lengths
            ]
        )


class PrefillWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.calls = self.directory / "calls.jsonl"
        self.logs = self.directory / "logs"
        self.server.events = []
        self.server.fail = False
        self.server.context_length = 1048576
        self.python = self.directory / "record-python"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, subprocess, sys\n"
            "with open(os.environ['CALLS_FILE'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1].endswith('warmup_mimo_prefill.py'):\n"
            "    sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
        )
        self.python.chmod(0o755)

    def invoke(self, **overrides):
        env = dict(
            os.environ,
            PYTHON_BIN=str(self.python),
            CALLS_FILE=str(self.calls),
            BENCHMARK_PRESET="simple",
            TOKEN_LIST_OVERRIDE="4096 8192",
            NUM_PROMPTS_OVERRIDE="2",
            CONCURRENCY_LIST_OVERRIDE="2",
            HOST="127.0.0.1",
            PORT=str(self.server.server_port),
            LOG_DIR=str(self.logs),
            DRY_RUN="0",
            WARMUP_MODE="representative",
        )
        env.pop("WARMUP_REQUESTS_OVERRIDE", None)
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=env, text=True, capture_output=True
        )
        calls = (
            [json.loads(line) for line in self.calls.read_text().splitlines()]
            if self.calls.exists()
            else []
        )
        return result, calls

    def test_representative_warmup_finishes_before_timed_cases(self):
        result, calls = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0][0].endswith("warmup_mimo_prefill.py"))
        self.assertEqual(
            [call[call.index("--warmup-requests") + 1] for call in calls[1:]],
            ["0", "0"],
        )
        batches = [
            lengths for event, lengths in self.server.events if event == "generate"
        ]
        self.assertEqual(sum(map(len, batches)), 32)
        self.assertEqual(len(batches), 16)
        self.assertGreater(len({length for batch in batches for length in batch}), 1)
        report = json.loads((self.logs / "prefill_warmup.json").read_text())
        self.assertEqual((report["status"], report["completed"]), ("complete", 32))

    def test_dry_run_has_no_requests_or_logs(self):
        result, calls = self.invoke(DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(calls)
        self.assertFalse(self.server.events)
        self.assertFalse(self.logs.exists())
        self.assertEqual(result.stdout.count("Warmup command:"), 1)

    def test_uniform_and_disabled_warmup(self):
        for mode, count in (("uniform", "4"), ("representative", "0")):
            with self.subTest(mode=mode):
                self.calls.unlink(missing_ok=True)
                result, calls = self.invoke(
                    WARMUP_MODE=mode, WARMUP_REQUESTS_OVERRIDE=count
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    [c[c.index("--warmup-requests") + 1] for c in calls], [count, count]
                )
                self.assertFalse(self.server.events)

    def test_warmup_failure_stops_before_benchmark(self):
        self.server.fail = True
        result, calls = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        report = json.loads((self.logs / "prefill_warmup.json").read_text())
        self.assertEqual(report["status"], "failed")

    def test_pr46_warmup_policy_and_explicit_overrides(self):
        for override, expected in (
            (None, ["16", "4"]),
            ("3", ["3", "3"]),
            ("0", ["0", "0"]),
        ):
            with self.subTest(override=override):
                self.calls.unlink(missing_ok=True)
                options = dict(WARMUP_MODE="pr46", BENCHMARK_PRESET="customer_pressure")
                if override is not None:
                    options["WARMUP_REQUESTS_OVERRIDE"] = override
                result, calls = self.invoke(**options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 2)
                self.assertEqual(
                    [c[c.index("--warmup-requests") + 1] for c in calls], expected
                )
                self.assertFalse(self.server.events)

    def test_invalid_matrix_does_not_start_warmup(self):
        result, calls = self.invoke(TOKEN_LIST_OVERRIDE="invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertFalse(self.server.events)
        self.assertFalse(self.logs.exists())

    def test_context_limit_and_odd_request_count(self):
        self.server.context_length = 128
        result, calls = self.invoke(WARMUP_REQUESTS_OVERRIDE="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        batches = [
            lengths for event, lengths in self.server.events if event == "generate"
        ]
        self.assertEqual(batches, [[127, 127], [127]])


if __name__ == "__main__":
    unittest.main()
