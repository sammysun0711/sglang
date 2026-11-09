"""CPU-only checks for automatic and explicit PA partition selection."""

import ast
import os
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-b-test-cpu")
source = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/attention/aiter_utils.py"
tree = ast.parse(source.read_text())
constants = {"FLYDSL_MIMO_DEFAULT_NUM_PARTITIONS", "FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS", "FLYDSL_MIMO_NUM_PARTITIONS_ENV"}
nodes = [node for node in tree.body if (
    isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
) or (isinstance(node, ast.FunctionDef) and node.name == "get_flydsl_mimo_num_partitions")]
namespace = {"os": os}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
get_flydsl_mimo_num_partitions = namespace["get_flydsl_mimo_num_partitions"]


def test_default_uses_automatic_selection(monkeypatch):
    monkeypatch.delenv("SGLANG_FLYDSL_PA_NUM_PARTITIONS", raising=False)
    assert get_flydsl_mimo_num_partitions() == 0


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4, 8, 16, 24, 32, 64])
def test_supported_partition_override(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FLYDSL_PA_NUM_PARTITIONS", str(value))
    assert get_flydsl_mimo_num_partitions() == value


@pytest.mark.parametrize("value", ["-1", "48", "65", "16.0", "invalid"])
def test_invalid_partition_override_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FLYDSL_PA_NUM_PARTITIONS", value)
    with pytest.raises(ValueError):
        get_flydsl_mimo_num_partitions()
