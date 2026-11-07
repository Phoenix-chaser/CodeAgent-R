# -*- coding: utf-8 -*-
"""P00 smoke test：不依赖任何模型/API，验证包可导入、CLI 与配置加载真实可用。"""
import subprocess
import sys

import pytest

from agentlab import __version__
from agentlab.config import ConfigError, load_config


def test_package_importable_with_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize("module", ["agentlab", "agentlab.cli"])
def test_cli_help_lists_only_implemented_commands(module):
    """未实现的 run/train 不允许出现在帮助里（防伪装完成）。"""
    out = subprocess.run([sys.executable, "-m", module, "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for banned in ("run", "train", "benchmark", "trajectories"):
        assert banned not in out.stdout, f"未实现命令 {banned} 不得出现在帮助中"
    assert "doctor" in out.stdout


@pytest.mark.parametrize("module", ["agentlab", "agentlab.cli"])
def test_cli_doctor_reports_unknown_for_unprobeable(module):
    out = subprocess.run([sys.executable, "-m", module, "doctor"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "model_revision" in out.stdout
    assert "unknown" in out.stdout  # 未探测能力必须保留 unknown，不得伪造


def test_config_roundtrip_and_validation(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "schema_version: 1\n"
        "project_name: CodeAgent-R\n"
        "execution_limits: {mode: offline, gpu_job_authorized: false}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.project_name == "CodeAgent-R"
    assert cfg.execution_limits.mode == "offline"

    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\nproject_name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)

    missing = tmp_path / "missing.yaml"
    missing.write_text("project_name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(missing)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        "schema_version: 1\nproject_name: x\n"
        "execution_limits: {mode: offline, turbo: true}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(unknown)
