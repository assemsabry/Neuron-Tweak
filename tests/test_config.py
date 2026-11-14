from pathlib import Path

import pytest

from neuron_tweak.config import (
    ConfigError,
    config_value,
    load_config,
    resolve_config_path,
)
from neuron_tweak.main import apply_config, build_parser


def test_load_config_reads_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'profile = "identity"\n\n[model]\ndtype = "float32"\n\n'
        "[prompts]\nevaluation_size = 42\nsplit_seed = 7\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["profile"] == "identity"
    assert config["model"]["dtype"] == "float32"
    assert config_value(config, "prompts", "evaluation_size") == 42
    assert config_value(config, "missing", default="fallback") == "fallback"


def test_load_config_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")


def test_load_config_reports_invalid_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("profile = [", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_strips_utf8_bom(tmp_path):
    path = tmp_path / "bom.toml"
    path.write_bytes(b"\xef\xbb\xbfprofile = \"safety\"\n")

    assert load_config(path)["profile"] == "safety"


def test_apply_config_fills_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'profile = "identity"\n\n[model]\nquantization = "none"\n\n'
        "[prompts]\nevaluation_size = 42\nsplit_seed = 7\n",
        encoding="utf-8",
    )
    parser = build_parser()
    argv = ["some-model", "--config", str(path)]
    args = parser.parse_args(argv)

    apply_config(args, argv)

    assert args.profile == "identity"
    assert args.evaluation_size == 42
    assert args.seed == 7
    assert args.quantization == "none"


def test_apply_config_respects_cli_flags(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'profile = "identity"\n\n[prompts]\nevaluation_size = 42\n',
        encoding="utf-8",
    )
    parser = build_parser()
    argv = [
        "some-model",
        "--config",
        str(path),
        "--profile",
        "safety",
        "--evaluation-size",
        "5",
    ]
    args = parser.parse_args(argv)

    apply_config(args, argv)

    assert args.profile == "safety"
    assert args.evaluation_size == 5


def test_apply_config_rejects_invalid_profile(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('profile = "nonsense"\n', encoding="utf-8")
    parser = build_parser()
    argv = ["some-model", "--config", str(path)]
    args = parser.parse_args(argv)

    with pytest.raises(ConfigError):
        apply_config(args, argv)


def test_resolve_config_path_prefers_local_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None) is None

    (tmp_path / "config.toml").write_text(
        'profile = "safety"\n', encoding="utf-8"
    )

    assert resolve_config_path(None) == Path("config.toml")
