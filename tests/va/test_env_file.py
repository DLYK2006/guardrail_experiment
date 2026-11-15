"""Local credential file loading."""

import pytest

from va_experiment.env_file import load_env_file


def test_loads_values_without_overriding_terminal(tmp_path, monkeypatch):
    for name in ("VA_T_A", "VA_T_B", "VA_T_C", "VA_T_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VA_T_C", "from-terminal")
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nVA_T_A=plain\nexport VA_T_B=\"quoted value\"\n"
        "VA_T_C=from-file\nVA_T_EMPTY=\n",
        encoding="utf-8",
    )
    assert load_env_file(path) == ["VA_T_A", "VA_T_B"]
    import os
    assert os.environ["VA_T_A"] == "plain"
    assert os.environ["VA_T_B"] == "quoted value"
    assert os.environ["VA_T_C"] == "from-terminal"
    assert "VA_T_EMPTY" not in os.environ


def test_missing_file_is_ignored_and_bad_line_rejected(tmp_path):
    assert load_env_file(tmp_path / "absent.env") == []
    bad = tmp_path / ".env"
    bad.write_text("not a pair\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_env_file(bad)
