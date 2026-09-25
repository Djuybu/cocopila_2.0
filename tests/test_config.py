from pathlib import Path
import pytest
import yaml
from src.utils.config import load_config, run_directory


def test_paths_resolved_from_declaring_config(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "base.yaml").write_text("data:\n  chunks_path: input.json\nretrieval:\n  top_k: 5\n")
    (child / "run.yaml").write_text("extends: ../parent/base.yaml\nretrieval:\n  top_k: 2\n")
    result = load_config(child / "run.yaml")
    assert result["data"]["chunks_path"] == str(parent / "input.json")
    assert result["retrieval"]["top_k"] == 2


def test_config_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("name", ["../escape", "/tmp/path", "", "..", "with space"])
def test_unsafe_run_id(name):
    with pytest.raises(ValueError):
        run_directory({"run_name": name, "output_dir": "outputs"})


def test_shipped_configs_load():
    for path in Path("configs").rglob("*.yaml"):
        assert isinstance(load_config(path), dict)
