import json
from pathlib import Path

import pytest

from src.prediction import load_class_names, load_model_if_exists


def test_missing_model_behavior():
    with pytest.raises(FileNotFoundError):
        load_model_if_exists(Path("missing_model.keras"))


def test_class_name_loading_behavior(tmp_path):
    class_names_path = tmp_path / "class_names.json"
    class_names_path.write_text(json.dumps(["COVID", "Normal", "Lung Opacity"]), encoding="utf-8")

    loaded_names = load_class_names(class_names_path)

    assert loaded_names == ["COVID", "Normal", "Lung Opacity"]
