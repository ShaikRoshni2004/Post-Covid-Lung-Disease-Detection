import json
from pathlib import Path
from typing import Any, List


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_image_files(directory: str | Path) -> List[Path]:
    """Return image files in a directory with supported extensions."""
    directory = Path(directory)
    if not directory.exists():
        return []

    supported_extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        ]
    )


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON data to disk."""
    target_path = Path(path)
    ensure_directory(target_path.parent)
    with target_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def read_json(path: str | Path) -> Any:
    """Read JSON from disk if it exists."""
    target_path = Path(path)
    if not target_path.exists():
        return None
    with target_path.open("r", encoding="utf-8") as file:
        return json.load(file)
