from pathlib import Path
from typing import Any, Union
import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")

    return config
