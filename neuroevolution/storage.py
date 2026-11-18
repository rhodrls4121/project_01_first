"""Save and reload trained individuals as versioned JSON."""

from collections.abc import Iterable
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .network import Individual


def save_individual(individual: Individual, path: str | Path) -> None:
    data = {"version": 1, "layers": individual.layers, "genes": individual.genes}
    _write_json(data, path)


def _write_json(data: dict, path: str | Path) -> None:
    destination = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, allow_nan=False)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_individual(path: str | Path) -> Individual:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("unsupported individual file format")
    if not isinstance(data.get("layers"), list) or not isinstance(data.get("genes"), list):
        raise ValueError("individual file must contain layers and genes arrays")
    return Individual(tuple(data["layers"]), tuple(data["genes"]))


def run_saved(path: str | Path, inputs: Iterable[float]) -> tuple[float, ...]:
    return load_individual(path)(inputs)
