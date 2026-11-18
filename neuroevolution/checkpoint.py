"""Persist a completed generation so breeding can resume from it."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from random import Random

from .breeding import validate_score
from .config import Config, configure
from .network import Individual
from .storage import _write_json


@dataclass(frozen=True)
class EvolutionResult:
    individual: Individual
    score: float
    generation: int


@dataclass(frozen=True)
class Checkpoint:
    config: Config
    generation: int
    ranked: tuple[tuple[float, Individual], ...]
    best: EvolutionResult
    rng_state: tuple
    competitive: bool
    run_id: str | None


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    _write_json({
        "kind": "neuroevolution-checkpoint",
        "version": 1,
        "config": asdict(checkpoint.config),
        "generation": checkpoint.generation,
        "ranked": [(score, item.genes) for score, item in checkpoint.ranked],
        "best": {
            "generation": checkpoint.best.generation,
            "score": checkpoint.best.score,
            "genes": checkpoint.best.individual.genes,
        },
        "rng_state": checkpoint.rng_state,
        "competitive": checkpoint.competitive,
        "run_id": checkpoint.run_id,
    }, path)


def load_checkpoint(path: str | Path) -> Checkpoint:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("kind") != "neuroevolution-checkpoint"
            or type(data.get("version")) is not int or data["version"] != 1):
        raise ValueError("unsupported checkpoint format")
    try:
        config = configure(**data["config"])
        generation = data["generation"]
        if type(generation) is not int or not 1 <= generation <= config.generations:
            raise ValueError("invalid completed generation")
        ranked = tuple(
            (validate_score(score), Individual(config.layers, tuple(genes)))
            for score, genes in data["ranked"]
        )
        if len(ranked) != config.population_size:
            raise ValueError("population size does not match the settings")
        if any(left[0] < right[0] for left, right in zip(ranked, ranked[1:])):
            raise ValueError("population is not in descending score order")
        record = data["best"]
        if type(record["generation"]) is not int or not 1 <= record["generation"] <= generation:
            raise ValueError("invalid best generation")
        best = EvolutionResult(
            Individual(config.layers, tuple(record["genes"])),
            validate_score(record["score"]), record["generation"],
        )
        version, internal, gaussian = data["rng_state"]
        rng_state = (version, tuple(internal), gaussian)
        Random().setstate(rng_state)
        competitive, run_id = data["competitive"], data["run_id"]
        if type(competitive) is not bool or (run_id is not None and not isinstance(run_id, str)):
            raise ValueError("invalid evaluation settings")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid checkpoint: {error}") from error
    return Checkpoint(config, generation, ranked, best, rng_state, competitive, run_id)
