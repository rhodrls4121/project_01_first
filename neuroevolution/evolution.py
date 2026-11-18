"""Run neuroevolution and save the highest-scoring individual."""

from dataclasses import dataclass
from pathlib import Path
from random import Random

from .breeding import ScoreFunction, breed, evaluate
from .config import Config
from .network import Individual
from .progress import Progress
from .storage import save_individual


@dataclass(frozen=True)
class EvolutionResult:
    individual: Individual
    score: float
    generation: int


def evolve(
    config: Config, score: ScoreFunction, *, save_path: str | Path | None = "best_individual.json"
) -> EvolutionResult:
    if not callable(score):
        raise TypeError("score must be callable")
    rng = Random(config.seed)
    population = [Individual.random(config.layers, rng) for _ in range(config.population_size)]
    progress = Progress(config.debug)
    best = None
    try:
        for generation in range(1, config.generations + 1):
            evaluated = []
            progress.update(generation, config.generations, 0, len(population))
            for index, individual in enumerate(population, start=1):
                evaluated.append((evaluate(individual, score), individual))
                progress.update(generation, config.generations, index, len(population))
            evaluated.sort(key=lambda entry: entry[0], reverse=True)
            top_score, top_individual = evaluated[0]
            if best is None or top_score > best.score:
                best = EvolutionResult(top_individual, top_score, generation)
            progress.finish(generation, config.generations, [value for value, _ in evaluated], best.score)
            if generation < config.generations:
                population = breed([individual for _, individual in evaluated], config, rng)
    finally:
        progress.clear()
    if save_path is not None:
        save_individual(best.individual, save_path)
    return best
