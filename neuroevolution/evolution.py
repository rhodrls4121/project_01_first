"""Run neuroevolution and save the highest-scoring individual."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from .breeding import ScoreFunction, breed, evaluate, validate_score
from .config import Config
from .network import Individual
from .progress import Progress
from .storage import save_individual


ReportProgress = Callable[[int, int], None]
PopulationScore = Callable[[tuple[Individual, ...], ReportProgress], Iterable[float]]


@dataclass(frozen=True)
class EvolutionResult:
    individual: Individual
    score: float
    generation: int


def evolve(
    config: Config,
    score: ScoreFunction | None = None,
    *,
    population_score: PopulationScore | None = None,
    save_path: str | Path | None = "best_individual.json",
) -> EvolutionResult:
    """Evaluate individuals independently or rank an entire population together.

    Population scores are relative to the current opponents, so in that mode
    the saved result is the winner of the final generation.
    """
    if (score is None) == (population_score is None):
        raise TypeError("provide exactly one of score or population_score")
    if not callable(score if population_score is None else population_score):
        raise TypeError("the evaluator must be callable")
    rng = Random(config.seed)
    population = [Individual.random(config.layers, rng) for _ in range(config.population_size)]
    progress = Progress(config.debug)
    best = None
    try:
        for generation in range(1, config.generations + 1):
            if population_score is None:
                scores = []
                progress.update(generation, config.generations, 0, len(population))
                for index, individual in enumerate(population, start=1):
                    scores.append(evaluate(individual, score))
                    progress.update(generation, config.generations, index, len(population))
            else:
                def report(done: int, total: int) -> None:
                    progress.update(generation, config.generations, done, total)

                scores = [validate_score(value) for value in population_score(tuple(population), report)]
                if len(scores) != len(population):
                    raise ValueError("population_score must return one score per individual")
            evaluated = sorted(zip(scores, population), key=lambda entry: entry[0], reverse=True)
            top_score, top_individual = evaluated[0]
            if best is None or population_score is not None or top_score > best.score:
                best = EvolutionResult(top_individual, top_score, generation)
            overall_best = best.score if population_score is None else None
            progress.finish(generation, config.generations, scores, overall_best)
            if generation < config.generations:
                population = breed([individual for _, individual in evaluated], config, rng)
    finally:
        progress.clear()
    if save_path is not None:
        save_individual(best.individual, save_path)
    return best
