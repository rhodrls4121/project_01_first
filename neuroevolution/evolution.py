"""Run neuroevolution and save the highest-scoring individual."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from random import Random

from .breeding import ScoreFunction, breed, evaluate, validate_score
from .checkpoint import Checkpoint, EvolutionResult, save_checkpoint
from .config import Config
from .network import Individual
from .progress import Progress
from .storage import save_individual


ReportProgress = Callable[[int, int], None]
PopulationScore = Callable[[tuple[Individual, ...], ReportProgress], Iterable[float]]


def evolve(
    config: Config,
    score: ScoreFunction | None = None,
    *,
    population_score: PopulationScore | None = None,
    save_path: str | Path | None = "best_individual.json",
    checkpoint_path: str | Path | None = None,
    resume: Checkpoint | None = None,
    run_id: str | None = None,
) -> EvolutionResult:
    """Evaluate individuals independently or rank an entire population together.

    Population scores are relative to the current opponents, so in that mode
    the saved result is the winner of the final generation. On resume,
    config.generations is the new total. Pass the same evaluator and run_id.
    """
    if (score is None) == (population_score is None):
        raise TypeError("provide exactly one of score or population_score")
    if not callable(score if population_score is None else population_score):
        raise TypeError("the evaluator must be callable")
    if run_id is not None and not isinstance(run_id, str):
        raise TypeError("run_id must be a string or None")
    if (checkpoint_path is not None and save_path is not None
            and Path(checkpoint_path).resolve() == Path(save_path).resolve()):
        raise ValueError("checkpoint and model paths must be different")

    rng = Random(config.seed)
    progress = Progress(config.debug)
    best = None
    first_generation = 1
    if resume is None:
        population = [Individual.random(config.layers, rng) for _ in range(config.population_size)]
    else:
        if replace(resume.config, generations=config.generations, debug=config.debug) != config:
            raise ValueError("resume requires the same training settings except generations and debug")
        if resume.competitive != (population_score is not None) or resume.run_id != run_id:
            raise ValueError("resume requires the same evaluation mode and run_id")
        if config.generations <= resume.generation:
            raise ValueError("target generation must exceed the completed generation")
        first_generation = resume.generation + 1
        best = resume.best
        rng.setstate(resume.rng_state)
        population = breed([individual for _, individual in resume.ranked], config, rng)

    try:
        for generation in range(first_generation, config.generations + 1):
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
            if checkpoint_path is not None:
                save_checkpoint(Checkpoint(
                    config, generation, tuple(evaluated), best, rng.getstate(),
                    population_score is not None, run_id,
                ), checkpoint_path)
            overall_best = best.score if population_score is None else None
            progress.finish(generation, config.generations, scores, overall_best)
            if generation < config.generations:
                population = breed([individual for _, individual in evaluated], config, rng)
    finally:
        progress.clear()
    if save_path is not None:
        save_individual(best.individual, save_path)
    return best
