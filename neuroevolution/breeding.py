"""Fitness evaluation, parent selection, crossover, and mutation."""

from collections.abc import Callable
from math import isfinite
from numbers import Real
from random import Random

from .config import Config
from .network import Individual

ScoreFunction = Callable[[Individual], float]


def evaluate(individual: Individual, score: ScoreFunction) -> float:
    return validate_score(score(individual))


def validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("score must return one real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError("score must return a finite number")
    return result


def crossover(first: Individual, second: Individual, method: str, rng: Random) -> tuple[float, ...]:
    if first.layers != second.layers:
        raise ValueError("parents must share the same layer widths")
    a, b = first.genes, second.genes
    if method == "uniform":
        return tuple(rng.choice((left, right)) for left, right in zip(a, b))
    if method == "single_point":
        split = rng.randrange(1, len(a))
        return a[:split] + b[split:]
    if method == "average":
        return tuple(left / 2 + right / 2 for left, right in zip(a, b))
    raise ValueError(f"unknown crossover method: {method}")


def mutate(
    genes: tuple[float, ...], rate: float, bounds: tuple[float, float], rng: Random
) -> tuple[float, ...]:
    low, high = bounds
    return tuple(
        gene + rng.uniform(low, high) if rng.random() < rate else gene
        for gene in genes
    )


def breed(ranked: list[Individual], config: Config, rng: Random) -> list[Individual]:
    parents = ranked[:config.parent_count]
    population = ranked[:config.survivor_count]
    while len(population) < config.population_size:
        first, second = rng.sample(parents, 2)
        genes = crossover(first, second, config.crossover, rng)
        genes = mutate(genes, config.mutation_rate, config.mutation_range, rng)
        population.append(Individual(config.layers, genes))
    return population
