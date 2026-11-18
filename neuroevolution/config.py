"""Validate and store settings supplied by the calling program."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Config:
    input_size: int
    hidden_layers: tuple[int, ...]
    output_size: int
    generations: int
    population_size: int
    parent_count: int
    survivor_count: int
    crossover: str
    mutation_rate: float
    mutation_range: tuple[float, float]
    seed: int | None
    debug: bool

    @property
    def layers(self) -> tuple[int, ...]:
        return (self.input_size, *self.hidden_layers, self.output_size)


def configure(
    *,
    input_size: int,
    hidden_layers: tuple[int, ...] | list[int],
    output_size: int,
    generations: int = 100,
    population_size: int = 100,
    parent_count: int = 20,
    survivor_count: int = 2,
    crossover: str = "uniform",
    mutation_rate: float = 0.1,
    mutation_range: tuple[float, float] = (-0.5, 0.5),
    seed: int | None = None,
    debug: bool = True,
) -> Config:
    if not isinstance(hidden_layers, (list, tuple)):
        raise TypeError("hidden_layers must be a list or tuple of widths")
    positive_integers = {
        "input_size": input_size,
        "output_size": output_size,
        "generations": generations,
        "population_size": population_size,
        "parent_count": parent_count,
        **{f"hidden_layers[{i}]": width for i, width in enumerate(hidden_layers)},
    }
    for name, value in positive_integers.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if type(survivor_count) is not int:
        raise TypeError("survivor_count must be an integer")
    if not 2 <= parent_count <= population_size:
        raise ValueError("parent_count must be between 2 and population_size")
    if not 0 <= survivor_count < population_size:
        raise ValueError("survivor_count must be in [0, population_size)")
    if crossover not in ("uniform", "single_point", "average"):
        raise ValueError("crossover must be uniform, single_point, or average")
    if type(mutation_rate) not in (int, float):
        raise TypeError("mutation_rate must be a number")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be in [0, 1]")
    if not isinstance(mutation_range, (list, tuple)) or len(mutation_range) != 2:
        raise TypeError("mutation_range must contain two numbers")
    low, high = mutation_range
    if any(type(value) not in (int, float) for value in (low, high)):
        raise TypeError("mutation_range must contain two numbers")
    if not (isfinite(low) and isfinite(high) and low <= high):
        raise ValueError("mutation_range must contain finite, ordered bounds")
    if seed is not None and type(seed) is not int:
        raise TypeError("seed must be an integer or None")
    if type(debug) is not bool:
        raise TypeError("debug must be a boolean")
    return Config(
        input_size, tuple(hidden_layers), output_size, generations, population_size,
        parent_count, survivor_count, crossover, mutation_rate, (low, high), seed, debug,
    )
